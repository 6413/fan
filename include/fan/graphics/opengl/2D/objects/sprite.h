#pragma once

#include _FAN_PATH(graphics/opengl/gl_core.h)
#include _FAN_PATH(graphics/opengl/gl_shader.h)
#include _FAN_PATH(graphics/shared_graphics.h)
#include _FAN_PATH(physics/collision/rectangle.h)
#include _FAN_PATH(graphics/opengl/texture_pack.h)

namespace fan_2d {
  namespace opengl {

    struct sprite_t {

      using draw_cb_t = void(*)(fan::opengl::context_t* context, sprite_t*, void*);

      struct instance_t {
        fan::vec3 position = 0;
      private:
        f32_t pad;
      public:
        fan::vec2 size = 0;
        fan::vec2 rotation_point = 0;
        fan::color color = fan::colors::white;
        fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
        f32_t angle = 0;
        fan::vec2 tc_position = 0;
        fan::vec2 tc_size = 1;
      };

      struct block_properties_t {
        fan::opengl::image_t image;
      };

      struct properties_t : public instance_t {
        using type_t = sprite_t;

        union {
          struct {
            fan::opengl::image_t image;
          };
          block_properties_t block_properties;
        };

        void load_texturepack(fan::opengl::context_t* context, fan::opengl::texturepack* texture_packd, fan::tp::ti_t* ti) {
          image = texture_packd->pixel_data_list[ti->pack_id].image;
          const fan::vec2 texture_position = fan::cast<f32_t>(ti->position) / image.size;
          const fan::vec2 texture_size = fan::cast<f32_t>(ti->size) / image.size;
          instance_t::tc_position = texture_position;
          instance_t::tc_size = texture_size;
        }
        void load_texturepack(fan::opengl::context_t* context, fan::tp::texture_packe0* texture_packd, fan::tp::ti_t* ti) {
          image.load(context, texture_packd->get_pixel_data(ti->pack_id));
          const fan::vec2 texture_position = fan::cast<f32_t>(ti->position) / image.size;
          const fan::vec2 texture_size = fan::cast<f32_t>(ti->size) / image.size;
          instance_t::tc_position = texture_position;
          instance_t::tc_size = texture_size;
        }
      };

      static constexpr uint32_t max_instance_size = 128;

      struct id_t{
        id_t(fan::opengl::cid_t* cid) {
          block = cid->id / max_instance_size;
          instance = cid->id % max_instance_size;
        }
        uint32_t block;
        uint32_t instance;
      };

      void open(fan::opengl::context_t* context) {
        m_shader.open(context);

        m_shader.set_vertex(
          context,
#include _FAN_PATH(graphics/glsl/opengl/2D/objects/sprite.vs)
        );

        m_shader.set_fragment(
          context,
#include _FAN_PATH(graphics/glsl/opengl/2D/objects/sprite.fs)
        );

        m_shader.compile(context);

        blocks.open();

        m_draw_node_reference = fan::uninitialized;
      }
      void close(fan::opengl::context_t* context) {
        m_shader.close(context);
        for (uint32_t i = 0; i < blocks.size(); i++) {
          blocks[i].uniform_buffer.close(context);
        }
        blocks.close();
      }

      void enable_draw(fan::opengl::context_t* context) {

#if fan_debug >= fan_debug_low
        if (m_draw_node_reference != fan::uninitialized) {
          fan::throw_error("trying to call enable_draw twice");
        }
#endif

        m_draw_node_reference = context->enable_draw(this, [](fan::opengl::context_t* c, void* d) { ((decltype(this))d)->draw(c); });
      }
      void disable_draw(fan::opengl::context_t* context) {
#if fan_debug >= fan_debug_low
        if (m_draw_node_reference == fan::uninitialized) {
          fan::throw_error("trying to disable unenabled draw call");
        }
#endif
        context->disable_draw(m_draw_node_reference);
      }

      void set_vertex(fan::opengl::context_t* context, const std::string& str) {
        m_shader.set_vertex(context, str);
      }
      void set_fragment(fan::opengl::context_t* context, const std::string& str) {
        m_shader.set_fragment(context, str);
      }
      void compile(fan::opengl::context_t* context) {
        m_shader.compile(context);
      }

      void draw(fan::opengl::context_t* context) {
        m_shader.use(context);

        draw_cb(context, this, draw_userdata);

        uint32_t texture_id = fan::uninitialized;
        for (uint32_t block_id = 0; block_id < blocks.size(); block_id++) {
          blocks[block_id].uniform_buffer.bind_buffer_range(context, blocks[block_id].uniform_buffer.size());

          uint32_t from = 0;
          uint32_t to = 0;

          for (uint32_t i = 0; i < blocks[block_id].uniform_buffer.size(); i++) {
            if (texture_id != *blocks[block_id].p[i].image.get_texture(context)) {
              if (to) {
                blocks[block_id].uniform_buffer.draw(
                  context,
                  from * 6,
                  to * 6
                );
              }
              from = i;
              to = 0;
              texture_id = *blocks[block_id].p[i].image.get_texture(context);
              m_shader.set_int(context, "texture_sampler", 0);
              context->opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
              context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, texture_id);
            }
            to++;
          }

          if (to) {
            blocks[block_id].uniform_buffer.draw(
              context,
              from * 6,
              to * 6
            );
          }
        }
      }

      template <typename T>
      T get(fan::opengl::context_t* context, const id_t& id, T instance_t::*member) {
        return blocks[id.block].uniform_buffer.get_instance(context, id.instance)->*member;
      }
      template <typename T, typename T2>
      void set(fan::opengl::context_t* context, const id_t& id, T instance_t::*member, const T2& value) {
        blocks[id.block].uniform_buffer.edit_ram_instance(context, id.instance, (T*)&value, fan::ofof<instance_t, T>(member), sizeof(T));
        blocks[id.block].uniform_buffer.common.edit(
          context,
          id.instance * sizeof(instance_t) + fan::ofof<instance_t, T>(member),
          id.instance * sizeof(instance_t) + fan::ofof<instance_t, T>(member) + sizeof(T)
        );
      }

      fan::shader_t m_shader;

      struct block_t {
        fan::opengl::core::uniform_block_t<instance_t, max_instance_size> uniform_buffer;
        fan::opengl::cid_t* cid[max_instance_size];
        block_properties_t p[max_instance_size];
      };
      uint32_t m_draw_node_reference;

      fan::hector_t<block_t> blocks;

      draw_cb_t draw_cb;
      void* draw_userdata;
    };
  }
}