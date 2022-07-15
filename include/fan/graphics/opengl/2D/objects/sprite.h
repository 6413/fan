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
        fan::vec2 position = 0;
        fan::vec2 size = 0;
        fan::color color = fan::colors::white;
        fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
        f32_t angle = 0;
        fan::vec2 rotation_point = 0;
        fan::vec2 tc_position = 0;
        fan::vec2 tc_size = 1;
        fan::vec2 pad;
      };

      struct properties_t : public instance_t {
        using type_t = sprite_t;

        fan::opengl::image_t image;

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
        draw_cb = [](fan::opengl::context_t* context, sprite_t*, void*) {};
      }
      void close(fan::opengl::context_t* context) {
        m_shader.close(context);
        for (uint32_t i = 0; i < blocks.size(); i++) {
          blocks[i].uniform_buffer.close(context);
        }
        blocks.close();
      }

      void push_back(fan::opengl::context_t* context, fan::opengl::cid_t* cid, const properties_t& p) {
        instance_t it = p;

        uint32_t i = 0;

        for (; i < blocks.size(); i++) {
          if (blocks[i].uniform_buffer.size() != max_instance_size) {
            break;
          }
        }

        if (i == blocks.size()) {
          blocks.push_back({});
          blocks[i].uniform_buffer.open(context);
          blocks[i].uniform_buffer.bind_uniform_block(context, m_shader.id, "instance_t");
        }

        blocks[i].uniform_buffer.push_ram_instance(context, it);

        const uint32_t instance_id = blocks[i].uniform_buffer.size() - 1;

        blocks[i].cid[instance_id] = cid;

        blocks[i].uniform_buffer.common.edit(
          context,
          instance_id,
          instance_id + 1
        );

        cid->id = i * max_instance_size + instance_id;

        if (p.image.texture != fan::uninitialized) {
          blocks[i].image[blocks[i].uniform_buffer.size() - 1] = p.image;
        }
      }
      void erase(fan::opengl::context_t* context, fan::opengl::cid_t* cid) {

        uint32_t block_id = cid->id / max_instance_size;
        uint32_t instance_id = cid->id % max_instance_size;

        
        #if fan_debug >= fan_debug_medium
        if (block_id >= blocks.size()) {
          fan::throw_error("invalid access");
        }
        if (instance_id >= blocks[block_id].uniform_buffer.size()) {
          fan::throw_error("invalid access");
        }
        #endif

        if (block_id == blocks.size() - 1 && instance_id == blocks.ge()->uniform_buffer.size() - 1) {
          blocks[block_id].uniform_buffer.common.m_size -= blocks[block_id].uniform_buffer.common.buffer_bytes_size;
          if (blocks[block_id].uniform_buffer.size() == 0) {
            blocks[block_id].uniform_buffer.close(context);
            blocks.m_size -= 1;
          }
          return;
        }

        uint32_t last_block_id = blocks.size() - 1;
        uint32_t last_instance_id = blocks[last_block_id].uniform_buffer.size() - 1;

        instance_t* last_instance_data = blocks[last_block_id].uniform_buffer.get_instance(context, last_instance_id);

        blocks[block_id].uniform_buffer.edit_ram_instance(
          context,
          instance_id,
          last_instance_data,
          0,
          sizeof(instance_t)
        );


      /*  blocks[block_id].uniform_buffer.common.edit(
          context,
          instance_id,
          instance_id + 1
        );*/

        blocks[last_block_id].uniform_buffer.common.m_size -= blocks[block_id].uniform_buffer.common.buffer_bytes_size;

        blocks[block_id].image[instance_id] = blocks[last_block_id].image[last_instance_id];

        blocks[block_id].cid[instance_id] = blocks[last_block_id].cid[last_instance_id];
        blocks[block_id].cid[instance_id]->id = block_id * max_instance_size + instance_id;

        if (blocks[last_block_id].uniform_buffer.size() == 0) {
          blocks[last_block_id].uniform_buffer.close(context);
          blocks.m_size -= 1;
        }
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

      void set_draw_cb(fan::opengl::context_t* context, draw_cb_t draw_cb_, void* userptr = 0) {
        draw_cb = draw_cb_;
        if (userptr != nullptr) {
          draw_userdata = userptr;
        }
      }
      void set_draw_cb_userptr(fan::opengl::context_t* context, void* userptr) {
        draw_userdata = userptr;
      }

      void draw(fan::opengl::context_t* context) {
        m_shader.use(context);

        draw_cb(context, this, draw_userdata);

        //context->opengl.glActiveTexture(fan::opengl::GL_TEXTURE1);
        //context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, light_map_id); /* TODO get by magic */

        uint32_t texture_id = fan::uninitialized;
        for (uint32_t block_id = 0; block_id < blocks.size(); block_id++) {
          blocks[block_id].uniform_buffer.bind_buffer_range(context, blocks[block_id].uniform_buffer.size());

          /* TODO what is this */
          //m_shader.set_int(context, "texture_light_map", 1);

          uint32_t from = 0;
          uint32_t to = 0;

          for (uint32_t i = 0; i < blocks[block_id].uniform_buffer.size(); i++) {
            if (texture_id != blocks[block_id].image[i].texture) {
              if (to) {
                blocks[block_id].uniform_buffer.draw(
                  context,
                  from, // TODO may need multiple with something
                  to * 6
                );
              }
              from = i;
              to = 0;
              texture_id = blocks[block_id].image[i].texture;
              m_shader.set_int(context, "texture_sampler", 0);
              context->opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
              context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, texture_id);
            }
            to++;
          }

          if (to) {
            blocks[block_id].uniform_buffer.draw(
              context,
              from, // TODO may need multiple with something
              to * 6
            );
          }
        }
      }

      void bind_matrices(fan::opengl::context_t* context, fan::opengl::matrices_t* matrices) {
        m_shader.bind_matrices(context, matrices);
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
          id.instance,
          id.instance + 1
        );
      }

      //uint32_t size() const {
      //  return blocks.size();
      //}

      fan::shader_t m_shader;

      struct block_t {
        fan::opengl::core::uniform_block_t<instance_t, max_instance_size> uniform_buffer;
        fan::opengl::image_t image[max_instance_size];
        fan::opengl::cid_t* cid[max_instance_size];
      };
      uint32_t m_draw_node_reference;

      fan::hector_t<block_t> blocks;

      draw_cb_t draw_cb;
      void* draw_userdata;
    };
  }
}