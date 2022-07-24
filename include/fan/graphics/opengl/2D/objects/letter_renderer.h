#pragma once

#include _FAN_PATH(font.h)
#include _FAN_PATH(graphics/opengl/font.h)

#include _FAN_PATH(graphics/graphics.h)

namespace fan_2d {
  namespace graphics {

    struct letter_t {

      struct properties_t {
        f32_t font_size = 16;
        fan::vec3 position = 0;
        fan::color color = fan::colors::white;
        uint16_t letter_id;
      };

      struct instance_t {
        fan::vec3 position;
        f32_t outline_size;
        fan::vec2 size;
        fan::vec2 tc_position;
        fan::color color;
        fan::color outline_color;
        fan::vec2 tc_size;
      private:
        f32_t pad[2];
      };

      static constexpr uint32_t max_instance_size = 256;

      struct id_t{
        id_t(fan::opengl::cid_t* cid) {
          block = cid->id / max_instance_size;
          instance = cid->id % max_instance_size;
        }
        uint32_t block;
        uint32_t instance;
      };


      void open(fan::opengl::context_t* context, fan_2d::graphics::font_t* font_) {

        m_shader.open(context);

        m_shader.set_vertex(
          context,
#include _FAN_PATH(graphics/glsl/opengl/2D/objects/letter.vs)
        );

        m_shader.set_fragment(
          context,
#include _FAN_PATH(graphics/glsl/opengl/2D/objects/letter.fs)
        );

        m_shader.compile(context);

        blocks.open();

        m_draw_node_reference = fan::uninitialized;

        font = font_;
      }
      void close(fan::opengl::context_t* context) {
        m_shader.close(context);
        for (uint32_t i = 0; i < blocks.size(); i++) {
          blocks[i].uniform_buffer.close(context);
        }
        blocks.close();
      }

      void push_back(fan::opengl::context_t* context, fan::opengl::cid_t* cid, const properties_t& p) {
        instance_t it;
        //   it.offset = 0;
        it.position = p.position;
        it.color = p.color;

        fan::font::single_info_t si = font->info.get_letter_info(p.letter_id, p.font_size);

        it.tc_position = si.glyph.position / font->image.size;
        it.tc_size.x = si.glyph.size.x / font->image.size.x;
        it.tc_size.y = si.glyph.size.y / font->image.size.y;

        it.size = si.metrics.size / 2;

        uint32_t block_id = blocks.size() - 1;

        if (block_id == (uint32_t)-1 || blocks[block_id].uniform_buffer.size() == max_instance_size) {
          blocks.push_back({});
          block_id++;
          blocks[block_id].uniform_buffer.open(context);
          blocks[block_id].uniform_buffer.init_uniform_block(context, m_shader.id, "instance_t");
        }

        blocks[block_id].uniform_buffer.push_ram_instance(context, it);

        uint32_t src = blocks[block_id].uniform_buffer.size() % max_instance_size;

        const uint32_t instance_id = blocks[block_id].uniform_buffer.size() - 1;

        blocks[block_id].cid[instance_id] = cid;

        blocks[block_id].uniform_buffer.common.edit(
          context,
          src - 1,
          std::min(src, max_instance_size)
        );

        cid->id = block_id * max_instance_size + instance_id;
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

        blocks[last_block_id].uniform_buffer.common.m_size -= blocks[block_id].uniform_buffer.common.buffer_bytes_size;

        blocks[block_id].cid[instance_id] = blocks[last_block_id].cid[last_instance_id];
        blocks[block_id].cid[instance_id]->id = block_id * max_instance_size + instance_id;

        if (blocks[last_block_id].uniform_buffer.size() == 0) {
          blocks[last_block_id].uniform_buffer.close(context);
          blocks.m_size -= 1;
        }

        blocks[block_id].uniform_buffer.common.edit(
          context,
          instance_id,
          instance_id + 1
        );
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

      void draw(fan::opengl::context_t* context) {
        m_shader.use(context);

        m_shader.set_int(context, "texture_sampler", 0);
        context->opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
        context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, font->image.texture);

        for (uint32_t i = 0; i < blocks.size(); i++) {
          blocks[i].uniform_buffer.bind_buffer_range(context, blocks[i].uniform_buffer.size());

          blocks[i].uniform_buffer.draw(
            context,
            0 * 6,
            blocks[i].uniform_buffer.size() * 6
          );
        }
      }

      void bind_matrices(fan::opengl::context_t* context, fan::opengl::matrices_t* matrices) {
        m_shader.bind_matrices(context, matrices);
      }
      void unbind_matrices(fan::opengl::context_t* context, fan::opengl::matrices_t* matrices) {
        m_shader.unbind_matrices(context, matrices);
      }

      template <typename T>
      T get(fan::opengl::context_t* context, const id_t& id, T instance_t::*member) {
        return blocks[id.block].uniform_buffer.get_instance(context, id.instance)->*member;
      }
      template <typename T, typename T2>
      void set(fan::opengl::context_t* context, const id_t& id, T instance_t::*member, const T2& value) {
        blocks[id.block].uniform_buffer.edit_ram_instance(context, id.instance, (T*)&value, fan::ofof<instance_t, T>(member), sizeof(T));
      }

      fan::shader_t m_shader;
      struct block_t {
        fan::opengl::core::uniform_block_t<instance_t, max_instance_size> uniform_buffer;
        fan::opengl::cid_t* cid[max_instance_size];
      };
      uint32_t m_draw_node_reference;

      fan::hector_t<block_t> blocks;

      fan_2d::graphics::font_t* font;
    };
  }
}