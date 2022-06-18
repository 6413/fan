#pragma once

#include _FAN_PATH(graphics/graphics.h)

namespace fan_2d {
  namespace graphics {

    template <typename T_user_global_data, typename T_user_letter_data>
    struct letter_t {

      using user_global_data_t = T_user_global_data;
      using user_letter_data_t = T_user_letter_data;

      using move_cb_t = void(*)(letter_t*, uint32_t src, uint32_t dst, user_letter_data_t*);

      struct properties_t {
        f32_t font_size = 16;
        fan::vec2 position = 0;
        fan::color color = fan::colors::white;
        uint16_t letter_id;
        user_letter_data_t data;
      };

      struct instance_t {
        alignas(8)  fan::vec2 position;
        alignas(8)  fan::vec2 size;
        alignas(16) fan::color color;
        alignas(8)  fan::vec2 tc_position;
        alignas(8) fan::vec2 tc_size;
      };

      static constexpr uint32_t letter_max_size = 256;

      static constexpr uint32_t offset_position = offsetof(instance_t, position);
      static constexpr uint32_t offset_size = offsetof(instance_t, size);
      static constexpr uint32_t offset_color = offsetof(instance_t, color);
      static constexpr uint32_t offset_tc_position = offsetof(instance_t, tc_position);
      static constexpr uint32_t offset_tc_size = offsetof(instance_t, tc_size);
      static constexpr uint32_t element_byte_size = sizeof(instance_t);

      void open(fan::opengl::context_t* context, fan_2d::graphics::font_t* font_, move_cb_t move_cb_, user_global_data_t* gd) {

        m_shader.open(context);

        m_shader.set_vertex(
          context,
#include _FAN_PATH(graphics/glsl/opengl/2D/objects/text.vs)
        );

        m_shader.set_fragment(
          context,
#include _FAN_PATH(graphics/glsl/opengl/2D/objects/text.fs)
        );

        m_shader.compile(context);

        blocks.open();

        m_draw_node_reference = fan::uninitialized;

        font = font_;
        user_global_data = *gd;
        move_cb = move_cb_;
      }
      void close(fan::opengl::context_t* context) {
        m_shader.close(context);
        for (uint32_t i = 0; i < blocks.size(); i++) {
          blocks[i].uniform_buffer.close(context);
          blocks[i].queue_helper.close(context);
        }
        blocks.close();
      }

      /*void set_position(fan::opengl::context_t* context, uint32_t i, const fan::vec2& position) {
        uniform_blocks.edit_ram_instance(
          context,
          i,
          &position,
          element_byte_size,
          offset_position,
          sizeof(properties_t::position)
        );
        m_queue_helper.edit(context, 0, element_byte_size, &uniform_block);
      }*/

      uint32_t push_back(fan::opengl::context_t* context, const properties_t& p) {
        instance_t it;
        //   it.offset = 0;
        it.position = p.position;
        it.color = p.color;

        fan::font::single_info_t si = font->info.get_letter_info(p.letter_id, p.font_size);

        it.tc_position = si.glyph.position / font->image.size;
        it.tc_size.x = si.glyph.size.x / font->image.size.x;
        it.tc_size.y = si.glyph.size.y / font->image.size.y;

        it.size = si.metrics.size / 2;

        uint32_t i = 0;

        for (; i < blocks.size(); i++) {
          if (blocks[i].uniform_buffer.m_buffer.size() / element_byte_size != letter_max_size) {
            break;
          }
        }

        if (i == blocks.size()) {
          blocks.push_back({});
          blocks[i].queue_helper.open();
          blocks[i].uniform_buffer.open(context);
          blocks[i].uniform_buffer.op.target = fan::opengl::GL_UNIFORM_BUFFER;
          blocks[i].uniform_buffer.bind_uniform_block(context, m_shader.id, "instance_t");
        }

        blocks[i].uniform_buffer.push_ram_instance(context, &it, element_byte_size);
        blocks[i].user_letter_data[blocks[i].uniform_buffer.m_buffer.size() / element_byte_size - 1] = p.data;

        /* blocks[i].queue_helper.edit(
           context,
           blocks[i].uniform_buffer.m_buffer.size() - element_byte_size,
           blocks[i].uniform_buffer.m_buffer.size(),
           &blocks[i].uniform_buffer
         );*/

        blocks[i].uniform_buffer.write_vram_all(context);

        return i * letter_max_size + (blocks[i].uniform_buffer.m_buffer.size() / element_byte_size - 1);
      }
      void erase(fan::opengl::context_t* context, uint32_t id) {
        uint32_t block_id = id / letter_max_size;
        uint32_t letter_id = id % letter_max_size;

        if (block_id == blocks.size() - 1 && letter_id == blocks.ge()->uniform_buffer.m_buffer.size() / element_byte_size - 1) {
          blocks[block_id].uniform_buffer.m_buffer.m_size -= element_byte_size;
          if (blocks[block_id].uniform_buffer.m_buffer.size() == 0) {
            blocks[block_id].uniform_buffer.close(context);
            blocks[block_id].queue_helper.close(context);
            blocks.m_size -= 1;
            return;
          }
        }

        uint32_t last_block_id = blocks.size() - 1;
        uint32_t last_letter_id = blocks[last_block_id].uniform_buffer.m_buffer.size() / element_byte_size - 1;

        instance_t* data = (instance_t*)blocks[block_id].uniform_buffer.get_instance(context, last_letter_id, element_byte_size, 0);

        blocks[block_id].uniform_buffer.edit_ram_instance(
          context,
          letter_id,
          data,
          element_byte_size,
          0,
          element_byte_size
        );
        blocks[block_id].queue_helper.edit(
          context,
          letter_id * element_byte_size,
          letter_id * element_byte_size + element_byte_size,
          &blocks[block_id].uniform_buffer
        );

        blocks[last_block_id].uniform_buffer.m_buffer.m_size -= element_byte_size;

        if (blocks[last_block_id].uniform_buffer.m_buffer.size() == 0) {
          blocks[last_block_id].uniform_buffer.close(context);
          blocks[last_block_id].queue_helper.close(context);
          blocks.m_size -= 1;
        }

        move_cb(
          this,
          last_letter_id + last_block_id * letter_max_size, id,
          &blocks[block_id].user_letter_data[letter_id]
        );
      }

      void enable_draw(fan::opengl::context_t* context) {
        this->draw(context);

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
          blocks[i].uniform_buffer.bind_buffer_range(context, blocks[i].uniform_buffer.m_buffer.size());

          blocks[i].uniform_buffer.draw(
            context,
            0,
            blocks[i].uniform_buffer.m_buffer.size() / element_byte_size * 6
          );
        }
      }

      void bind_matrices(fan::opengl::context_t* context, fan::opengl::matrices_t* matrices) {
        m_shader.bind_matrices(context, matrices);
      }

      fan::shader_t m_shader;
      struct block_t {
        fan::opengl::core::glsl_buffer_t uniform_buffer;
        fan::opengl::core::queue_helper_t queue_helper;
        user_letter_data_t user_letter_data[letter_max_size];
      };
      uint32_t m_draw_node_reference;

      fan::hector_t<block_t> blocks;

      fan_2d::graphics::font_t* font;
      user_global_data_t user_global_data;
      move_cb_t move_cb;
    };
  }
}