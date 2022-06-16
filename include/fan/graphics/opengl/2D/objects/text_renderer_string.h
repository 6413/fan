#pragma once

#include _FAN_PATH(graphics/graphics.h)

namespace fan_2d {
  namespace graphics {

    struct letter_t {

      struct properties_t {
        f32_t font_size = 16;
        fan::vec2 position = 0;
        fan::color color = fan::colors::white;
        uint16_t letter_id;
      };

      struct instance_t {
        fan::vec2 position;
        fan::vec2 size;
        fan::color color;
        fan::vec2 tc_position;
        fan::vec2 tc_size;
      };

      static constexpr uint32_t letter_max_size = 256;
      
      static constexpr uint32_t offset_position = offsetof(instance_t, position);
      static constexpr uint32_t offset_size = offsetof(instance_t, size);
      static constexpr uint32_t offset_color = offsetof(instance_t, color);
      static constexpr uint32_t offset_tc_position = offsetof(instance_t, tc_position);
      static constexpr uint32_t offset_tc_size = offsetof(instance_t, tc_size);
      static constexpr uint32_t element_byte_size = offset_tc_size + sizeof(instance_t::tc_size);

      void open(fan::opengl::context_t* context, fan_2d::graphics::font_t* font_) {
        instances.open();

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

        uniform_block.open(context);
        uniform_block.op.target = fan::opengl::GL_UNIFORM_BUFFER;
        uniform_block.init(context, m_shader.id, element_byte_size);
        // add types here, position, size..
        uniform_block.bind_uniform_block(context, m_shader.id, "instance_t");
        m_queue_helper.open();

        uniform_block.m_buffer.resize(letter_max_size * sizeof(instance_t));

        m_draw_node_reference = fan::uninitialized;

        font = font_;
      }
      void close(fan::opengl::context_t* context) {
        instances.close();
        m_shader.close(context);
        uniform_block.close(context);
        m_queue_helper.close(context);
      }

      void set_position(fan::opengl::context_t* context, uint32_t i, const fan::vec2& position) {
        uniform_block.edit_ram_instance(
          context,
          i,
          &position,
          element_byte_size,
          offset_position,
          sizeof(properties_t::position)
        );
        m_queue_helper.edit(context, 0, element_byte_size, &uniform_block);
      }

      uint32_t push_back(fan::opengl::context_t* context, const properties_t& p) {
        instance_t it;
        it.position = p.position;
        it.color = p.color;
        
        fan::font::single_info_t si = font->info.get_letter_info(p.letter_id, p.font_size);
        
        it.tc_position = si.glyph.position / font->image.size;
        it.tc_size = si.glyph.size / font->image.size;
        
        it.size = si.metrics.size / 2;

        instances.push_back(it);

        return size() - 1;
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

        m_shader.set_vec2(context, "matrix_ratio", context->viewport_size / context->viewport_size.max());
        m_shader.set_int(context, "texture_sampler", 0);
        context->opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
        context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, font->image.texture);

        for (uint32_t i = 0; i < this->size(); i += letter_max_size) {

          uint32_t idx = std::min(size() - i, letter_max_size);

          std::copy(
            &instances[i], 
            &instances[i] + idx,
            (instance_t*)uniform_block.m_buffer.data()
          );

          uniform_block.edit_vram_buffer(context, 0, idx * element_byte_size);

          uniform_block.draw(
            context,
            0,
            idx * 6
          );
        }
      }

      uint32_t size() const {
        return instances.size();
      }

      fan::shader_t m_shader;
      fan::opengl::core::glsl_buffer_t uniform_block;
      fan::opengl::core::queue_helper_t m_queue_helper;
      fan::hector_t<instance_t> instances;
      uint32_t m_draw_node_reference;

      fan_2d::graphics::font_t* font;
    };

    
    //struct text_renderer_string_t {




    //  void open(fan::opengl::context_t* context) {

    //  }
    //  void close(fan::opengl::context_t* context) {

    //  }

    //  void push_back()

    //};
  }
}