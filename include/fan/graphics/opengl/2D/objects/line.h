#pragma once

#include _FAN_PATH(graphics/opengl/gl_core.h)
#include _FAN_PATH(graphics/opengl/gl_shader.h)
#include _FAN_PATH(graphics/shared_graphics.h)

namespace fan_2d {

  namespace opengl {

    struct line_t {

      struct properties_t {
        fan::color color;
        fan::vec2 src;
        fan::vec2 dst;
      };

    private:

      struct instance_t {
        fan::color color;
        fan::vec2 vertex;
      };

    public:

      static constexpr uint32_t vertex_count = 2;

      static constexpr uint32_t offset_color = offsetof(instance_t, color);
      static constexpr uint32_t offset_position = offsetof(instance_t, vertex);
      static constexpr uint32_t element_byte_size = sizeof(instance_t);

      void open(fan::opengl::context_t* context) {

				m_shader.open(context);

				m_shader.set_vertex(
					context, 
					#include _FAN_PATH(graphics/glsl/opengl/2D/objects/line.vs>
				);

				m_shader.set_fragment(
					context, 
					#include _FAN_PATH(graphics/glsl/opengl/2D/objects/line.fs>
				);

				m_shader.compile(context);

				m_glsl_buffer.open(context);
				m_glsl_buffer.init(context, m_shader.id, element_byte_size);
				m_queue_helper.open();
				m_draw_node_reference = fan::uninitialized;
			}
			void close(fan::opengl::context_t* context) {

				m_glsl_buffer.close(context);
				m_queue_helper.close(context);
				m_shader.close(context);

				if (m_draw_node_reference == fan::uninitialized) {
					return;
				}

				context->disable_draw(m_draw_node_reference);
				m_draw_node_reference = fan::uninitialized;
			}

      void push_back(fan::opengl::context_t* context, const properties_t& p) {
        instance_t instance;
        instance.color = p.color;
        instance.vertex = p.src;
        m_glsl_buffer.push_ram_instance(context, &instance, sizeof(instance));
        instance.vertex = p.dst;
        m_glsl_buffer.push_ram_instance(context, &instance, sizeof(instance));
				m_glsl_buffer.write_vram_all(context);
      }

      void set_line(fan::opengl::context_t* context, uint32_t i, const fan::vec2& src, const fan::vec2& dst) {
        i *= vertex_count;
        m_glsl_buffer.edit_ram_instance(context, i, &src, element_byte_size, sizeof(fan::color), sizeof(src));
        m_glsl_buffer.edit_ram_instance(context, i + 1, &dst, element_byte_size, sizeof(fan::color), sizeof(dst));
				m_queue_helper.edit(
					context,
					i * element_byte_size,
					(i + vertex_count) * element_byte_size,
					&m_glsl_buffer
				);
      }

      fan::color get_color(fan::opengl::context_t* context, uint32_t i) {
        return *(fan::color*)m_glsl_buffer.get_buffer_data(i * 2 * element_byte_size);
      }
      void set_color(fan::opengl::context_t* context, uint32_t i, const fan::color& color) {
        i *= vertex_count;
        m_glsl_buffer.edit_ram_instance(context, i, &color, element_byte_size, 0, sizeof(fan::color));
				m_glsl_buffer.edit_ram_instance(context, i + 1, &color, element_byte_size, 0, sizeof(fan::color));
				m_queue_helper.edit(
					context,
					i * element_byte_size,
					(i + vertex_count) * element_byte_size,
					&m_glsl_buffer
				);
      }

      
			void enable_draw(fan::opengl::context_t* context) {
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

			// pushed to window draw queue
			void draw(fan::opengl::context_t* context) {
				context->set_depth_test(false);

				m_shader.use(context);

				m_glsl_buffer.m_vao.bind(context);

        // possibly disable depth test here
        context->opengl.glDrawArrays(fan::opengl::GL_LINES, 0, this->size(context) * vertex_count);
			}

			uint32_t size(fan::opengl::context_t* context) const {
				return m_glsl_buffer.m_buffer.size() / element_byte_size / vertex_count;
			}

			void bind_matrices(fan::opengl::context_t* context, fan::opengl::matrices_t* matrices) {
        m_shader.bind_matrices(context, matrices);
      }

      uint32_t m_draw_node_reference;

    //private:

      fan::shader_t m_shader;
      fan::opengl::core::glsl_buffer_t m_glsl_buffer;
      fan::opengl::core::queue_helper_t m_queue_helper;

    };

  }
}