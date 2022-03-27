#pragma once

#include <fan/graphics/opengl/gl_core.h>
#include <fan/graphics/shared_graphics.h>
#include <fan/physics/collision/rectangle.h>

namespace fan_2d {
  namespace opengl {

    struct particles_t {

      particles_t() = default;

      static constexpr auto vertex_count = 1;

      void open(fan::opengl::context_t* context) {

        m_shader.open();

        m_shader.set_vertex(
        #include <fan/graphics/glsl/opengl/2D/effects/particles.vs>
        );

        m_shader.set_fragment(
        #include <fan/graphics/glsl/opengl/2D/effects/particles.fs>
        );

        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

        m_shader.compile();

        m_draw_node_reference = fan::uninitialized;
        
        instance.count = 0;
      }
      void close(fan::opengl::context_t* context) {
        m_shader.close();

        if (m_draw_node_reference == fan::uninitialized) {
          return;
        }

        context->disable_draw(m_draw_node_reference);
        m_draw_node_reference = fan::uninitialized;
      }

      struct properties_t {
        fan::vec2 position = 0;
        fan::vec2 size = 0;
        f32_t angle = 0;
        fan::vec2 rotation_point = 0;
        fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
        fan::vec2 position_velocity = 0;
        f32_t angle_velocity = 0;

        fan::opengl::image_t* image;

        uint64_t timeout = 1e+9;
        uint32_t count = 1;
      };

      void set(const properties_t& p) {
        instance = p;
      }

      void reload_sprite(fan::opengl::context_t* context, uint32_t i, fan::opengl::image_t* image) {
        instance.image = image;
      }

      fan::vec2 get_position(fan::opengl::context_t* context, uint32_t i) const {
        return instance.position;
      }
      void set_position(fan::opengl::context_t* context, const fan::vec2& position) {
        instance.position = position;
      }

      fan::vec2 get_size(fan::opengl::context_t* context, uint32_t i) const {
        return instance.size;
      }
      void set_size(fan::opengl::context_t* context, const fan::vec2& size) {
        instance.size = size;
      }

      fan::vec3 get_rotation_vector(fan::opengl::context_t* context, uint32_t i) const {
        return instance.rotation_vector;
      }
      void set_rotation_vector(fan::opengl::context_t* context, const fan::vec3& rotation_vector) {
        instance.rotation_vector = rotation_vector;
      }

      fan::vec2 get_position_velocity(fan::opengl::context_t* context, uint32_t i) const {
        return instance.position_velocity;
      }
      void set_position_velocity(fan::opengl::context_t* context, const fan::vec2& position_velocity) {
        instance.position_velocity = position_velocity;
      }

      fan::vec2 get_angle_velocity(fan::opengl::context_t* context, uint32_t i) const {
        return instance.angle_velocity;
      }
      void set_angle_velocity(fan::opengl::context_t* context, f32_t angle_velocity) {
        instance.angle_velocity = angle_velocity;
      }

      uint32_t size(fan::opengl::context_t* context) const {
        return instance.count;
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

      //	protected:

      void draw_vertex(fan::opengl::context_t* context, uint32_t begin, uint32_t end) {
          const fan::vec2 viewport_size = context->viewport_size;

          fan::mat4 projection(1);
		      projection = fan::math::ortho<fan::mat4>(
            (f32_t)viewport_size.x * 0.5,
            ((f32_t)viewport_size.x + (f32_t)viewport_size.x * 0.5), 
            ((f32_t)viewport_size.y + (f32_t)viewport_size.y * 0.5), 
            ((f32_t)viewport_size.y * 0.5), 
            0.01,
            1000.0
          );

		      fan::mat4 view(1);
		      view = context->camera.get_view_matrix(view.translate(fan::vec3((f_t)viewport_size.x * 0.5, (f_t)viewport_size.y * 0.5, -700.0f)));

		      m_shader.set_projection(projection);
		      m_shader.set_view(view);

		      glDrawArrays(GL_POINTS, begin, end - begin);
      }

      void draw(fan::opengl::context_t* context) {
        m_shader.use();

        //uint32_t texture_id = fan::uninitialized;
        //uint32_t from = 0;
        //uint32_t to = 0;
        //for (uint32_t i = 0; i < this->size(context); i++) {
        //  if (texture_id != instance.image->texture) {
        //    if (to) {
        //      draw_vertex(
        //        context,
        //        (from)*vertex_count,
        //        (from + to) * vertex_count
        //      );
        //    }
        //    from = i;
        //    to = 0;
        //    texture_id = instance.image->texture;
        //  //  m_shader.set_int("texture_sampler", 0);
        //    glActiveTexture(GL_TEXTURE0);
        //    glBindTexture(GL_TEXTURE_2D, texture_id);
        //  }
        //  to++;
        //}

        m_shader.set_uint("vertex_count", vertex_count);
        m_shader.set_uint("count", instance.count);
        m_shader.set_vec2("position", instance.position);
        m_shader.set_vec2("size", instance.size);
        m_shader.set_float("angle", instance.angle);
        m_shader.set_vec2("position_velocity", instance.position_velocity);
        m_shader.set_vec2("angle_velocity", instance.angle_velocity);
        m_shader.set_vec3("rotation_vector", instance.rotation_vector);

        m_shader.set_float("time", m_delta);

        draw_vertex(
          context,
          0,
          vertex_count * instance.count
        );
      }

      void set_delta(fan::opengl::context_t* context, f32_t delta) {
        m_delta = delta;
			}

      properties_t instance;

      fan::shader_t m_shader;
      uint32_t m_draw_node_reference;

      f32_t m_delta;
    };
  }
}