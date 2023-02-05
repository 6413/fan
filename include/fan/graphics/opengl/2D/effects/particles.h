#pragma once

#include _FAN_PATH(graphics/opengl/gl_shader.h)
#include _FAN_PATH(graphics/shared_graphics.h)
#include _FAN_PATH(physics/collision/rectangle.h)

namespace fan_2d {
  namespace opengl {

    struct particles_t {

      particles_t() = default;

      static constexpr auto vertex_count = 3;

      void open(fan::opengl::context_t* context) {

        m_shader.open(context);

        m_shader.set_vertex(
          context,
          #include _FAN_PATH(graphics/glsl/opengl/2D/effects/particles.vs)
        );

        m_shader.set_fragment(
          context,
          #include _FAN_PATH(graphics/glsl/opengl/2D/effects/particles.fs)
        );

        context->opengl.glEnable(fan::opengl::GL_VERTEX_PROGRAM_POINT_SIZE);

        m_shader.compile(context);

        m_draw_node_reference = fan::uninitialized;
        
        instance.count = 0;
      }
      void close(fan::opengl::context_t* context) {
        m_shader.close(context);

        if (m_draw_node_reference == fan::uninitialized) {
          return;
        }

        context->disable_draw(m_draw_node_reference);
        m_draw_node_reference = fan::uninitialized;
      }

      struct properties_t {
        properties_t() = default;

        uint64_t begin_time;
        uint64_t alive_time;
        uint64_t respawn_time;
        uint32_t count;
        fan::vec2 position;
        fan::vec2 size;
        f32_t angle;
        fan::vec3 rotation_vector;
        fan::vec2 position_velocity;
        f32_t angle_velocity;
        f32_t begin_angle;
        f32_t end_angle;

        fan::opengl::image_t image;
      };

      void set(properties_t p) {
        //p.begin_time;
        p.alive_time = .5e+9;
        p.respawn_time = 0e+9;
        //p.count = 1;
        //p.position = 0;
        p.size = 15;
        p.angle = 0;
        p.rotation_vector = fan::vec3(0, 0, 1);
        //p.position_velocity = 0;
        //p.angle_velocity = 0;
        p.begin_angle = -fan::math::pi / 4;
        p.end_angle = -fan::math::pi / 2;
        instance = p;
        instance.begin_time = fan::time::clock::now();
      }

      void reload_sprite(fan::opengl::context_t* context, uint32_t i, const fan::opengl::image_t& image) {
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
      void set_size(fan::opengl::context_t* context, uint32_t size) {
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

      f32_t get_begin_angle(fan::opengl::context_t* context, uint32_t i) const {
        return instance.angle_velocity;
      }
      void set_begin_angle(fan::opengl::context_t* context, f32_t begin_angle) {
        instance.begin_angle = begin_angle;
      }

      f32_t get_end_angle(fan::opengl::context_t* context, uint32_t i) const {
        return instance.end_angle;
      }
      void set_end_angle(fan::opengl::context_t* context, f32_t end_angle) {
        instance.end_angle = end_angle;
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
		      context->opengl.glDrawArrays(fan::opengl::GL_TRIANGLES, begin, end - begin);
      }

      void draw(fan::opengl::context_t* context) {
        m_shader.use(context);

        //uint32_t texture_id = fan::uninitialized;
        //uint32_t from = 0;
        //uint32_t to = 0;
        //for (uint32_t i = 0; i < this->size(context); i++) {
        //  if (texture_id != instance.image.texture) {
        //    if (to) {
        //      draw_vertex(
        //        context,
        //        (from)*vertex_count,
        //        (from + to) * vertex_count
        //      );
        //    }
        //    from = i;
        //    to = 0;
        //    texture_id = instance.image.texture;
        //  //  m_shader.set_int("texture_sampler", 0);
        //    glActiveTexture(GL_TEXTURE0);
        //    glBindTexture(GL_TEXTURE_2D, texture_id);
        //  }
        //  to++;
        //}

        m_shader.set_float(context, "time", m_delta);
        m_shader.set_uint(context, "vertex_count", vertex_count);
        m_shader.set_uint(context, "count", instance.count);
        m_shader.set_float(context, "alive_time", (f32_t)instance.alive_time / 1e+9);
        m_shader.set_float(context, "respawn_time", (f32_t)instance.respawn_time / 1e+9);
        m_shader.set_vec2(context, "position", instance.position);
        m_shader.set_vec2(context, "size", instance.size);
        m_shader.set_vec2(context, "position_velocity", instance.position_velocity);
        m_shader.set_float(context, "angle_velocity", instance.angle_velocity);
        m_shader.set_vec3(context, "rotation_vector", instance.rotation_vector);
        m_shader.set_float(context, "begin_angle", instance.begin_angle);
        m_shader.set_float(context, "end_angle", instance.end_angle);

        draw_vertex(
          context,
          0,
          vertex_count * instance.count
        );
      }

      void set_delta(fan::opengl::context_t* context, uint64_t delta) {
        m_delta = (f64_t)(delta - instance.begin_time) / 1e+9;
			}

      properties_t instance;

      fan::opengl::shader_t m_shader;
      uint32_t m_draw_node_reference;

      f32_t m_delta;
    };
  }
}