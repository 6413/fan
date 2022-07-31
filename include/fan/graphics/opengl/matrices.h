#pragma once

#include _FAN_PATH(bll.h)
#include _FAN_PATH(graphics/opengl/gl_core.h)

namespace fan {
	namespace opengl {

		struct matrices_t {

      void open() {
        m_view = fan::mat4(1);
        camera_position = 0;
      }
      void close() {
      }

      fan::vec3 get_camera_position() const {
        return camera_position;
      }
      void set_camera_position(const fan::vec3& cp) {
        camera_position = cp;

        m_view[3][0] = 0;
        m_view[3][1] = 0;
        m_view[3][2] = 0;
        m_view = m_view.translate(camera_position);
        fan::vec3 position = m_view.get_translation();
        constexpr fan::vec3 front(0, 0, 1);

        m_view = fan::math::look_at_left<fan::mat4>(position, position + front, fan::camera::world_up);
      }

			void set_ortho(const fan::vec2& x, const fan::vec2& y) {
        m_projection = fan::math::ortho<fan::mat4>(
          x.x,
          x.y,
          y.y,
          y.x,
          0.1,
          100.0
        );

        m_view[3][0] = 0;
        m_view[3][1] = 0;
        m_view[3][2] = 0;
        m_view = m_view.translate(camera_position);
        fan::vec3 position = m_view.get_translation();
        constexpr fan::vec3 front(0, 0, 1);

        m_view = fan::math::look_at_left<fan::mat4>(position, position + front, fan::camera::world_up);
      }

			fan::mat4 m_projection;
      // temporary
      fan::mat4 m_view;

      fan::vec3 camera_position;
		};

    static void open_matrices(matrices_t* matrices, fan::vec2 window_size, const fan::vec2& x, const fan::vec2& y) {
      matrices->open();
      fan::vec2 ratio = window_size / window_size.max();
      std::swap(ratio.x, ratio.y);
      matrices->set_ortho(fan::vec2(-1, 1), fan::vec2(-1, 1));
    }
	}
}