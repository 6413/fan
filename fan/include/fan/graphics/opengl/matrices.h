#pragma once

#include <fan/bll.h>
#include <fan/graphics/opengl/gl_core.h>

namespace fan {
	namespace opengl {

		struct matrices_t {

      typedef void(*inform_cb_t)(matrices_t*, void* updateptr, void* userptr);

      struct inform_data_t {
        inform_cb_t cb;
        void* userptr;
      };

      void open() {
        m_inform_data_list.open();
      }
      void close() {
        m_inform_data_list.close();
      }

			void set_ortho(void* updateptr, const fan::vec2& x, const fan::vec2& y) {
        m_projection = fan::math::ortho<fan::mat4>(
          x.x,
          x.y,
          y.y,
          y.x,
          0.1,
          1000.0
        );

        uint32_t it = m_inform_data_list.begin();

        while (it != m_inform_data_list.end()) {
          m_inform_data_list.start_safe_next(it);

          m_inform_data_list[it].cb(this, updateptr, m_inform_data_list[it].userptr);

          it = m_inform_data_list.end_safe_next();
        }
      }

      uint32_t push_inform(inform_cb_t cb, void* userptr) {
        inform_data_t data;
        data.cb = cb;
        data.userptr = userptr;
        return m_inform_data_list.push_back(data);
      }
      void erase_inform(uint32_t id) {
        m_inform_data_list.erase(id);
      }

      bll_t<inform_data_t> m_inform_data_list;

			fan::mat4 m_projection;
		};

	}
}