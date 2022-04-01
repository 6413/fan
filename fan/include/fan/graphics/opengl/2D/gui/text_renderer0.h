#pragma once

#include <fan/graphics/opengl/2D/gui/text_renderer.h>

namespace fan_2d {
  namespace opengl {
    namespace gui {
      	struct text_renderer0_t : public text_renderer_t {

				typedef void(*index_change_cb_t)(void*, uint64_t);

				text_renderer0_t() = default;

				void open(fan::opengl::context_t* context, void* gp, index_change_cb_t index_change_cb) {
					m_index_change_cb = index_change_cb;
					m_entity_ids.open();
					m_gp = gp;
					text_renderer_t::open(context);
				}

				void close(fan::opengl::context_t* context) {
					m_entity_ids.close();
				}

				struct properties_t : public text_renderer_t::properties_t {
					uint32_t entity_id;
				};

				void push_back(fan::opengl::context_t* context, properties_t properties) {
					m_entity_ids.emplace_back(properties.entity_id);
					text_renderer_t::push_back(context, properties);
				}

				void erase(fan::opengl::context_t* context, uintptr_t i) {
					for (uint32_t start = i + 1; start < text_renderer_t::size(context); start++) {
						m_index_change_cb(m_gp, m_entity_ids[start]);
					}
					text_renderer_t::erase(context, i);
					m_entity_ids.erase(i);
				}

				void erase(uintptr_t begin, uintptr_t end) = delete;

			protected:

				index_change_cb_t m_index_change_cb;

				void* m_gp;
				fan::hector_t<uint64_t> m_entity_ids;

			};
    }
  }
}