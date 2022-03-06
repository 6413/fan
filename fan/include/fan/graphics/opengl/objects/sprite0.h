#pragma once

#include <fan/graphics/opengl/objects/sprite.h>

namespace fan_2d {
  namespace graphics {

		// moves last to erased spot
		struct sprite0 : public sprite {

			typedef void(*erase_cb_t)(void*, uint64_t, uint32_t);

			sprite0() = default;

			void open(fan::opengl::context_t* context, void* user_ptr, erase_cb_t erase_cb) {
				sprite::open(context);

				m_erase_cb = erase_cb;
				m_user_ptr = user_ptr;
				m_push_back_ids.open();
			}

			void close(fan::opengl::context_t* context) {
				m_push_back_ids.close();
				sprite::close(context);
			}

			struct properties_t : public sprite::properties_t {
				uint64_t id = fan::uninitialized;
			};

			void push_back(fan::opengl::context_t* context, const properties_t& properties) {
				m_push_back_ids.push_back(properties.id);
				fan_2d::graphics::sprite::push_back(context, properties);
			}

			void erase(fan::opengl::context_t* context, uint32_t i) {

				if (i != this->size(context) - 1) {

					std::memmove(
						m_glsl_buffer.m_buffer.begin() + i * vertex_count * element_byte_size, 
						m_glsl_buffer.m_buffer.end() - element_byte_size * vertex_count, 
						element_byte_size * vertex_count
					);

					m_glsl_buffer.erase_instance((this->size(context) - 1) * vertex_count, 1, element_byte_size, sprite::vertex_count);

					m_erase_cb(m_user_ptr, *(m_push_back_ids.end() - 1), i);

					m_push_back_ids[i] = *(m_push_back_ids.end() - 1);
					m_push_back_ids.pop_back();

					m_store_sprite[i] = *(m_store_sprite.end() - 1);

					m_store_sprite.pop_back();

					uint32_t to = m_glsl_buffer.m_buffer.size();

					m_queue_helper.edit(
						context,
						i * vertex_count * element_byte_size,
						to,
						&m_glsl_buffer
					);
				}
				else {
					sprite::erase(context, i);
					m_push_back_ids.pop_back();
				}
			}

			void erase(fan::opengl::context_t* context, uint32_t, uint32_t) = delete;

		protected:

			void* m_user_ptr;

			fan::hector_t<uint64_t> m_push_back_ids;

			erase_cb_t m_erase_cb;

		};

  }
}