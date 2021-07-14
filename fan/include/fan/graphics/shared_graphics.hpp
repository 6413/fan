#pragma once

#include <fan/window/window.hpp>

#include <fan/graphics/camera.hpp>

namespace fan_2d {

	namespace graphics {

		struct texture_id_handler {

			texture_id_handler(
				fan::window* window
			) : window(window) {

			}

			texture_id_handler(texture_id_handler&& handler) = delete;

			texture_id_handler& operator=(texture_id_handler&& handler) = delete;

			texture_id_handler(const texture_id_handler& handler) = delete;

			texture_id_handler& operator=(const texture_id_handler& handler) = delete;

			void free() {
				if (texture_id) {
#if fan_renderer == fan_renderer_opengl
					if (texture_id) {
						glDeleteTextures(1, &texture_id);
						texture_id = 0;
					}
#elif fan_renderer == fan_renderer_vulkan
					if (texture_id && window->m_vulkan->device) {
						
						vkDestroyImage(window->m_vulkan->device, texture_id, nullptr);
						texture_id = nullptr;
					}
#endif
				}
			}

			~texture_id_handler() {
				this->free();
			}

#if fan_renderer == fan_renderer_opengl
			uint32_t texture_id = 0;
#elif fan_renderer == fan_renderer_vulkan
			VkImage texture_id = nullptr;
#endif

		protected:

			fan::window* window = nullptr;

		};


		struct image_info {

			image_info(fan::window* window) : texture(std::make_unique<texture_id_handler>(window)) {

			}

			fan::vec2i size = 0;

			std::unique_ptr<texture_id_handler> texture;

		};

		struct sprite_properties {

			std::array<fan::vec2, 6> texture_coordinates = {
				fan::vec2(0, 1),
				fan::vec2(1, 1),
				fan::vec2(1, 0),

				fan::vec2(0, 1),
				fan::vec2(0, 0),
				fan::vec2(1, 0)
			};

			f32_t transparency = 1;

		};

	}

}