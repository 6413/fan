#pragma once

#include <fan/window/window.hpp>

#include <fan/graphics/camera.hpp>

constexpr f32_t meter_scale = 100;

namespace fan_2d {

	namespace graphics {

		struct pixel_data_t {
			uint8_t* pixels[4]{};
			int linesize[4]{};
			fan::vec2i size;
			AVPixelFormat format;
			// 32bpp AVPixelFormat::AV_PIX_FMT_BGR0
		};

		struct rectangle;

		struct rectangle_corners_t {

			fan::vec2 top_left;
			fan::vec2 top_right;
			fan::vec2 bottom_left;
			fan::vec2 bottom_right;

			const fan::vec2 operator[](const uintptr_t i) const {
				return !i ? top_left : i == 1 ? top_right : i == 2 ? bottom_left : bottom_right;
			}

			fan::vec2& operator[](const uintptr_t i) {
				return !i ? top_left : i == 1 ? top_right : i == 2 ? bottom_left : bottom_right;
			}

		};

		// 0 top left, 1 top right, 2 bottom left, 3 bottom right
		constexpr rectangle_corners_t get_rectangle_corners_no_rotation(const fan::vec2& position, const fan::vec2& size) {
			return { 
				position, 
				position + fan::vec2(size.x, 0), 
				position + fan::vec2(0, size.y), 
				position + size
			};
		}

		static fan::vec2 get_transformed_point(fan::vec2 input, f32_t a) {
			float x = input.x * cos(a) - input.y * sin(a);
			float y = input.x * sin(a) + input.y * cos(a);
			return fan::vec2(x, y);
		}

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

			image_info(fan::window* window) : texture(std::make_unique<texture_id_handler>(window)) {}

			fan::vec2i size = 0;

			std::unique_ptr<texture_id_handler> texture;

		};
	}
}