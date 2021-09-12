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

		struct load_properties_t {
			uint32_t visual_output;
			uintptr_t internal_format;
			uintptr_t format;
			uintptr_t type;
			uintptr_t filter;
		};

		struct image_T {

			image_T() {}
			image_T(fan::window* window) : window(window) {}

			image_T(image_T&& image_T_) = delete;

			image_T& operator=(image_T&& image_T_)  = delete;

			image_T(const image_T& image_T_) = delete;

			image_T& operator=(const image_T& image_T_) = delete;

			void free() {
				if (texture) {
#if fan_renderer == fan_renderer_opengl
						glDeleteTextures(1, &texture);
						texture = 0;
#elif fan_renderer == fan_renderer_vulkan
					if (texture && window->m_vulkan->device) {
						
						vkDestroyImage(window->m_vulkan->device, texture, nullptr);
						texture = nullptr;
					}
#endif
				}
			}

			~image_T() {
				this->free();
			}

#if fan_renderer == fan_renderer_opengl
			uint32_t texture = 0;
#elif fan_renderer == fan_renderer_vulkan
			VkImage texture = nullptr;
#endif

			load_properties_t properties;

			fan::vec2i size = 0;

			fan::window* window = nullptr;

		};

		using image_t = image_T*;

		static void copy_texture(image_t src, image_t dst) {
			// opengl alloc pixels, glGetTexImage(pixels), glTexImage2D(pixels)
			assert(0);
		}
	}
}