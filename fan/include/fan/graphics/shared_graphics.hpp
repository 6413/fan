#pragma once

#include <fan/window/window.hpp>

#include <fan/graphics/camera.hpp>

#define get_properties(v) decltype(v)::properties_t

namespace fan_2d {

	namespace graphics {

		struct pixel_data_t {
			pixel_data_t() {}
			pixel_data_t(fan::image_loader::image_data& image_data)
				: 
				size(image_data.size), format(image_data.format) {
				std::memcpy(pixels, image_data.data, sizeof(image_data.data));
				std::memcpy(linesize, image_data.linesize, sizeof(image_data.linesize));
			}

			uint8_t* pixels[4];
			int linesize[4];
			fan::vec2i size;
			AVPixelFormat format;
			// 32bpp AVPixelFormat::AV_PIX_FMT_BGR0
		};

		struct queue_helper_t {
			queue_helper_t() {}

			queue_helper_t(fan::window* window) : window_(window) {

			}
			~queue_helper_t() {
				if (m_edit_index != -1) {
					window_->remove_write_call(m_edit_index);
					m_edit_index = -1;
				}
				if (m_write_index != -1) {
					window_->remove_write_call(m_write_index);
					m_write_index = -1;
				}
			}


			void edit(uint32_t begin, uint32_t end, std::function<void()> edit_function) {

				if (m_write) {
					return;
				}

				m_min_edit = std::min(m_min_edit, begin);
				m_max_edit = std::max(m_max_edit, end);

				if (!m_edit) {
					m_edit_index = window_->push_write_call(this, [&, f = edit_function] {
						if (m_min_edit == (uint32_t)-1) {
							return;
						}

						f();
						m_edit_index = -1;
					});

					m_edit = true;
				}
			}

			void write(std::function<void()> write_function) {
				if (m_edit) {

					m_min_edit = -1;
					m_max_edit = 0;

					window_->edit_write_call(m_edit_index, this, [&] {});

					m_edit = false;
				}

				if (!m_write) {

					m_write_index = window_->push_write_call(this, [&, f = write_function] {

						f();
					});

					m_write = true;
				}
			}

			void on_write(fan::window* window) {

				if (m_edit) {
					m_min_edit = -1;
					m_max_edit = 0;

					window->edit_write_call(m_edit_index, this, [&] {});

					m_edit = false;
				}

				m_write = false;
				m_write_index = -1;
			}

			void on_edit() {
				m_min_edit = -1;
				m_max_edit = 0;

				m_edit = false;
			}

			bool m_write = false;
			bool m_edit = false;

			uint32_t m_write_index = -1;
			uint32_t m_edit_index = -1;

			uint32_t m_min_edit = -1;
			uint32_t m_max_edit = 0;

			fan::window* window_ = nullptr;
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
				position - fan::vec2(size.x, size.y ),
				position + fan::vec2(size.x, -size.y), 
				position + fan::vec2(-size.x, size.y), 
				position + fan::vec2(size.x , size.y)
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
						texture = -1;
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

		static void unload_image(image_t image) {
			if (image) {
				delete image;
				image = nullptr;
			}
		}

		static void copy_texture(image_t src, image_t dst) {
			// opengl alloc pixels, glGetTexImage(pixels), glTexImage2D(pixels)
			assert(0);
		}
	}
}