#pragma once

#include <fan/types/types.hpp>

#include <fan/window/window.hpp>

#include <fan/graphics/camera.hpp>

#include <fan/graphics/webp.h>

#define get_properties(v) decltype(v)::properties_t

namespace fan_2d {

	namespace graphics {

		struct pixel_data_t {
			pixel_data_t() {}
			/*pixel_data_t(fan::image_loader::image_data& image_data)
				: 
				size(image_data.size), format(image_data.format) {
				std::memcpy(pixels, image_data.data, sizeof(image_data.data));
				std::memcpy(linesize, image_data.linesize, sizeof(image_data.linesize));
			}*/

			uint8_t* pixels[4]{};
			int linesize[4]{};
			// width and height - in some cases there is linesize so it can be different than original image size
			fan::vec2i size=0;
			//AVPixelFormat format;
			// 32bpp AVPixelFormat::AV_PIX_FMT_BGR0
		};

		struct image_info_t {
			uint8_t* data;
			fan::vec2i size;
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

		static constexpr fan::vec2 convert_tc_4_2_6(const std::array<fan::vec2, 4>* tx, uint32_t i) {
			constexpr std::array<uint8_t, 8> table{0, 1, 2, 2, 3, 0};
			return (*tx)[table[i]];
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

		namespace image_load_properties {
			inline uint32_t visual_output = GL_CLAMP_TO_BORDER;
			inline uintptr_t internal_format = GL_RGBA;
			inline uintptr_t format = GL_RGBA;
			inline uintptr_t type = GL_UNSIGNED_BYTE;
			inline uintptr_t filter = GL_NEAREST;
		}

		// fan::get_device(window)
		static image_t load_image(const std::string& path) {
		#if fan_renderer == fan_renderer_opengl

		#if fan_assert_if_same_path_loaded_multiple_times

			static std::unordered_map<std::string, bool> existing_images;

			if (existing_images.find(path) != existing_images.end()) {
				fan::throw_error("image already existing " + path);
			}

			existing_images[path] = 0;

		#endif

			fan_2d::graphics::image_t info = new fan_2d::graphics::image_T;

			auto image = fan::webp::load_image(path);

			glGenTextures(1, &info->texture);

			glBindTexture(GL_TEXTURE_2D, info->texture);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);

			uintptr_t internal_format = 0, format = 0, type = 0;

			internal_format = GL_RGBA;
			format = GL_RGBA;
			type = GL_UNSIGNED_BYTE;

			info->size = image.size;

			glTexImage2D(GL_TEXTURE_2D, 0, internal_format, info->size.x, info->size.y, 0, format, type, image.data);

			fan::webp::free_image(image.data);

			glGenerateMipmap(GL_TEXTURE_2D);
			glBindTexture(GL_TEXTURE_2D, 0);

			return info;
		#endif

		}

		//image_t load_image(fan::window* window, const pixel_data_t& pixel_data);
		static fan_2d::graphics::image_t load_image(const fan::webp::image_info_t& image_info)
		{
		#if fan_renderer == fan_renderer_opengl
			fan_2d::graphics::image_t info = new fan_2d::graphics::image_T;

			glGenTextures(1, &info->texture);

			glBindTexture(GL_TEXTURE_2D, info->texture);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);

			uintptr_t internal_format = 0, format = 0, type = 0;

			internal_format = GL_RGBA;
			format = GL_RGBA;
			type = GL_UNSIGNED_BYTE;

			info->size = image_info.size;

			glTexImage2D(GL_TEXTURE_2D, 0, internal_format, info->size.x, info->size.y, 0, format, type, image_info.data);

			fan::webp::free_image(image_info.data);

			glGenerateMipmap(GL_TEXTURE_2D);

			glBindTexture(GL_TEXTURE_2D, 0);

			return info;
		#endif
		}
		static fan_2d::graphics::image_t load_image(const fan_2d::graphics::image_info_t& image_info) {
		#if fan_renderer == fan_renderer_opengl
			fan_2d::graphics::image_t info = new fan_2d::graphics::image_T;

			glGenTextures(1, &info->texture);

			glBindTexture(GL_TEXTURE_2D, info->texture);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);

			uintptr_t internal_format = 0, format = 0, type = 0;

			internal_format = GL_RGBA;
			format = GL_RGBA;
			type = GL_UNSIGNED_BYTE;

			info->size = image_info.size;

			glTexImage2D(GL_TEXTURE_2D, 0, internal_format, info->size.x, info->size.y, 0, format, type, image_info.data);

			glGenerateMipmap(GL_TEXTURE_2D);

			glBindTexture(GL_TEXTURE_2D, 0);

			return info;
		#endif
		}

	}
}