#pragma once

#include _FAN_PATH(types/types.h)

#include _FAN_PATH(window/window.h)

#include _FAN_PATH(graphics/camera.h)

#include _FAN_PATH(graphics/webp.h)

#if fan_renderer == fan_renderer_opengl
	#include _FAN_PATH(graphics/opengl/gl_image.h)
#endif

#define get_properties(v) decltype(v)::properties_t

namespace fan_2d {

	namespace opengl {

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
	}
}