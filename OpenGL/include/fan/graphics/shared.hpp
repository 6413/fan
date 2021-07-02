#pragma once

#include <fan/types/vector.hpp>

#include <functional>

namespace fan {


	inline bool gpu_queue = false;

	static void begin_queue() {
		gpu_queue = true;
	}

	static void end_queue() {
		gpu_queue = false;
	}

	static void start_queue(const std::function<void()>& function) {
		gpu_queue = true;
		function();
		gpu_queue = false;
	}

	namespace vertex_queue {
		constexpr uint32_t position = 1;
		constexpr uint32_t size = 2;
		constexpr uint32_t color = 4;
		constexpr uint32_t angle = 8;
		constexpr uint32_t indices = 16;
	};

}

namespace fan_2d {

	namespace graphics {

		struct rectangle_corners_t {

			fan::vec2 top_left;
			fan::vec2 top_right;
			fan::vec2 bottom_left;
			fan::vec2 bottom_right;

			const fan::vec2 operator[](const uint_t i) const {
				return !i ? top_left : i == 1 ? top_right : i == 2 ? bottom_left : bottom_right;
			}

			fan::vec2& operator[](const uint_t i) {
				return !i ? top_left : i == 1 ? top_right : i == 2 ? bottom_left : bottom_right;
			}

		};

		constexpr rectangle_corners_t get_rectangle_corners_no_rotation(const fan::vec2& position, const fan::vec2& size) {
			return { position, position + fan::vec2(size.x, 0), position + fan::vec2(0, size.y), position + size };
		}

		static fan::vec2 get_transformed_point(fan::vec2 input, f32_t a) {
			float x = input.x * cos(a) - input.y * sin(a);
			float y = input.x * sin(a) + input.y * cos(a);
			return fan::vec2(x, y);
		}
	}
}