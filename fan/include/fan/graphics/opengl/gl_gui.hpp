#pragma once

#include <fan/graphics/renderer.hpp>

#if fan_renderer == fan_renderer_opengl

#include <fan/graphics/graphics.hpp>
#include <fan/graphics/themes.hpp>

#include <fan/physics/collision/rectangle.hpp>

#include <fan/graphics/shared_gui.hpp>
//
//namespace fan_2d {
//
//	namespace graphics {
//
//		namespace gui {
//
//			struct circle : public fan_2d::graphics::circle {
//
//				circle(fan::camera* camera);
//
//			};
//
//			//struct sprite : public fan_2d::graphics::sprite {
//
//			//	sprite(fan::camera* camera);
//			//	// scale with default is sprite size
//			//	sprite(fan::camera* camera, const std::string& path, const fan::vec2& position, const fan::vec2& size = 0, f32_t transparency = 1);
//			//	sprite(fan::camera* camera, unsigned char* pixels, const fan::vec2& position, const fan::vec2i& size = 0, f32_t transparency = 1);
//
//			//};
//
//		}
//
//	}
//}
//
#endif