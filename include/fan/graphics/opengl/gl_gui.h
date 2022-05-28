#pragma once

#include <fan/types/types.h>

#include <fan/graphics/renderer.h>

#if fan_renderer == fan_renderer_opengl

#include <fan/graphics/graphics.h>
#include <fan/graphics/gui/themes.h>

#include <fan/physics/collision/rectangle.h>

#include <fan/graphics/shared_gui.h>
//
//namespace fan_2d {
//
//	namespace graphics {
//
//		namespace gui {
//
//			struct circle : public fan_2d::opengl::circle {
//
//				circle(fan::camera* camera);
//
//			};
//
//			//struct sprite : public fan_2d::opengl::sprite_t {
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