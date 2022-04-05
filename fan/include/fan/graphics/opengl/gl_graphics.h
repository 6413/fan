#pragma once

#include <fan/types/types.h>

#include <fan/graphics/renderer.h>

#if fan_renderer == fan_renderer_opengl

#include <fan/graphics/opengl/gl_core.h>
#include <fan/graphics/opengl/gl_texture.h>

#include <fan/graphics/shared_core.h>
#include <fan/graphics/shared_graphics.h>

#include <fan/graphics/webp.h>

#include <fan/graphics/opengl/2D/objects/line.h>
#include <fan/graphics/opengl/2D/objects/rectangle.h>
#include <fan/graphics/opengl/2D/objects/circle.h>
#include <fan/graphics/opengl/2D/objects/sprite.h>
#include <fan/graphics/opengl/2D/objects/sprite0.h>
#include <fan/graphics/opengl/2D/objects/yuv420p_renderer.h>

#include <fan/graphics/opengl/2D/objects/depth/depth_rectangle.h>

#include <fan/graphics/opengl/2D/effects/particles.h>
#include <fan/graphics/opengl/2D/effects/flame.h>

#include <fan/graphics/opengl/3D/objects/model.h>
#include <fan/graphics/opengl/3D/objects/skybox.h>

namespace fan_3d {
	namespace opengl {
		static void add_camera_rotation_callback(fan::opengl::context_t* context, fan::window_t* window) {
			window->add_mouse_move_callback(context, [](fan::window_t* w, const fan::vec2i& position, void* context) {
				fan::vec2 offset(w->get_raw_mouse_offset().x, -w->get_raw_mouse_offset().y);
				((fan::opengl::context_t*)context)->camera.rotate_camera(offset);
			});
		}
	}
}

#include <fan/graphics/shared_inline_graphics.h>

#endif