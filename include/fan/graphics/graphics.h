#pragma once

#include _FAN_PATH(types/types.h)

#include _FAN_PATH(graphics/renderer.h)

#if fan_renderer == fan_renderer_opengl
	#include _FAN_PATH(graphics/opengl/gl_graphics.h)
#else
	#include _FAN_PATH(graphics/vulkan/vk_graphics.h)
#endif

namespace fan {
	namespace graphics {

		#if fan_renderer == fan_renderer_opengl
			using fan::opengl::context_t;
			using fan::opengl::matrices_t;
			using fan::opengl::viewport_t;
			using fan::opengl::open_matrices;
		#endif
	}
}