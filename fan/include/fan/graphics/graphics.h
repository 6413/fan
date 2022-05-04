#pragma once

#include <fan/types/types.h>

#include <fan/graphics/renderer.h>

#if fan_renderer == fan_renderer_opengl
	#include <fan/graphics/opengl/gl_graphics.h>
#else
	#include <fan/graphics/vulkan/vk_graphics.h>
#endif

namespace fan {
	namespace graphics {

		#if fan_renderer == fan_renderer_opengl
			using fan::opengl::load_image;
		#endif

	}
}

namespace fan {
	namespace graphics {

		#if fan_renderer == fan_renderer_opengl
			using fan::opengl::context_t;
			using fan::opengl::matrices_t;
			using fan::opengl::viewport_t;
		#endif


	}
}

namespace fan_2d {
	namespace graphics {

		#if fan_renderer == fan_renderer_opengl
			using fan_2d::opengl::line_t;
			using fan_2d::opengl::rectangle_t;
			using fan_2d::opengl::circle_t;
			using fan_2d::opengl::sprite_t;

		#endif


	}
}