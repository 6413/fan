#pragma once

#include <fan/types/types.h>

#include <fan/graphics/renderer.h>

#include <fan/graphics/opengl/gl_defines.h>

namespace fan_2d {

    namespace opengl {

    enum class shape {
			line,
			line_strip,
			triangle,
			triangle_strip,
			triangle_fan,
			last
		};

		enum class face_e {
#if fan_renderer == fan_renderer_opengl
			front = fan::opengl::GL_FRONT,
			back = fan::opengl::GL_BACK,
			front_and_back = fan::opengl::GL_FRONT_AND_BACK
#elif fan_renderer == fan_renderer_vulkan
			front = VK_CULL_MODE_FRONT_BIT,
			back = VK_CULL_MODE_BACK_BIT,
			front_and_back = VK_CULL_MODE_FRONT_AND_BACK,
			none = VK_CULL_MODE_NONE
#endif
		};

		enum class fill_mode_e {

#if fan_renderer == fan_renderer_opengl

			point = fan::opengl::GL_POINT,
			line = fan::opengl::GL_LINE,
			fill = fan::opengl::GL_FILL

#elif fan_renderer == fan_renderer_vulkan
			point = VK_POLYGON_MODE_POINT,
			line = VK_POLYGON_MODE_LINE,
			fill = VK_POLYGON_MODE_FILL
#endif

		};

    }

}