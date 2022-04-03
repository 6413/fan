#pragma once

#include <fan/types/types.h>

#include <fan/graphics/renderer.h>

#if fan_renderer == fan_renderer_opengl
	
	#include <fan/graphics/opengl/gl_gui.h>

#elif fan_renderer == fan_renderer_vulkan
	
	#include <fan/graphics/vulkan/vk_gui.h>

#endif

namespace fan_2d {
	namespace graphics {
		namespace gui {
			#if fan_renderer == fan_renderer_opengl
	
				using fan_2d::opengl::gui::text_renderer_t;
				using fan_2d::opengl::gui::text_renderer0_t;

			#elif fan_renderer == fan_renderer_vulkan
	

			#endif
		}
	}
}