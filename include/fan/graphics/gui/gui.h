#pragma once

#include _FAN_PATH(types/types.h)

#include _FAN_PATH(graphics/renderer.h)

#if fan_renderer == fan_renderer_opengl
	
	#include _FAN_PATH(graphics/opengl/gl_gui.h)

#elif fan_renderer == fan_renderer_vulkan
	
	#include _FAN_PATH(graphics/vulkan/vk_gui.h)

#endif

#include _FAN_PATH(graphics/gui/be.h)

namespace fan_2d {
	namespace graphics {
		namespace gui {
			#if fan_renderer == fan_renderer_opengl
	

			#elif fan_renderer == fan_renderer_vulkan
	

			#endif
		}
	}
}