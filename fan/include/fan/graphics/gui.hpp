#pragma once

#include <fan/types/types.hpp>

#include <fan/graphics/renderer.hpp>

#if fan_renderer == fan_renderer_opengl
	
	#include <fan/graphics/opengl/gl_gui.hpp>

#elif fan_renderer == fan_renderer_vulkan
	
	#include <fan/graphics/vulkan/vk_gui.hpp>

#endif