#pragma once

#include <fan/types/types.hpp>

#include <fan/graphics/renderer.hpp>

#if fan_renderer == fan_renderer_opengl
	#include <fan/graphics/opengl/gl_graphics.hpp>
#else
	#include <fan/graphics/vulkan/vk_graphics.hpp>
#endif