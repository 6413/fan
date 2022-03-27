#pragma once

#include <fan/types/types.h>

#include <fan/graphics/renderer.h>

#if fan_renderer == fan_renderer_opengl
	#include <fan/graphics/opengl/gl_graphics.h>
#else
	#include <fan/graphics/vulkan/vk_graphics.h>
#endif