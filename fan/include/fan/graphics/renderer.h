#pragma once

#include <fan/types/types.h>

#define fan_renderer_opengl 0
#define fan_renderer_vulkan 1

// make clean when changing

#define fan_set_graphics_renderer fan_renderer_opengl

#ifndef fan_set_graphics_renderer
	#define fan_set_graphics_renderer fan_renderer_vulkan
#endif
//
#define fan_renderer fan_set_graphics_renderer

#if fan_renderer == fan_renderer_opengl
	#include <glad/gl.h>

	#ifdef fan_platform_windows

		#include <wgl/wgl.h>

		#pragma comment(lib, "User32.lib")

		#pragma comment(lib, "opengl32.lib")

		#pragma comment(lib, "Gdi32.lib")

		#undef min
		#undef max
	#endif

#else

#include <fan/vulkan.h>

#endif