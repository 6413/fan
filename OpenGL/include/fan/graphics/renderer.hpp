#pragma once

#define fan_renderer_opengl 0
#define fan_renderer_vulkan 1

#define fan_set_graphics_renderer fan_renderer_vulkan

#ifndef fan_set_graphics_renderer
	#define fan_set_graphics_renderer fan_renderer_vulkan
#endif

#define fan_renderer fan_set_graphics_renderer 