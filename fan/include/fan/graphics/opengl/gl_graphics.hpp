#pragma once

#include <fan/types/types.hpp>

#include <fan/graphics/renderer.hpp>

#if fan_renderer == fan_renderer_opengl

#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/Importer.hpp>

#include <fan/graphics/opengl/gl_core.hpp>

#include <fan/graphics/shared_core.hpp>

#include <fan/graphics/shared_graphics.hpp>

#include <fan/graphics/webp.h>

#ifdef fan_platform_windows
	#pragma comment(lib, "lib/assimp/assimp.lib")
#endif

#include <fan/graphics/opengl/objects/rectangle.h>
#include <fan/graphics/opengl/objects/sprite.h>
#include <fan/graphics/opengl/objects/sprite0.h>
#include <fan/graphics/opengl/objects/yuv420p_renderer.h>

#include <fan/graphics/shared_inline_graphics.hpp>

#endif