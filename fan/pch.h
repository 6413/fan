#pragma once

#if defined(loco_assimp)
  #define loco_model_3d
  #include <assimp/Importer.hpp>
  #include <assimp/scene.h>
  #include <assimp/postprocess.h>
#endif

#ifndef fan_verbose_print_level
  #define fan_verbose_print_level 1
#endif

#if defined(loco_imgui) && !defined(text_editor_include)
  #include <fan/imgui/text_editor.h>
#endif

#include <fan/memory/memory.hpp>

#include <fan/types/types.h>

//#define loco_vulkan
//#define loco_compute_shader

//
#include <fan/graphics/loco.h>
