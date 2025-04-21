#pragma once

#if defined(fan_3d)
  #define loco_model_3d
  #include <assimp/Importer.hpp>
  #include <assimp/scene.h>
  #include <assimp/postprocess.h>
#endif

#ifndef fan_verbose_print_level
  #define fan_verbose_print_level 1
#endif

#if defined(fan_gui) && !defined(text_editor_include)
  #include <fan/imgui/text_editor.h>
#endif

#include <fan/memory/memory.hpp>

#include <fan/types/types.h>

//#define loco_vulkan
//#define loco_compute_shader

//
#if !defined(fan_gui)
  #include <fan/graphics/graphics.h>
#else
  #include <fan/graphics/gui/gui.h>
#endif

