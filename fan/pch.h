#pragma once

#if defined(fan_3d)
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

#include <fan/types/types.h>


//#define loco_vulkan
//#define loco_compute_shader

import fan.types.vector;
import fan.types.color;

//
#if !defined(fan_gui)
  import fan.graphics.loco;
  import fan.graphics;
#else
  import fan.graphics.loco;
  import fan.graphics;
  import fan.graphics.gui;
#endif
#if defined(fan_physics)
  import fan.graphics.physics_shapes;
#endif