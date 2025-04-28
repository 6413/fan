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
#include <fan/types/vector.h>
#include <fan/types/matrix.h>
#include <fan/types/quaternion.h>


//#define loco_vulkan
//#define loco_compute_shader

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

struct fan_window_loop_t{
  fan_window_loop_t(const auto& lambda) {
    gloco->loop(lambda);
  }
};

// static called inside scope, so its fine for linking
#define fan_window_loop \
  static fan_window_loop_t __fan_window_loop_entry = [&]()

#define fan_window_close() \
  gloco->close(); \
  return