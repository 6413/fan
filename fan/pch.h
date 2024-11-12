#pragma once

//#define loco_assimp

#define loco_model_3d

#if defined(loco_assimp)
  #include <assimp/Importer.hpp>
  #include <assimp/scene.h>
  #include <assimp/postprocess.h>
#endif

#ifndef fan_verbose_print_level
  #define fan_verbose_print_level 1
#endif


#include <fan/types/types.h>

//#define loco_vulkan
//#define loco_compute_shader

//
#include <fan/graphics/loco.h>


#if defined(loco_imgui) && defined(loco_vfi) && !defined(loco_no_model_maker)
  #include <fan/graphics/gui/model_maker/maker.h>
#endif

#if defined(loco_model_3d)
#include <fan/graphics/opengl/3D/objects/model.h>
#endif

