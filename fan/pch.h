#pragma once

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#include <iostream>
#include <regex>
#include <functional>
#include <ostream>
#include <fstream>
#include <string>

//#define loco_imgui
//#define loco_assimp

#if defined(loco_assimp)
  #include <assimp/Importer.hpp>
  #include <assimp/scene.h>
  #include <assimp/postprocess.h>
#endif

#define STRINGIFY(p0) #p0
#define STRINGIFY_DEFINE(a) STRINGIFY(a)


#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif

#ifndef FAN_INCLUDE_PATH
  #define _FAN_PATH(p0) <fan/p0>
  #else
  #define FAN_INCLUDE_PATH_END fan/
  #define _FAN_PATH(p0) <FAN_INCLUDE_PATH/fan/p0>
  #define _FAN_PATH_QUOTE(p0) STRINGIFY_DEFINE(FAN_INCLUDE_PATH) "/fan/" STRINGIFY(p0)
#endif

#if defined(loco_imgui)
#define IMGUI_IMPL_OPENGL_LOADER_CUSTOM
#define IMGUI_DEFINE_MATH_OPERATORS
#include _FAN_PATH(imgui/imgui.h)
#include _FAN_PATH(imgui/imgui_impl_opengl3.h)
#include _FAN_PATH(imgui/imgui_impl_win32.h)
#include _FAN_PATH(imgui/imgui_neo_sequencer.h)
#endif

#ifndef fan_verbose_print_level
  #define fan_verbose_print_level 1
#endif
#ifndef fan_debug
  #define fan_debug 1
#endif
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH,fan/types/types.h)

#define loco_window
#define loco_context
#define loco_legacy
//#define loco_gl_major 2
//#define loco_gl_minor 1

//#define loco_rectangle
//#define loco_light
//#define loco_line
//#define loco_circle
//#define loco_button
//#define loco_sprite
//#define loco_dropdown
//#define loco_pixel_format_renderer
//#define loco_tp
//#define loco_sprite_sheet
//
//#define loco_grass_2d
//
//// 
//#define loco_physics
/*
#define loco_cuda
#define loco_nv12
#define loco_pixel_format_renderer
*/

#include _FAN_PATH(graphics/loco.h)


#if defined(loco_imgui)
#include _FAN_PATH(graphics/gui/model_maker/maker.h)
#include _FAN_PATH(graphics/gui/keyframe_animator/loader.h)
#endif