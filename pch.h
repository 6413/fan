#pragma once

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#if defined(fan_build_pch)
  #include _INCLUDE_TOKEN(WITCH_INCLUDE_PATH,WITCH.h)
#endif

#include <iostream>
#include <regex>
#include <functional>
#include <ostream>
#include <fstream>
#include <string>

#define loco_imgui

#if defined(loco_imgui)
#define IMGUI_IMPL_OPENGL_LOADER_CUSTOM
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#endif

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH,fan/types/types.h)

#define loco_window
#define loco_context

#define loco_light
#define loco_line
#define loco_circle
#define loco_button
#define loco_sprite
#define loco_dropdown
#define loco_pixel_format_renderer
#define loco_tp
// 
#define loco_physics
/*
#define loco_cuda
#define loco_nv12
#define loco_pixel_format_renderer
*/

#include _FAN_PATH(graphics/loco.h)
