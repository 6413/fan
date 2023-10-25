#pragma once

#include <iostream>
#include <regex>
#include <functional>
#include <ostream>
#include <fstream>
#include <string>

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

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