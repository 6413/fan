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

//#define loco_post_process
#define loco_rectangle
#define loco_button
#include _FAN_PATH(graphics/loco.h)