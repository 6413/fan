#pragma once

#include <fan/utility.h>

#if defined(fan_platform_windows)
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#define GLFW_NATIVE_INCLUDE_NONE
#endif

#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <utility>
#include <string>
#include <functional>

inline int& fan_track_opengl_calls() {
  static int track = 0;
  return track;
}
inline std::function<void(std::string func_name, uint64_t elapsed)>& fan_opengl_track_print() {
  static std::function<void(std::string func_name, uint64_t elapsed)> cb = [](std::string func_name, uint64_t elapsed){ };
  return cb;
}
#define fan_opengl_call(func) \
  [&]() { \
    struct measure_func_t { \
      measure_func_t() { \
        c.start(); \
      }\
      ~measure_func_t() { \
        if (fan_track_opengl_calls) { \
          glFlush(); \
          glFinish(); \
          if (c.elapsed() / 1e+9 > 0.01) {\
            std::string func_call = #func; \
            std::string func_name = func_call.substr(0, func_call.find('(')); \
            fan_opengl_track_print()(func_name, c.elapsed()); \
          }\
        } \
      } \
      fan::time::timer c; \
    }mf; \
    return func; \
  }()