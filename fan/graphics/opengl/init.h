#pragma once

#if defined(fan_platform_windows)
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#define GLFW_NATIVE_INCLUDE_NONE
#endif
#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <utility>

#ifndef debug_glcall_timings
  #define debug_glcall_timings
#endif
#if defined(debug_glcall_timings)
  #include <fan/time/timer.h>
#endif

import fan.types.print;

inline int fan_track_opengl_calls = 0;
inline std::function<void(std::string func_name, uint64_t elapsed)> fan_opengl_track_print = [](std::string func_name, uint64_t elapsed){ };


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
            fan_opengl_track_print(func_name, c.elapsed()); \
          }\
        } \
      } \
      fan::time::clock c; \
    }mf; \
    return func; \
  }()

namespace fan {
  namespace opengl {
    struct opengl_t {
      int major = -1;
      int minor = -1;
      void open() {
        static uint8_t init = 1;
        if (init == 0) {
          return;
        }
        init = 0;
        if (GLenum err = glewInit() != GLEW_OK) {
          fan::throw_error(std::string("glew init error:") + std::string((const char*)glewGetErrorString(err)));
        }
      }
    };
  }

}