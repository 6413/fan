#pragma once

#include <fan/types/print.h>
#if defined(fan_platform_windows)
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#define GLFW_NATIVE_INCLUDE_NONE
#endif
#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <utility>

#define fan_opengl_call(x) x

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
          fan::throw_error(std::string("glew init error:") + (const char*)glewGetErrorString(err));
        }
      }
    };
  }

}