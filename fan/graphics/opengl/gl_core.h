#pragma once

#include <fan/types/vector.h>
#include <fan/types/color.h>
#include <fan/types/memory.h>

#include <fan/graphics/opengl/gl_init.h>
#include <fan/physics/collision/rectangle.h>

#include <fan/window/window.h>

#include <fan/graphics/camera.h>
#include <fan/graphics/opengl/gl_viewport.h>

#ifdef fan_platform_windows
#include <dbghelp.h>
#endif

namespace fan {

  static void print_callstack() {

    #ifdef fan_platform_windows
    uint16_t i;
    uint16_t frames;
    void* stack[0xff];
    SYMBOL_INFO* symbol;
    HANDLE process;

    SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_DEFERRED_LOADS | SYMOPT_INCLUDE_32BIT_MODULES);

    process = GetCurrentProcess();

    if (!SymInitialize(process, NULL, TRUE)) {
      int err = GetLastError();
      printf("[_PR_DumpTrace] SymInitialize failed %d", err);
    }

    frames = CaptureStackBackTrace(0, 0xff, stack, NULL);
    symbol = (SYMBOL_INFO*)calloc(sizeof(SYMBOL_INFO) + 1024 * sizeof(uint8_t), 1);
    symbol->MaxNameLen = 1023;
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);

    for (i = 0; i < frames; i++) {
      SymFromAddr(process, (DWORD64)(stack[i]), 0, symbol);
      DWORD Displacement;
      IMAGEHLP_LINE64 Line;
      Line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
      if (SymGetLineFromAddr64(process, (DWORD64)(stack[i]), &Displacement, &Line)) {
        printf("%i: %s:%lu\n", frames - i - 1, symbol->Name, Line.LineNumber);
      }
      else {
        printf("%i: %s:0x%llx\n", frames - i - 1, symbol->Name, symbol->Address);
      }
    }

    free(symbol);
    #endif

  }
}

#include "themes_list_builder_settings.h"
#define BLL_set_declare_NodeReference 1
#define BLL_set_declare_rest 0
#include _FAN_PATH(BLL/BLL.h)

#include "themes_list_builder_settings.h"
#define BLL_set_declare_NodeReference 0
#define BLL_set_declare_rest 1
#include _FAN_PATH(BLL/BLL.h)

namespace fan {
  namespace opengl {

    struct context_t {

      void print_version();

      struct properties_t {
        properties_t() {
          samples = 0;
        }

        uint16_t samples;
      };

      fan::opengl::GLuint current_program;
      fan::opengl::theme_list_t theme_list;
      fan::opengl::viewport_list_t viewport_list;
      fan::camera camera;
      fan::opengl::opengl_t opengl;

      context_t() {
        opengl.open();

        m_flags = 0;
        current_program = fan::uninitialized;
      }
      context_t(fan::window_t* window, const properties_t& p = properties_t()) : context_t() {
        glfwMakeContextCurrent(window->glfw_window);
      }
      ~context_t() {
      }

      void open();
      void close();

      void render(fan::window_t* window) {
        glfwSwapBuffers(window->glfw_window);
      }

      void set_depth_test(bool flag) {
        if (flag) {
          opengl.call(opengl.glEnable, fan::opengl::GL_DEPTH_TEST);
        }
        else {
          opengl.call(opengl.glDisable, fan::opengl::GL_DEPTH_TEST);
        }
      }

      void set_vsync(fan::window_t* window, bool flag) {
        glfwSwapInterval(flag);
      }

      static void message_callback(GLenum source,
        GLenum type,
        GLuint id,
        GLenum severity,
        GLsizei length,
        const GLchar* message,
        const void* userParam);

      void set_error_callback();

      void set_current(fan::window_t* window)
      {
        glfwMakeContextCurrent(window->glfw_window);
      }

      uint32_t m_flags;
    };
  }
}

//static void open_camera(fan::opengl::context_t& context, camera_t* camera, fan::vec2 window_size, const fan::vec2& x, const fan::vec2& y) {
//  camera->open(context);
//  fan::vec2 ratio = window_size / window_size.max();
//  std::swap(ratio.x, ratio.y);
//  camera->set_ortho(fan::vec2(x.x, x.y), fan::vec2(y.x, y.y));
//}

namespace fan {
  namespace opengl {
    namespace core {

      int get_buffer_size(fan::opengl::context_t& context, GLenum target_buffer, GLuint buffer_object);

      static void write_glbuffer(fan::opengl::context_t& context, GLuint buffer, const void* data, uintptr_t size, uint32_t usage, GLenum target);
      static void get_glbuffer(fan::opengl::context_t& context, void* data, GLuint buffer_id, uintptr_t size, uintptr_t offset, GLenum target);

      static void edit_glbuffer(fan::opengl::context_t& context, GLuint buffer, const void* data, uintptr_t offset, uintptr_t size, uintptr_t target);

      // not tested
      static int get_bound_buffer(fan::opengl::context_t& context);
      #pragma pack(push, 1)
      struct vao_t {

        void open(fan::opengl::context_t& context);
        void close(fan::opengl::context_t& context); const

        void bind(fan::opengl::context_t& context) const;
        void unbind(fan::opengl::context_t& context) const;

        GLuint m_buffer = -1;
      };

      struct vbo_t {

        void open(fan::opengl::context_t& context, GLenum target_);
        void close(fan::opengl::context_t& context);

        void bind(fan::opengl::context_t& context) const;

        void get_vram_instance(fan::opengl::context_t& context, void* data, uintptr_t size, uintptr_t offset);

        // only for target GL_UNIFORM_BUFFER
        void bind_buffer_range(fan::opengl::context_t& context, uint32_t total_size);

        void edit_buffer(fan::opengl::context_t& context, const void* data, uintptr_t offset, uintptr_t size);

        void write_buffer(fan::opengl::context_t& context, const void* data, uintptr_t size);

        GLuint m_buffer = -1;
        GLenum m_target = -1;
        uint32_t m_usage = fan::opengl::GL_DYNAMIC_DRAW;
      };

      #pragma pack(pop)

      struct framebuffer_t {

        struct properties_t {
          properties_t() {}
          fan::opengl::GLenum internalformat = fan::opengl::GL_DEPTH_STENCIL_ATTACHMENT;
        };

        void open(fan::opengl::context_t& context);
        void close(fan::opengl::context_t& context);

        void bind(fan::opengl::context_t& context) const;
        void unbind(fan::opengl::context_t& context) const;

        bool ready(fan::opengl::context_t& context) const;

        void bind_to_renderbuffer(fan::opengl::context_t& context, fan::opengl::GLenum renderbuffer, const properties_t& p = properties_t());

        // texture must be binded with texture.bind();
        static void bind_to_texture(fan::opengl::context_t& context, fan::opengl::GLuint texture, fan::opengl::GLenum attatchment);

        fan::opengl::GLuint framebuffer;
      };

      struct renderbuffer_t {

        struct properties_t {
          properties_t() {}
          GLenum internalformat = fan::opengl::GL_DEPTH24_STENCIL8;
          fan::vec2ui size;
        };

        void open(fan::opengl::context_t& context);
        void close(fan::opengl::context_t& context);
        void bind(fan::opengl::context_t& context) const;
        void set_storage(fan::opengl::context_t& context, const properties_t& p) const;
        void bind_to_renderbuffer(fan::opengl::context_t& context, const properties_t& p = properties_t());

        fan::opengl::GLuint renderbuffer;
      };
    }
  }
}