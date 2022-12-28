#pragma once

#include _FAN_PATH(types/color.h)
#include _FAN_PATH(types/memory.h)

#include _FAN_PATH(graphics/opengl/gl_init.h)
#include _FAN_PATH(graphics/light.h)

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


  namespace opengl {

    struct context_t;
    struct viewport_t;
    struct matrices_t;

    struct cid_t {
      uint16_t bm_id;
      uint16_t block_id;
      uint8_t instance_id;
    };

  }
}

namespace fan_2d {
  namespace graphics {
    namespace gui {
      struct theme_t;
    }
  }
}

#include "themes_list_builder_settings.h"
#define BLL_set_declare_NodeReference 1
#define BLL_set_declare_rest 0
#include _FAN_PATH(BLL/BLL.h)

#include _FAN_PATH(graphics/gui/themes.h)

#include "themes_list_builder_settings.h"
#define BLL_set_declare_NodeReference 0
#define BLL_set_declare_rest 1
#include _FAN_PATH(BLL/BLL.h)

fan::opengl::theme_list_NodeReference_t::theme_list_NodeReference_t(fan_2d::graphics::gui::theme_t* theme) {
  NRI = theme->theme_reference.NRI;
}

#include "viewport_list_builder_settings.h"
#define BLL_set_declare_NodeReference 1
#define BLL_set_declare_rest 0
#include _FAN_PATH(BLL/BLL.h)

namespace fan {
  namespace opengl {

    struct viewport_t {

      void open(fan::opengl::context_t* context);
      void close(fan::opengl::context_t* context);

      fan::vec2 get_position() const
      {
        return viewport_position;
      }

      fan::vec2 get_size() const
      {
        return viewport_size;
      }

      void set(fan::opengl::context_t* context, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size);
      static void set_viewport(fan::opengl::context_t* context, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size);

      bool inside(const fan::vec2& position) const {
        return fan_2d::collision::rectangle::point_inside_no_rotation(position, viewport_position - viewport_size / 2, viewport_size * 2);
      }

      fan::vec2 viewport_position;
      fan::vec2 viewport_size;

      fan::opengl::viewport_list_NodeReference_t viewport_reference;
    };

  }
}

#include "viewport_list_builder_settings.h"
#define BLL_set_declare_NodeReference 0
#define BLL_set_declare_rest 1
#include _FAN_PATH(BLL/BLL.h)

fan::opengl::viewport_list_NodeReference_t::viewport_list_NodeReference_t(fan::opengl::viewport_t* viewport) {
  NRI = viewport->viewport_reference.NRI;
}

namespace fan {
  namespace opengl {

    struct context_t {

      struct properties_t {
        properties_t() {
          samples = 1;
          major = 3;
          minor = 2;

        }

        uint16_t samples;
        uint8_t major;
        uint8_t minor;
      };

      fan::opengl::GLuint current_program;
      fan::opengl::theme_list_t theme_list;
      fan::opengl::viewport_list_t viewport_list;
      fan::camera camera;
      fan::opengl::opengl_t opengl;

      context_t();
      context_t(fan::window_t* window, const properties_t& p = properties_t());
      ~context_t();

      void open();
      void close();

      void render(fan::window_t* window);

      void set_depth_test(bool flag);

      void set_vsync(fan::window_t* window, bool flag);

      static void message_callback(GLenum source,
        GLenum type,
        GLuint id,
        GLenum severity,
        GLsizei length,
        const GLchar* message,
        const void* userParam)
      {
        //if (type == 33361 || type == 33360) { // gl_static_draw
        //  return;
        //}
        fan::print_no_space(type == GL_DEBUG_TYPE_ERROR ? "opengl error:" : "", type, ", severity:", severity, ", message:", message);
      }

      void set_error_callback() {
        opengl.call(opengl.glEnable, GL_DEBUG_OUTPUT);
        opengl.call(opengl.glDebugMessageCallback, message_callback, (void*)0);
      }

      void set_current(fan::window_t* window);

      uint32_t m_flags;
    };
  }
}

//static void open_matrices(fan::opengl::context_t* context, matrices_t* matrices, fan::vec2 window_size, const fan::vec2& x, const fan::vec2& y) {
//  matrices->open(context);
//  fan::vec2 ratio = window_size / window_size.max();
//  std::swap(ratio.x, ratio.y);
//  matrices->set_ortho(fan::vec2(x.x, x.y), fan::vec2(y.x, y.y));
//}

namespace fan {
  namespace opengl {
    namespace core {

      static int get_buffer_size(fan::opengl::context_t* context, uint32_t target_buffer, uint32_t buffer_object) {
        int size = 0;

        context->opengl.call(context->opengl.glBindBuffer, target_buffer, buffer_object);
        context->opengl.call(context->opengl.glGetBufferParameteriv, target_buffer, fan::opengl::GL_BUFFER_SIZE, &size);

        return size;
      }

      static void write_glbuffer(fan::opengl::context_t* context, unsigned int buffer, const void* data, uintptr_t size, uint32_t usage, uintptr_t target)
      {
        context->opengl.call(context->opengl.glBindBuffer, target, buffer);

        context->opengl.call(context->opengl.glBufferData, target, size, data, usage);
        /*if (target == GL_SHADER_STORAGE_BUFFER) {
        glBindBufferBase(target, location, buffer);
        }*/
      }
      static void get_glbuffer(fan::opengl::context_t* context, void* data, uint32_t buffer_id, uintptr_t size, uintptr_t offset, uintptr_t target) {
        context->opengl.call(context->opengl.glBindBuffer, target, buffer_id);
        context->opengl.call(context->opengl.glGetBufferSubData, target, offset, size, data);
      }

      static void edit_glbuffer(fan::opengl::context_t* context, unsigned int buffer, const void* data, uintptr_t offset, uintptr_t size, uintptr_t target)
      {
        context->opengl.call(context->opengl.glBindBuffer, target, buffer);

        #if fan_debug >= fan_debug_high

        int buffer_size = get_buffer_size(context, target, buffer);

        if (buffer_size < size || (offset + size) > buffer_size) {
          fan::throw_error("tried to write more than allocated");
        }

        #endif

        context->opengl.call(context->opengl.glBufferSubData, target, offset, size, data);
        /* if (target == GL_SHADER_STORAGE_BUFFER) {
        glBindBufferBase(target, location, buffer);
        }*/
      }

      // not tested
      static int get_bound_buffer(fan::opengl::context_t* context) {
        int buffer_id;
        context->opengl.call(context->opengl.glGetIntegerv, fan::opengl::GL_VERTEX_BINDING_BUFFER, &buffer_id);
        return buffer_id;
      }
      #pragma pack(push, 1)
      struct vao_t {

        vao_t() = default;

        void open(fan::opengl::context_t* context) {
          context->opengl.call(context->opengl.glGenVertexArrays, 1, &m_vao);
        }

        void close(fan::opengl::context_t* context) {
          context->opengl.call(context->opengl.glDeleteVertexArrays, 1, &m_vao);
        }

        void bind(fan::opengl::context_t* context) const {
          context->opengl.call(context->opengl.glBindVertexArray, m_vao);
        }
        void unbind(fan::opengl::context_t* context) const {
          context->opengl.call(context->opengl.glBindVertexArray, 0);
        }

        uint32_t m_vao;

      };

      #pragma pack(pop)

      struct framebuffer_t {

        struct properties_t {
          properties_t() {}
          fan::opengl::GLenum internalformat = fan::opengl::GL_DEPTH_STENCIL_ATTACHMENT;
        };

        void open(fan::opengl::context_t* context) {
          context->opengl.call(context->opengl.glGenFramebuffers, 1, &framebuffer);
        }
        void close(fan::opengl::context_t* context) {
          context->opengl.call(context->opengl.glDeleteFramebuffers, 1, &framebuffer);
        }

        void bind(fan::opengl::context_t* context) const {
          context->opengl.call(context->opengl.glBindFramebuffer, fan::opengl::GL_FRAMEBUFFER, framebuffer);
        }
        void unbind(fan::opengl::context_t* context) const {
          context->opengl.call(context->opengl.glBindFramebuffer, fan::opengl::GL_FRAMEBUFFER, 0);
        }

        bool ready(fan::opengl::context_t* context) const {
          return context->opengl.call(context->opengl.glCheckFramebufferStatus, fan::opengl::GL_FRAMEBUFFER) == 
            fan::opengl::GL_FRAMEBUFFER_COMPLETE;
        }

        void bind_to_renderbuffer(fan::opengl::context_t* context, fan::opengl::GLenum renderbuffer, const properties_t& p = properties_t()) {
          bind(context);
          context->opengl.call(context->opengl.glFramebufferRenderbuffer, GL_FRAMEBUFFER, p.internalformat, GL_RENDERBUFFER, renderbuffer);
        }

        // texture must be binded with texture.bind();
        static void bind_to_texture(fan::opengl::context_t* context, fan::opengl::GLuint texture, fan::opengl::GLenum attatchment) {
          context->opengl.call(context->opengl.glFramebufferTexture2D, GL_FRAMEBUFFER, attatchment, GL_TEXTURE_2D, texture, 0);
        }

        fan::opengl::GLuint framebuffer;
      };

      struct renderbuffer_t {

        struct properties_t {
          properties_t() {}
          GLenum internalformat = fan::opengl::GL_DEPTH24_STENCIL8;
          fan::vec2ui size;
        };

        void open(fan::opengl::context_t* context) {
          context->opengl.call(context->opengl.glGenRenderbuffers, 1, &renderbuffer);
          //set_storage(context, p);
        }
        void close(fan::opengl::context_t* context) {
          context->opengl.call(context->opengl.glDeleteRenderbuffers, 1, &renderbuffer);
        }
        void bind(fan::opengl::context_t* context) const {
          context->opengl.call(context->opengl.glBindRenderbuffer, fan::opengl::GL_RENDERBUFFER, renderbuffer);
        }
        void set_storage(fan::opengl::context_t* context, const properties_t& p) const {
          bind(context);
          context->opengl.call(context->opengl.glRenderbufferStorage, fan::opengl::GL_RENDERBUFFER, p.internalformat, p.size.x, p.size.y);
        }
        void bind_to_renderbuffer(fan::opengl::context_t* context, const properties_t& p = properties_t()) {
          bind(context);
          context->opengl.call(context->opengl.glFramebufferRenderbuffer, GL_FRAMEBUFFER, p.internalformat, GL_RENDERBUFFER, renderbuffer);
        }

        fan::opengl::GLuint renderbuffer;
      };
    }
  }
}

inline fan::opengl::context_t::context_t() {
  theme_list.Open();
  viewport_list.Open();

  opengl.open();

  m_flags = 0;
  current_program = fan::uninitialized;
}

inline fan::opengl::context_t::context_t(fan::window_t* window, const properties_t& p) : context_t() {

  #if defined(fan_platform_windows)

  window->m_hdc = GetDC(window->m_window_handle);

  int pixel_format_attribs[19] = {
    WGL_DRAW_TO_WINDOW_ARB, fan::opengl::GL_TRUE,
    WGL_SUPPORT_OPENGL_ARB, fan::opengl::GL_TRUE,
    WGL_DOUBLE_BUFFER_ARB, fan::opengl::GL_TRUE,
    WGL_PIXEL_TYPE_ARB, WGL_TYPE_RGBA_ARB,
    WGL_COLOR_BITS_ARB, 32,
    WGL_DEPTH_BITS_ARB, 24,
    WGL_STENCIL_BITS_ARB, 8,
    WGL_SAMPLE_BUFFERS_ARB, true, // Number of buffers (must be 1 at time of writing)
    WGL_SAMPLES_ARB, p.samples,        // Number of samples
    0
  };
  if (!p.samples) {
    // set back to zero to disable antialising
    for (int i = 0; i < 4; i++) {
      pixel_format_attribs[14 + i] = 0;
    }
  }
  int pixel_format;
  UINT num_formats;

  // /mtd or enable maybe /LTCG to fix with sanitizer on
  opengl.call(opengl.internal.wglChoosePixelFormatARB, window->m_hdc, pixel_format_attribs, (float*)0, 1, &pixel_format, &num_formats);

  if (!num_formats) {
    fan::throw_error("failed to choose pixel format:" + fan::to_string(GetLastError()));
  }

  PIXELFORMATDESCRIPTOR pfd;
  memset(&pfd, 0, sizeof(pfd));
  if (!DescribePixelFormat(window->m_hdc, pixel_format, sizeof(pfd), &pfd)) {
    fan::throw_error("failed to describe pixel format:" + fan::to_string(GetLastError()));
  }
  if (!SetPixelFormat(window->m_hdc, pixel_format, &pfd)) {
    fan::throw_error("failed to set pixel format:" + fan::to_string(GetLastError()));
  }

  const int gl_attributes[] = {
    WGL_CONTEXT_MINOR_VERSION_ARB, p.minor,
    WGL_CONTEXT_MAJOR_VERSION_ARB, p.major,
    WGL_CONTEXT_PROFILE_MASK_ARB,  WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
    0,
  };

  if (!window->m_context) {

    if (p.major < 3) {
      window->m_context = wglCreateContext(window->m_hdc);
    }
    else {
      window->m_context = opengl.call(opengl.internal.wglCreateContextAttribsARB, window->m_hdc, (HGLRC)0, gl_attributes);
    }

    if (!window->m_context) {
      fan::print("failed to create context");
      exit(1);
    }
  }

  if (!wglMakeCurrent(window->m_hdc, window->m_context)) {
    fan::print("failed to make current");
    exit(1);
  }

  if (wglGetCurrentContext() != window->m_context) {
    wglMakeCurrent(window->m_hdc, window->m_context);
  }

  #elif defined(fan_platform_unix)

  if (opengl.internal.glXGetCurrentContext() != window->m_context) {
    opengl.internal.glXMakeCurrent(fan::sys::m_display, window->m_window_handle, window->m_context);
  }

  #endif

  opengl.call(opengl.glEnable, GL_BLEND);
  opengl.call(opengl.glBlendFunc, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  set_depth_test(true);

  // opengl.call(opengl.glFrontFace, GL_CCW);
  //opengl.glEnable(GL_CULL_FACE);;
  //opengl.glCullFace(GL_FRONT);
  #if fan_debug >= fan_debug_high
  context_t::set_error_callback();
  #endif
}

inline fan::opengl::context_t::~context_t() {
  theme_list.Close();
  viewport_list.Close();
}

inline void fan::opengl::context_t::render(fan::window_t* window) {
  #ifdef fan_platform_windows
  SwapBuffers(window->m_hdc);
  #elif defined(fan_platform_unix)
  opengl.internal.glXSwapBuffers(fan::sys::m_display, window->m_window_handle);
  #endif
}

inline void fan::opengl::context_t::set_depth_test(bool flag) {
  if (flag) {
    opengl.call(opengl.glEnable, fan::opengl::GL_DEPTH_TEST);
  }
  else {
    opengl.call(opengl.glDisable, fan::opengl::GL_DEPTH_TEST);
  }
}

inline void fan::opengl::context_t::set_vsync(fan::window_t* window, bool flag)
{
  #if defined(fan_platform_windows)

  wglMakeCurrent(window->m_hdc, window->m_context);

  #elif defined(fan_platform_unix)

  opengl.internal.glXMakeCurrent(fan::sys::m_display, window->m_window_handle, window->m_context);

  #endif

  #ifdef fan_platform_windows
  if (opengl.internal.wglSwapIntervalEXT) {
    opengl.call(opengl.internal.wglSwapIntervalEXT, flag);
  }

  #elif defined(fan_platform_unix)
  opengl.internal.glXSwapIntervalEXT(fan::sys::m_display, opengl.internal.glXGetCurrentDrawable(), flag);
  #endif
}

inline void fan::opengl::context_t::set_current(fan::window_t* window)
{
  #ifdef fan_platform_windows
    wglMakeCurrent(window->m_hdc, window->m_context);
  #elif defined(fan_platform_unix)
    opengl.internal.glXMakeCurrent(fan::sys::m_display, window->m_window_handle, window->m_context);
  #endif
}

inline void fan::opengl::viewport_t::open(fan::opengl::context_t * context) {
  viewport_reference = context->viewport_list.NewNode();
  context->viewport_list[viewport_reference].viewport_id = this;
}

inline void fan::opengl::viewport_t::close(fan::opengl::context_t * context) {
  context->viewport_list.Recycle(viewport_reference);
}

void fan::opengl::viewport_t::set(fan::opengl::context_t* context, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size)  {
  viewport_position = viewport_position_;
  viewport_size = viewport_size_;

  context->opengl.call(
    context->opengl.glViewport, 
    viewport_position.x,
    window_size.y - viewport_size_.y - viewport_position.y,
    viewport_size.x, viewport_size.y
  );
}

inline void fan::opengl::viewport_t::set_viewport(fan::opengl::context_t* context, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
  context->opengl.call(
    context->opengl.glViewport,
    viewport_position_.x,
    window_size.y - viewport_size_.y - viewport_position_.y,
    viewport_size_.x, viewport_size_.y
  );
}

#include _FAN_PATH(graphics/opengl/uniform_block.h)