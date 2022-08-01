#pragma once

#include _FAN_PATH(types/types.h)

#include _FAN_PATH(graphics/renderer.h)

#include _FAN_PATH(types/color.h)
#include _FAN_PATH(graphics/camera.h)
#include _FAN_PATH(window/window.h)
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
        printf("%i: %s:%u\n", frames - i - 1, symbol->Name, Line.LineNumber);
      }
      else {
        printf("%i: %s:0x%x\n", frames - i - 1, symbol->Name, symbol->Address);
      }
    }

    free(symbol);
#endif

  }


  namespace opengl {

    struct context_t;
    struct matrices_t;

    struct cid_t {
      uint32_t id;
    };

  }
}

#define BLL_set_BaseLibrary 1
#define BLL_set_namespace fan::opengl
#define BLL_set_prefix image_list
#define BLL_set_type_node uint8_t
#define BLL_set_node_data fan::opengl::GLuint texture_id;
#define BLL_set_Link 0
#include _FAN_PATH(BLL/BLL.h)

#define BLL_set_BaseLibrary 1
#define BLL_set_namespace fan::opengl
#define BLL_set_prefix matrices_list
#define BLL_set_type_node uint8_t
#define BLL_set_node_data fan::opengl::matrices_t* matrices_id;
#define BLL_set_Link 0
#define BLL_set_declare_basic_types 1
#define BLL_set_declare_rest 0
#define BLL_set_KeepSettings 1
#define BLL_set_StructFormat 1
#define BLL_set_NodeReference_Overload_Declare \
  void operator=(fan::opengl::matrices_t* matrices);
#include _FAN_PATH(BLL/BLL.h)

namespace fan {
  namespace opengl {

    namespace core {
      struct uniform_block_common_t;
    }

    struct matrices_t {

      void open(fan::opengl::context_t* context);
      void close(fan::opengl::context_t* context);

      fan::vec3 get_camera_position() const {
        return camera_position;
      }
      void set_camera_position(const fan::vec3& cp) {
        camera_position = cp;

        m_view[3][0] = 0;
        m_view[3][1] = 0;
        m_view[3][2] = 0;
        m_view = m_view.translate(camera_position);
        fan::vec3 position = m_view.get_translation();
        constexpr fan::vec3 front(0, 0, 1);

        m_view = fan::math::look_at_left<fan::mat4>(position, position + front, fan::camera::world_up);
      }

      void set_ortho(const fan::vec2& x, const fan::vec2& y) {
        m_projection = fan::math::ortho<fan::mat4>(
          x.x,
          x.y,
          y.y,
          y.x,
          0.1,
          100.0
          );

        m_view[3][0] = 0;
        m_view[3][1] = 0;
        m_view[3][2] = 0;
        m_view = m_view.translate(camera_position);
        fan::vec3 position = m_view.get_translation();
        constexpr fan::vec3 front(0, 0, 1);

        m_view = fan::math::look_at_left<fan::mat4>(position, position + front, fan::camera::world_up);
      }
      
      fan::mat4 m_projection;
      // temporary
      fan::mat4 m_view;

      fan::vec3 camera_position;

      matrices_list_NodeReference_t matrices_reference;
    };

    static void open_matrices(fan::opengl::context_t* context, matrices_t* matrices, fan::vec2 window_size, const fan::vec2& x, const fan::vec2& y);
  }
}

#define BLL_set_declare_basic_types 0
#define BLL_set_declare_rest 1
#define BLL_set_KeepSettings 0
#undef BLL_set_NodeReference_Overload_Declare
#include _FAN_PATH(BLL/BLL.h)

void fan::opengl::matrices_list_NodeReference_t::operator=(fan::opengl::matrices_t* matrices) {
  NRI = matrices->matrices_reference.NRI;
}

namespace fan {
  namespace opengl {

    struct context_t {

      struct properties_t {
        properties_t() {
          samples = 1;
          major = 3;
          minor = 1;
          
        }

        uint16_t samples;
        uint8_t major;
        uint8_t minor;
      };

      fan::opengl::image_list_t image_list;
      fan::opengl::matrices_list_t matrices_list;
      fan::camera camera;
      fan::vec2 viewport_position;
      fan::vec2 viewport_size;
      fan::opengl::opengl_t opengl;

      typedef void(*draw_cb_t)(context_t*, void*);

      struct draw_queue_t {
        void* data;
        draw_cb_t draw_cb;
      };

      bll_t<draw_queue_t> m_draw_queue;
      bll_t<core::uniform_block_common_t*> m_write_queue;

      void open();
      void close();

      void bind_to_window(fan::window_t* window, const properties_t& p = properties_t());

      fan::vec2 get_viewport_position() const;
      fan::vec2 get_viewport_size() const;
      void set_viewport(const fan::vec2& viewport_position, const fan::vec2& viewport_size_);

      void process();

      void render(fan::window_t* window);

      uint32_t enable_draw(void* data, draw_cb_t);
      void disable_draw(uint32_t node_reference);

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


      struct uniform_block_common_t {
        uint32_t m_vbo;
        fan::opengl::core::vao_t m_vao;
        uint32_t buffer_bytes_size;
        uint32_t m_size;

        void open(fan::opengl::context_t* context) {

          m_edit_index = fan::uninitialized;

          m_min_edit = 0xffffffff;
          m_max_edit = 0x00000000;

          m_size = 0;
          m_vao.open(context);
        }
        void close(fan::opengl::context_t* context) {
          if (is_queued()) {
            context->m_write_queue.erase(m_edit_index);
            reset_edit();
          }
          m_vao.close(context);
        }

        bool is_queued() const {
          return m_edit_index != fan::uninitialized;
        }

        void edit(fan::opengl::context_t* context, uint32_t begin, uint32_t end) {

          m_min_edit = std::min(m_min_edit, begin);
          m_max_edit = std::max(m_max_edit, end);

          if (is_queued()) {
            return;
          }
          m_edit_index = context->m_write_queue.push_back(this);

         // context->process();
        }

        void on_edit(fan::opengl::context_t* context) {
          reset_edit();
        }

        void reset_edit() {
          m_min_edit = 0xffffffff;
          m_max_edit = 0x00000000;

          m_edit_index = fan::uninitialized;
        }

        uint32_t m_edit_index;

        uint32_t m_min_edit;
        uint32_t m_max_edit;
      };

      template <typename type_t, uint32_t element_size>
      struct uniform_block_t {

        static constexpr uint32_t element_byte_size = element_size;

        uniform_block_t() = default;

        struct open_properties_t {
          open_properties_t() {}

          uint32_t target = fan::opengl::GL_UNIFORM_BUFFER;
          uint32_t usage = fan::opengl::GL_DYNAMIC_DRAW;
        }op;

        void open(fan::opengl::context_t* context, open_properties_t op_ = open_properties_t()) {
          context->opengl.call(context->opengl.glGenBuffers, 1, &common.m_vbo);
          op = op_;
          common.open(context);
          common.buffer_bytes_size = sizeof(type_t);
          fan::opengl::core::write_glbuffer(context, common.m_vbo, 0, sizeof(type_t) * element_size, op.usage, op.target);
        }

        void close(fan::opengl::context_t* context) {
#if fan_debug >= fan_debug_low
          if (common.m_vbo == -1) {
            fan::throw_error("tried to remove non existent vbo");
          }
#endif
          context->opengl.call(context->opengl.glDeleteBuffers, 1, &common.m_vbo);

          common.close(context);
        }

        void bind_buffer_range(fan::opengl::context_t* context, uint32_t bytes_size) {
          context->opengl.call(context->opengl.glBindBufferRange, fan::opengl::GL_UNIFORM_BUFFER, 0, common.m_vbo, 0, bytes_size * sizeof(type_t));
        }

        void bind(fan::opengl::context_t* context) const {
          context->opengl.call(context->opengl.glBindBuffer, op.target, common.m_vbo);
        }
        void unbind() const {
          //glBindBuffer(GL_ARRAY_BUFFER, 0);
        }

        void push_ram_instance(fan::opengl::context_t* context, const type_t& data) {
          std::memmove(&buffer[common.m_size], (void*)&data, common.buffer_bytes_size);
          common.m_size += sizeof(type_t);
        }

        // uniform block preallocates
        /*void write_vram_all(fan::opengl::context_t* context) {

          common.m_vao.bind(context);

          this->bind(context);

          fan::opengl::core::write_glbuffer(context, common.m_vbo, (void*)buffer, common.m_size, op.usage, op.target);
        }*/

        type_t* get_instance(fan::opengl::context_t* context, uint32_t i) {
          return (type_t*)&buffer[i * sizeof(type_t)];
        }
        void get_vram_instance(fan::opengl::context_t* context, type_t* data, uint32_t i) {
          fan::opengl::core::get_glbuffer(context, data, common.m_vbo, sizeof(type_t), i * sizeof(type_t), op.target);
        }
        void edit_ram_instance(fan::opengl::context_t* context, uint32_t i, const void* data, uint32_t byte_offset, uint32_t sizeof_data) {
#if fan_debug >= fan_debug_low
          if (i + byte_offset + sizeof_data > common.m_size) {
            fan::throw_error("invalid access");
          }
#endif
          std::memmove(buffer + i * sizeof(type_t) + byte_offset, data, sizeof_data);
        }

        void init_uniform_block(fan::opengl::context_t* context, uint32_t program, const char* name, uint32_t buffer_index = 0) {
          uint32_t index = context->opengl.call(context->opengl.glGetUniformBlockIndex, program, name);
#if fan_debug >= fan_debug_low
          if (index == fan::uninitialized) {
            fan::throw_error(std::string("failed to initialize uniform block:") + name);
          }
#endif

          context->opengl.call(context->opengl.glUniformBlockBinding, program, index, buffer_index);
        }

        void draw(fan::opengl::context_t* context, uint32_t begin, uint32_t count) {

          common.m_vao.bind(context);

          // possibly disable depth test here
          context->opengl.call(context->opengl.glDrawArrays, fan::opengl::GL_TRIANGLES, begin, count);
        }

        uint32_t size() const {
          return common.m_size / sizeof(type_t);
        }

        uniform_block_common_t common;
        uint8_t buffer[element_size * sizeof(type_t)];
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
        void bind_to_texture(fan::opengl::context_t* context, fan::opengl::GLuint texture) {
          context->opengl.call(context->opengl.glFramebufferTexture2D, GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
        }

        fan::opengl::GLuint framebuffer;
      };

      struct renderbuffer_t {

        struct properties_t {
          properties_t() {}
          GLenum internalformat = fan::opengl::GL_DEPTH24_STENCIL8;
          fan::vec2ui size;
        };

        void open(fan::opengl::context_t* context, const properties_t& p) {
          context->opengl.call(context->opengl.glGenRenderbuffers, 1, &renderbuffer);
          set_storage(context, p);
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

        fan::opengl::GLuint renderbuffer;
      };
    }
  }
}

inline void fan::opengl::context_t::open() {
  image_list_open(&image_list);
  matrices_list_open(&matrices_list);

  m_draw_queue.open();
  m_write_queue.open();

  opengl.open();

  m_flags = 0;
}
inline void fan::opengl::context_t::close() {
  image_list_close(&image_list);
  matrices_list_close(&matrices_list);

  m_draw_queue.close();
  m_write_queue.close();
}

inline void fan::opengl::context_t::bind_to_window(fan::window_t* window, const properties_t& p) {

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

  opengl.call(opengl.internal.wglChoosePixelFormatARB, window->m_hdc, pixel_format_attribs, (float*)0, 1, &pixel_format, &num_formats);

  if (!num_formats) {
    fan::throw_error("failed to choose pixel format:" + std::to_string(GetLastError()));
  }

  PIXELFORMATDESCRIPTOR pfd;
  memset(&pfd, 0, sizeof(pfd));
  if (!DescribePixelFormat(window->m_hdc, pixel_format, sizeof(pfd), &pfd)) {
    fan::throw_error("failed to describe pixel format:" + std::to_string(GetLastError()));
  }
  if (!SetPixelFormat(window->m_hdc, pixel_format, &pfd)) {
    fan::throw_error("failed to set pixel format:" + std::to_string(GetLastError()));
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

  #if fan_debug >= fan_debug_high
    context_t::set_error_callback();
  #endif
}

inline fan::vec2 fan::opengl::context_t::get_viewport_position() const
{
  return viewport_position;
}

inline fan::vec2 fan::opengl::context_t::get_viewport_size() const
{
  return viewport_size;
}

inline void fan::opengl::context_t::set_viewport(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_) {
  opengl.call(opengl.glViewport, viewport_position_.x, viewport_position_.y, viewport_size_.x, viewport_size_.y);
  viewport_position = viewport_position_;
  viewport_size = viewport_size_;
}

inline void fan::opengl::context_t::process() {
#if fan_renderer == fan_renderer_opengl

  opengl.call(opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT | fan::opengl::GL_DEPTH_BUFFER_BIT);


#endif

  uint32_t it = m_write_queue.begin();

  while (it != m_write_queue.end()) {

    uint64_t src = m_write_queue[it]->m_min_edit;
    uint64_t dst = m_write_queue[it]->m_max_edit;

    uint8_t* buffer = (uint8_t*)&m_write_queue[it][1];

    buffer += src;

    fan::opengl::core::edit_glbuffer(this, m_write_queue[it]->m_vbo, buffer, src, dst - src, fan::opengl::GL_UNIFORM_BUFFER);

    m_write_queue[it]->on_edit(this);

    it = m_write_queue.next(it);
  }

  m_write_queue.clear();

  it = m_draw_queue.begin();

  while (it != m_draw_queue.end()) {
    m_draw_queue.start_safe_next(it);
    m_draw_queue[it].draw_cb(this, m_draw_queue[it].data);
    it = m_draw_queue.end_safe_next();
  }
}

inline void fan::opengl::context_t::render(fan::window_t* window) {
#ifdef fan_platform_windows
  SwapBuffers(window->m_hdc);
#elif defined(fan_platform_unix)
  opengl.internal.glXSwapBuffers(fan::sys::m_display, window->m_window_handle);
#endif
}

inline uint32_t fan::opengl::context_t::enable_draw(void* data, draw_cb_t cb)
{
  return m_draw_queue.push_back(fan::opengl::context_t::draw_queue_t{ data, cb });
}

inline void fan::opengl::context_t::disable_draw(uint32_t node_reference)
{
  m_draw_queue.erase(node_reference);
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

  opengl.call(opengl.internal.wglSwapIntervalEXT, flag);

#elif defined(fan_platform_unix)
  opengl.internal.glXSwapIntervalEXT(fan::sys::m_display, opengl.internal.glXGetCurrentDrawable(), flag);
#endif
}

void fan::opengl::matrices_t::open(fan::opengl::context_t* context) {
  m_view = fan::mat4(1);
  camera_position = 0;
  matrices_reference = matrices_list_NewNode(&context->matrices_list);
}
void fan::opengl::matrices_t::close(fan::opengl::context_t* context) {
  matrices_list_Recycle(&context->matrices_list, matrices_reference);
}

void fan::opengl::open_matrices(fan::opengl::context_t* context, fan::opengl::matrices_t* matrices, fan::vec2 window_size, const fan::vec2& x, const fan::vec2& y) {
  matrices->open(context);
  fan::vec2 ratio = window_size / window_size.max();
  std::swap(ratio.x, ratio.y);
  matrices->set_ortho(fan::vec2(x.x, x.y), fan::vec2(y.x, y.y));
}