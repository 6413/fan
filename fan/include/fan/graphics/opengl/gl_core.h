#pragma once

#include <fan/types/types.h>

#include <fan/graphics/renderer.h>

#include <fan/types/color.h>
#include <fan/graphics/camera.h>
#include <fan/window/window.h>
#include <fan/types/memory.h>

#include <fan/graphics/opengl/gl_init.h>
#include <fan/graphics/light.h>

#ifdef fan_platform_windows
  #include <dbghelp.h>
#endif

namespace fan {

  static void print_callstack() {

  #ifdef fan_platform_windows
      uint16_t i;
      uint16_t frames;
      void *stack[0xff];
      SYMBOL_INFO *symbol;
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

  }

  namespace opengl{

    namespace core {

      struct glsl_buffer_t;

      struct queue_helper_t {

        queue_helper_t() = default;

        void open();

        void close(fan::opengl::context_t* context);

        bool is_queued() const;

        void edit(fan::opengl::context_t* context, uint32_t begin, uint32_t end, glsl_buffer_t* buffer);

        void on_edit(fan::opengl::context_t* context);

        void reset_edit();

        uint32_t m_edit_index;

        uint32_t m_min_edit;
        uint32_t m_max_edit;
      };

      struct buffer_queue_t {
        fan::opengl::core::queue_helper_t* queue_helper;
        fan::opengl::core::glsl_buffer_t* glsl_buffer;
      };
    }
  }
}

namespace fan {

  namespace opengl {
    struct render_flags {
      static constexpr uint16_t depth_test = 1;
    };
  }

  namespace opengl {

    struct context_t {

      struct properties_t {
        properties_t() {
          samples = 0;
          major = 2;
          minor = 1;
        }

        uint16_t samples;
        uint8_t major;
        uint8_t minor;
      };

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
      bll_t<fan::opengl::core::buffer_queue_t> m_write_queue;

      void init();

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
      const void* userParam )
      {
        if (type == 33361 || type == 33360) { // gl_static_draw
          return;
        }
        fan::print_no_space(type == GL_DEBUG_TYPE_ERROR ? "opengl error:" : "", type, ", severity:", severity, ", message:", message);
      }

      void set_error_callback() {
        opengl.call(opengl.glEnable, GL_DEBUG_OUTPUT);
        opengl.call(opengl.glDebugMessageCallback, (GLDEBUGPROC)message_callback, (void*)0);
      }

      uint32_t m_flags;
    };

  }
}

namespace fan {

  namespace opengl {

    namespace core {

      static int get_buffer_size(fan::opengl::context_t* context, uint32_t target_buffer, uint32_t buffer_object) {
        int size = 0;

        context->opengl.call(context->opengl.glBindBuffer, target_buffer, buffer_object);
        context->opengl.call(context->opengl.glGetBufferParameteriv, target_buffer, fan::opengl::GL_BUFFER_SIZE, &size);

        return size;
      }

      static void write_glbuffer(fan::opengl::context_t* context, unsigned int buffer, const void* data, uintptr_t size, uint32_t usage = GL_DYNAMIC_DRAW, uintptr_t target = GL_ARRAY_BUFFER)
      {
        context->opengl.call(context->opengl.glBindBuffer, target, buffer);

        context->opengl.call(context->opengl.glBufferData, target, size, data, usage);
        /*if (target == GL_SHADER_STORAGE_BUFFER) {
          glBindBufferBase(target, location, buffer);
        }*/
      }

      static void edit_glbuffer(fan::opengl::context_t* context, unsigned int buffer, const void* data, uintptr_t offset, uintptr_t size, uintptr_t target = GL_ARRAY_BUFFER)
      {
        context->opengl.call(context->opengl.glBindBuffer, target, buffer);

    #if fan_debug

        int buffer_size = get_buffer_size(context, target, buffer);

        if (buffer_size < size || (offset + size) > buffer_size) {
          throw std::runtime_error("tried to write more than allocated");
        }

    #endif

        context->opengl.call(context->opengl.glBufferSubData, target, offset, size, data);
        context->opengl.call(context->opengl.glBindBuffer, target, 0);
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

      struct vao_t {

        vao_t() = default;

        void open(fan::opengl::context_t* context) {
          context->opengl.call(context->opengl.glCreateVertexArrays, 1, &m_vao);
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

      struct glsl_buffer_t {

        glsl_buffer_t() = default;

        static constexpr uint32_t default_buffer_size = 0xfff;

        void open(fan::opengl::context_t* context) {
          m_buffer_size = 0;

          m_vao.open(context);

          context->opengl.call(context->opengl.glGenBuffers, 1, &m_vbo);
          this->allocate_buffer(context, default_buffer_size);
          m_buffer.open();
          m_buffer.reserve(default_buffer_size);
        }

        void close(fan::opengl::context_t* context) {
        #if fan_debug >= fan_debug_low
          if (m_vbo == -1) {
            fan::throw_error("tried to remove non existent vbo");
          }
        #endif
          context->opengl.call(context->opengl.glDeleteBuffers, 1, &m_vbo);

          m_vao.close(context);
          m_buffer.close();
        }

        void init(fan::opengl::context_t* context, uint32_t program, uint32_t element_byte_size) {

          m_vao.bind(context);

          this->bind(context);

          uint32_t element_count = element_byte_size / sizeof(f32_t) / 4;

          for (int i = 0; i < element_count; i++) {

            int location = context->opengl.call(context->opengl.glGetAttribLocation, program, (std::string("input") + std::to_string(i)).c_str());
            context->opengl.call(context->opengl.glEnableVertexAttribArray, location);

            context->opengl.call(context->opengl.glVertexAttribPointer,
              location, 
              4, 
              GL_FLOAT, 
              GL_FALSE, 
              element_byte_size,
              (void*)(i * sizeof(fan::vec4))
            );
          }

          if ((element_byte_size / sizeof(f32_t)) % 4 == 0) {
            return;
          }

          int location = context->opengl.call(context->opengl.glGetAttribLocation, program, (std::string("input") + std::to_string(element_count)).c_str());
          context->opengl.call(context->opengl.glEnableVertexAttribArray, location);

          context->opengl.call(context->opengl.glVertexAttribPointer,
            location, 
            (element_byte_size / sizeof(f32_t)) % 4, 
            GL_FLOAT, 
            GL_FALSE, 
            element_byte_size,
            (void*)((element_count) * sizeof(fan::vec4))
          );
        }

        void bind(fan::opengl::context_t* context) const {
          context->opengl.call(context->opengl.glBindBuffer, fan::opengl::GL_ARRAY_BUFFER, m_vbo);
        }
        void unbind() const {
          //glBindBuffer(GL_ARRAY_BUFFER, 0);
        }

        void allocate_buffer(fan::opengl::context_t* context, uint64_t size) {
          fan::opengl::core::write_glbuffer(context, m_vbo, nullptr, size);
          m_buffer_size = size;
        }
        void* get_buffer_data(GLintptr offset) const {
        #if fan_debug >= fan_debug_low
          if (offset > m_buffer.size()) {
            fan::throw_error("invalid access");
          }
        #endif
          return (void*)&m_buffer[offset];
        }

        void push_ram_instance(fan::opengl::context_t* context, const void* data, uint32_t element_byte_size) {
          m_buffer.insert(m_buffer.size(), (uint8_t*)data, (uint8_t*)data + element_byte_size);
        }

        void insert_ram_instance(fan::opengl::context_t* context, uint32_t i, const void* data, uint32_t element_byte_size) {
          m_buffer.insert(i * element_byte_size, (uint8_t*)data, (uint8_t*)data + element_byte_size);
        }

        void write_vram_all(fan::opengl::context_t* context) {
          
          m_vao.bind(context);

          this->bind(context);

          m_buffer_size = m_buffer.capacity();

          fan::opengl::core::write_glbuffer(context, m_vbo, m_buffer.begin(), m_buffer_size);
        }
        
        void* get_instance(fan::opengl::context_t* context, uint32_t i, uint32_t element_byte_size, uint32_t byte_offset) const {
          return get_buffer_data(i * element_byte_size + byte_offset);
        }
        void edit_ram_instance(fan::opengl::context_t* context, uint32_t i, const void* data, uint32_t element_byte_size, uint32_t byte_offset, uint32_t sizeof_data) {
          #if fan_debug >= fan_debug_low
            if (i * element_byte_size + byte_offset + sizeof_data > m_buffer.size()) {
              fan::throw_error("invalid access");
            }
          #endif
          std::memmove(m_buffer.begin() + i * element_byte_size + byte_offset, data, sizeof_data);
        }
        void edit_vram_instance(fan::opengl::context_t* context, uint32_t i, const void* data, uint32_t element_byte_size, uint32_t byte_offset, uint32_t sizeof_data) {
          fan::opengl::core::edit_glbuffer(context, m_vbo, data, i * element_byte_size + byte_offset, sizeof_data);
        }
        void edit_vram_buffer(fan::opengl::context_t* context, uint32_t begin, uint32_t end) {
          if (begin == end || m_buffer.empty()) {
            return;
          }
          if (end > m_buffer_size) {
            this->write_vram_all(context);
          }
          else {
            fan::opengl::core::edit_glbuffer(context, m_vbo, &m_buffer[begin], begin, end - begin);
          }
        }
        // moves element from end to x - used for optimized earsing where draw order doesnt matter
        void move_ram_buffer(fan::opengl::context_t* context, uint32_t element_byte_size, uint32_t dst, uint32_t src) {
           #if fan_debug
            if (dst * element_byte_size + element_byte_size > m_buffer.size()) {
              fan::throw_error("invalid access");
            }
            if (src * element_byte_size + element_byte_size > m_buffer.size()) {
              fan::throw_error("invalid access");
            }
          #endif
          std::memmove(&m_buffer[dst * element_byte_size], &m_buffer[src * element_byte_size], element_byte_size);
        }

        void erase_instance(fan::opengl::context_t* context, uint32_t i, uint32_t count, uint32_t element_byte_size, uint32_t vertex_count) {
          #if fan_debug >= fan_debug_low
            if (i * element_byte_size > m_buffer.size()) {
              fan::throw_error("invalid access");
            }
            if (i * element_byte_size + element_byte_size * count * vertex_count > m_buffer.size()) {
              fan::throw_error("invalid access");
            }
          #endif
          m_buffer.erase(i * element_byte_size, i * element_byte_size + element_byte_size * count * vertex_count);

          //m_buffer_size = m_buffer.capacity();
        }
        void erase(fan::opengl::context_t* context, uint32_t begin, uint32_t end) {
        #if fan_debug >= fan_debug_low
          if (begin > m_buffer.size()) {
            fan::throw_error("invalid access");
          }
          if (end > m_buffer.size()) {
            fan::throw_error("invalid access");
          }
        #endif
          m_buffer.erase(begin, end);

          //m_buffer_size = m_buffer.capacity();
        }

        void clear_ram(fan::opengl::context_t* context) {
          m_buffer.clear();
        }

        template <typename T>
        void print_ram_buffer(fan::opengl::context_t* context, uint32_t i, uint32_t element_byte_size, uint32_t byte_offset) {
          fan::print((T)m_buffer[i * element_byte_size + byte_offset]);
        }

        template <typename T>
        void print_vram_buffer(fan::opengl::context_t* context, uint32_t i, uint32_t size, uint32_t element_byte_size, uint32_t byte_offset) {
          T value;

          this->bind(context);
          glGetBufferSubData(GL_ARRAY_BUFFER, i * element_byte_size + byte_offset, size, &value);
          fan::print(value);
        }

        void confirm_buffer(fan::opengl::context_t* context) {

          if (m_buffer.empty()) {
            return;
          }

          uint8_t* ptr = (uint8_t*)get_buffer_data(0);

          for (int i = 0; i < m_buffer.size(); i++) {
            if (m_buffer[i] 
              != ptr[i]) {
              fan::throw_error("ram and vram data is different");
            }
          }
        }

        void draw(fan::opengl::context_t* context, uint32_t begin, uint32_t end) {

          m_vao.bind(context);
          
          // possibly disable depth test here
          context->opengl.call(context->opengl.glDrawArrays, GL_TRIANGLES, begin, end - begin);
        }
        
        uint32_t m_vbo;
        uint64_t m_buffer_size;

        fan::opengl::core::vao_t m_vao;
        
        fan::hector_t<uint8_t> m_buffer;

      };

    }
  }
}

inline void fan::opengl::core::queue_helper_t::open() {

  m_edit_index = fan::uninitialized;

  m_min_edit = fan::uninitialized;
  m_max_edit = 0;
}

inline void fan::opengl::core::queue_helper_t::close(fan::opengl::context_t* context) {
  if (is_queued()) {
    context->m_write_queue.erase(m_edit_index);
    reset_edit();
  }
}

inline bool fan::opengl::core::queue_helper_t::is_queued() const {
  return m_edit_index != fan::uninitialized;
}

inline void fan::opengl::core::queue_helper_t::edit(fan::opengl::context_t* context, uint32_t begin, uint32_t end, glsl_buffer_t* buffer) {

  m_min_edit = std::min(m_min_edit, begin);
  m_max_edit = std::max(m_max_edit, end);

  #if fan_debug >= fan_debug_low
  if (buffer->m_buffer.data() == nullptr) {
    fan::throw_error("invalid edit");
  }

#endif

  m_max_edit = buffer->m_buffer.size();

  if (is_queued()) {
    return;
  }

  m_edit_index = context->m_write_queue.push_back(buffer_queue_t{this, buffer});
}

inline void fan::opengl::core::queue_helper_t::on_edit(fan::opengl::context_t* context) {
  context->m_write_queue.erase(m_edit_index);

  m_min_edit = fan::uninitialized;
  m_max_edit = 0;

  m_edit_index = fan::uninitialized;
}

inline void fan::opengl::core::queue_helper_t::reset_edit() {
  m_min_edit = fan::uninitialized;
  m_max_edit = 0;

  m_edit_index = fan::uninitialized;
}

inline void fan::opengl::context_t::init() {
  
  m_draw_queue.open();
  m_write_queue.open();

  opengl.open();

  #if fan_debug >= fan_debug_high
    context_t::set_error_callback();
  #endif

  m_flags = 0;
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

    m_write_queue.start_safe_next(it);
    
    if (m_write_queue[it].glsl_buffer->m_buffer.capacity() > m_write_queue[it].glsl_buffer->m_buffer_size) {
      m_write_queue[it].glsl_buffer->write_vram_all(this);
    }
    else {
      m_write_queue[it].glsl_buffer->edit_vram_buffer(this, m_write_queue[it].queue_helper->m_min_edit, m_write_queue[it].queue_helper->m_max_edit);
    }
    m_write_queue[it].queue_helper->on_edit(this);

    it = m_write_queue.end_safe_next();
  }

  m_write_queue.clear();
  m_write_queue.open();

  it = m_draw_queue.begin();
  if (it == 0) {
    
    fan::print("problemAAAAAAAAAAAAAAAAAAAAAAAAA\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
  }
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

inline uint32_t fan::opengl::context_t::enable_draw(void * data, draw_cb_t cb)
{
  return m_draw_queue.push_back(fan::opengl::context_t::draw_queue_t{data, cb});
}

inline void fan::opengl::context_t::disable_draw(uint32_t node_reference)
{
 m_draw_queue.erase(node_reference);
}

inline void fan::opengl::context_t::set_depth_test(bool flag)
{
  switch (flag) {
  case false: {
    if (m_flags & fan::opengl::render_flags::depth_test) {
      opengl.call(opengl.glDisable, fan::opengl::GL_DEPTH_TEST);
      m_flags &= ~fan::opengl::render_flags::depth_test;
    }
    break;
  }
  default: {
    if (!(m_flags & fan::opengl::render_flags::depth_test)) {
      opengl.call(opengl.glEnable, fan::opengl::GL_DEPTH_TEST);
      m_flags |= fan::opengl::render_flags::depth_test;
    }
  }
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
