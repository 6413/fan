#pragma once

#include <fan/types/types.h>

#include <fan/graphics/renderer.h>

#if fan_renderer == fan_renderer_opengl

#include <fan/types/types.h>
#include <fan/types/color.h>
#include <fan/graphics/camera.h>
#include <fan/graphics/opengl/gl_shader.h>
#include <fan/window/window.h>
#include <fan/types/memory.h>

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

    #if fan_debug >= fan_debug_medium

      bool m_init[3]{};

    #endif

			fan::camera camera;
			fan::vec2 viewport_size;

      typedef void(*draw_cb_t)(context_t*, void*);

      struct draw_queue_t {
        void* data;
        draw_cb_t draw_cb;
      };

			bll_t<draw_queue_t> m_draw_queue;
			bll_t<fan::opengl::core::buffer_queue_t> m_write_queue;

			void init();

			void bind_to_window(fan::window_t* window);

			void set_viewport(const fan::vec2& viewport_position, const fan::vec2& viewport_size_);

			void process();

			void render(fan::window_t* window);

      uint32_t enable_draw(void* data, draw_cb_t);
      void disable_draw(uint32_t node_reference);

      void set_depth_test(bool flag);

      static void* get_proc_address(const char* name) {
        #if defined(fan_platform_windows)
         void *p = (void *)wglGetProcAddress(name);
        if(p == 0 ||
          (p == (void*)0x1) || (p == (void*)0x2) || (p == (void*)0x3) ||
          (p == (void*)-1) )
        {
          HMODULE module = LoadLibraryA("opengl32.dll");
          p = (void *)GetProcAddress(module, name);
        }

        return p;

        #elif defined(fan_platform_unix)

        return (void*)glXGetProcAddress(name);

        #endif
      }

      static void message_callback(GLenum source,
	    GLenum type,
	    GLuint id,
	    GLenum severity,
	    GLsizei length,
	    const GLchar* message,
	    const void* userParam )
      {
	      if (type == 33361) { // gl_static_draw
		      return;
	      }
	      fan::print_no_space(type == GL_DEBUG_TYPE_ERROR ? "opengl error:" : "", type, ", severity:", severity, ", message:", message);
      }

      static void set_error_callback() {
        glEnable(GL_DEBUG_OUTPUT);
	      glDebugMessageCallback((GLDEBUGPROC)message_callback, 0);
      }

      uint32_t m_flags;
		};

  }
}

namespace fan {

  namespace opengl {

    namespace core {

      static int get_buffer_size(uint32_t target_buffer, uint32_t buffer_object) {
		    int size = 0;

		    glBindBuffer(target_buffer, buffer_object);
		    glGetBufferParameteriv(target_buffer, GL_BUFFER_SIZE, &size);

		    return size;
	    }

	    static void write_glbuffer(unsigned int buffer, const void* data, uintptr_t size, uint32_t usage = GL_STATIC_DRAW, uintptr_t target = GL_ARRAY_BUFFER)
	    {
		    glBindBuffer(target, buffer);

		    glBufferData(target, size, data, usage);
		    /*if (target == GL_SHADER_STORAGE_BUFFER) {
			    glBindBufferBase(target, location, buffer);
		    }*/
	    }

	    static void edit_glbuffer(unsigned int buffer, const void* data, uintptr_t offset, uintptr_t size, uintptr_t target = GL_ARRAY_BUFFER)
	    {
		    glBindBuffer(target, buffer);

    #if fan_debug

		    int buffer_size = get_buffer_size(target, buffer);

		    if (buffer_size < size || (offset + size) > buffer_size) {
			    throw std::runtime_error("tried to write more than allocated");
		    }

    #endif

		    glBufferSubData(target, offset, size, data);
		    glBindBuffer(target, 0);
		   /* if (target == GL_SHADER_STORAGE_BUFFER) {
			    glBindBufferBase(target, location, buffer);
		    }*/
	    }

      // not tested
      static int get_bound_buffer() {
        int buffer_id;
        glGetIntegerv(GL_VERTEX_BINDING_BUFFER, &buffer_id);
        return buffer_id;
      }

      struct vao_t {

        vao_t() = default;

        void open() {
          glCreateVertexArrays(1, &m_vao);
        }

        void close() {
          glDeleteVertexArrays(1, &m_vao);
        }

        void bind() const {
          glBindVertexArray(m_vao);
        }
        void unbind() const {
          glBindVertexArray(0);
        }

        uint32_t m_vao;

      };

      struct glsl_buffer_t {

        glsl_buffer_t() = default;

        static constexpr uint32_t default_buffer_size = 0xfff;

        void open() {
          m_buffer_size = 0;

          m_vao.open();

          glGenBuffers(1, &m_vbo);
          this->allocate_buffer(default_buffer_size);
          m_buffer.open();
          m_buffer.reserve(default_buffer_size);
        }

        void close() {
        #if fan_debug >= fan_debug_low
          if (m_vbo == -1) {
            fan::throw_error("tried to remove non existent vbo");
          }
        #endif
          glDeleteBuffers(1, &m_vbo);

          m_vao.close();
          m_buffer.close();
        }

        void init(uint32_t program, uint32_t element_byte_size) {

          m_vao.bind();

          this->bind();

          uint32_t element_count = element_byte_size / sizeof(f32_t) / 4;

          for (int i = 0; i < element_count; i++) {

            int location = glGetAttribLocation(program, (std::string("input") + std::to_string(i)).c_str());
            glEnableVertexAttribArray(location);

            glVertexAttribPointer(
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

          int location = glGetAttribLocation(program, (std::string("input") + std::to_string(element_count)).c_str());
          glEnableVertexAttribArray(location);

          glVertexAttribPointer(
            location, 
            (element_byte_size / sizeof(f32_t)) % 4, 
            GL_FLOAT, 
            GL_FALSE, 
            element_byte_size,
            (void*)((element_count) * sizeof(fan::vec4))
          );
        }

        void bind() const {
          glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
        }
        void unbind() const {
          //glBindBuffer(GL_ARRAY_BUFFER, 0);
        }

        void allocate_buffer(uint64_t size) {
          fan::opengl::core::write_glbuffer(m_vbo, nullptr, size);
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

        void push_ram_instance(const void* data, uint32_t element_byte_size) {
          m_buffer.insert(m_buffer.size(), (uint8_t*)data, (uint8_t*)data + element_byte_size);
        }

        void insert_ram_instance(uint32_t i, const void* data, uint32_t element_byte_size) {
          m_buffer.insert(i * element_byte_size, (uint8_t*)data, (uint8_t*)data + element_byte_size);
        }

        void write_vram_all() {
          
          m_vao.bind();

          this->bind();

          m_buffer_size = m_buffer.capacity();

          fan::opengl::core::write_glbuffer(m_vbo, m_buffer.begin(), m_buffer_size);
        }
        
        void* get_instance(uint32_t i, uint32_t element_byte_size, uint32_t byte_offset) const {
          return get_buffer_data(i * element_byte_size + byte_offset);
        }
        void edit_ram_instance(uint32_t i, const void* data, uint32_t element_byte_size, uint32_t byte_offset, uint32_t sizeof_data) {
          #if fan_debug >= fan_debug_low
            if (i * element_byte_size + byte_offset + sizeof_data > m_buffer.size()) {
              fan::throw_error("invalid access");
            }
          #endif
          std::memmove(m_buffer.begin() + i * element_byte_size + byte_offset, data, sizeof_data);
        }
        void edit_vram_instance(uint32_t i, const void* data, uint32_t element_byte_size, uint32_t byte_offset, uint32_t sizeof_data) {
          fan::opengl::core::edit_glbuffer(m_vbo, data, i * element_byte_size + byte_offset, sizeof_data);
        }
        void edit_vram_buffer(uint32_t begin, uint32_t end) {
          if (begin == end) {
            return;
          }
          if (end > m_buffer_size) {
            this->write_vram_all();
          }
          else {
            fan::opengl::core::edit_glbuffer(m_vbo, &m_buffer[begin], begin, end - begin);
          }
        }
        // moves element from end to x - used for optimized earsing where draw order doesnt matter
        void move_ram_buffer(uint32_t element_byte_size, uint32_t dst, uint32_t src) {
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

        void erase_instance(uint32_t i, uint32_t count, uint32_t element_byte_size, uint32_t vertex_count) {
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
        void erase(uint32_t begin, uint32_t end) {
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

        void clear_ram() {
          m_buffer.clear();
        }

        template <typename T>
        void print_ram_buffer(uint32_t i, uint32_t element_byte_size, uint32_t byte_offset) {
          fan::print((T)m_buffer[i * element_byte_size + byte_offset]);
        }

        template <typename T>
        void print_vram_buffer(uint32_t i, uint32_t size, uint32_t element_byte_size, uint32_t byte_offset) {
          T value;

          this->bind();
          glGetBufferSubData(GL_ARRAY_BUFFER, i * element_byte_size + byte_offset, size, &value);
          fan::print(value);
        }

        void confirm_buffer() {

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

        void draw(fan::opengl::context_t* context, fan::shader_t shader, uint32_t begin, uint32_t end) {
          m_vao.bind();
          const fan::vec2 viewport_size = context->viewport_size;

          fan::mat4 projection(1);
		      projection = fan::math::ortho<fan::mat4>(
            (f32_t)viewport_size.x * 0.5,
            ((f32_t)viewport_size.x + (f32_t)viewport_size.x * 0.5), 
            ((f32_t)viewport_size.y + (f32_t)viewport_size.y * 0.5), 
            ((f32_t)viewport_size.y * 0.5), 
            0.01,
            1000.0
          );

		      fan::mat4 view(1);
		      view = context->camera.get_view_matrix(view.translate(fan::vec3((f_t)viewport_size.x * 0.5, (f_t)viewport_size.y * 0.5, -700.0f)));

		      shader.use();
		      shader.set_projection(projection);
		      shader.set_view(view);

          // possibly disable depth test here
		      glDrawArrays(GL_TRIANGLES, begin, end - begin);
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

	if (is_queued()) {
    return;
  }

#if fan_debug >= fan_debug_low
  if (buffer->m_buffer.size() < begin || buffer->m_buffer.size() < end) {
    fan::throw_error("invalid edit");
  }
#endif

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

  #if fan_debug >= fan_debug_medium
    std::memset(m_init, 0, sizeof(m_init));
    m_init[0] = 1;
  #endif

  auto f = &get_proc_address;

  auto x = gladLoadGL((GLADloadfunc)f);

	if (!x) {
		fan::throw_error("failed to initialize glad");
	}

  glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  #if fan_debug >= fan_debug_high
    context_t::set_error_callback();
  #endif

  m_flags = 0;
}

inline void fan::opengl::context_t::bind_to_window(fan::window_t* window) {

  #if fan_debug >= fan_debug_medium
    m_init[1] = 1;
  #endif

	#if defined(fan_platform_windows)

	if (wglGetCurrentContext() != window->m_context) {
		wglMakeCurrent(window->m_hdc, window->m_context);
	}

	#elif defined(fan_platform_unix)

	if (glXGetCurrentContext() != window->m_context) {
		glXMakeCurrent(fan::sys::m_display, window->m_window, window->m_context);
	}

	#endif

  window->set_vsync(true);
}

inline void fan::opengl::context_t::set_viewport(const fan::vec2& viewport_position, const fan::vec2& viewport_size_) {
  #if fan_debug >= fan_debug_medium
    m_init[2] = 1;
  #endif
	glViewport(viewport_position.x, viewport_position.y, viewport_size_.x, viewport_size_.y);
	viewport_size = viewport_size_;
}

inline void fan::opengl::context_t::process() {
	#if fan_renderer == fan_renderer_opengl

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	#endif

  #if fan_debug >= fan_debug_medium
    for (int j = 0; j < 3; j++) {
      if (!m_init[j]) {
        switch(j) {
          case 0: {
            fan::throw_error("fan::opengl::context_t not initialized with fan::opengl::context_t::init()");
          }
          case 1: {
            fan::print_warning("possible missing bind for context_t - for binding context to window use fan::opengl::context_t::bind_to_window()");
          }
          case 2: {
            fan::print_warning("possibly not set view port - to set viewport for context use fan::opengl::context_t::set_viewport()");
          }
        }
      }
    }
  #endif

	uint32_t it = m_write_queue.begin();

	while (it != m_write_queue.end()) {

		m_write_queue.start_safe_next(it);
    
    if (m_write_queue[it].glsl_buffer->m_buffer.capacity() > m_write_queue[it].glsl_buffer->m_buffer_size) {
      m_write_queue[it].glsl_buffer->write_vram_all();
    }
    else {
      m_write_queue[it].glsl_buffer->edit_vram_buffer(m_write_queue[it].queue_helper->m_min_edit, m_write_queue[it].queue_helper->m_max_edit);
    }
    m_write_queue[it].queue_helper->on_edit(this);

		it = m_write_queue.end_safe_next();
	}

	m_write_queue.clear();
	m_write_queue.open();

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
		glXSwapBuffers(fan::sys::m_display, window->m_window);
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
      glDisable(GL_DEPTH_TEST);
      m_flags &= ~fan::opengl::render_flags::depth_test;
    }
    break;
  }
  default: {
    if (!(m_flags & fan::opengl::render_flags::depth_test)) {
      glEnable(GL_DEPTH_TEST);
      m_flags |= fan::opengl::render_flags::depth_test;
    }
  }
  }

  
}

#endif