#pragma once

#include <fan/types/types.hpp>

#include <fan/graphics/renderer.hpp>

#if fan_renderer == fan_renderer_opengl

#include <fan/types/types.hpp>
#include <fan/types/color.hpp>
#include <fan/graphics/camera.hpp>
#include <fan/graphics/opengl/gl_shader.hpp>
#include <fan/window/window.hpp>
#include <fan/types/memory.h>

namespace fan {
  #ifdef fan_platform_windows
    #include <DbgHelp.h>
  #endif

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

  namespace graphics{

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
        fan::graphics::core::queue_helper_t* queue_helper;
        fan::graphics::core::glsl_buffer_t* glsl_buffer;
      };
    }
  }
}

namespace fan {

  namespace graphics {
    struct render_flags {
      static constexpr uint16_t depth_test = 1;
    };
  }

  namespace opengl {

		struct context_t {

    #if fan_debug >= fan_debug_soft

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
			bll_t<fan::graphics::core::buffer_queue_t> m_write_queue;

			void init();

			void bind_to_window(fan::window* window);

			void set_viewport(const fan::vec2& viewport_position, const fan::vec2& viewport_size_);

			void process();

			void render(fan::window* window);

      uint32_t enable_draw(void* data, draw_cb_t);
      void disable_draw(uint32_t node_reference);

      void set_depth_test(bool flag);

      uint32_t m_flags;
		};

  }
}

namespace fan {

  namespace graphics {

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
        #if fan_debug >= fan_debug_soft
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
          fan::graphics::core::write_glbuffer(m_vbo, nullptr, size);
          m_buffer_size = size;
        }
        void* get_buffer_data(GLintptr offset) const {
        #if fan_debug >= fan_debug_soft
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

          fan::graphics::core::write_glbuffer(m_vbo, m_buffer.begin(), m_buffer_size);
        }
        
        void* get_instance(uint32_t i, uint32_t element_byte_size, uint32_t byte_offset) const {
          return get_buffer_data(i * element_byte_size + byte_offset);
        }
        void edit_ram_instance(uint32_t i, const void* data, uint32_t element_byte_size, uint32_t byte_offset, uint32_t sizeof_data) {
          #if fan_debug >= fan_debug_soft
            if (i * element_byte_size + byte_offset + sizeof_data > m_buffer.size()) {
              fan::throw_error("invalid access");
            }
          #endif
          std::memmove(m_buffer.begin() + i * element_byte_size + byte_offset, data, sizeof_data);
        }
        void edit_vram_instance(uint32_t i, const void* data, uint32_t element_byte_size, uint32_t byte_offset, uint32_t sizeof_data) {
          fan::graphics::core::edit_glbuffer(m_vbo, data, i * element_byte_size + byte_offset, sizeof_data);
        }
        void edit_vram_buffer(uint32_t begin, uint32_t end) {
          if (begin == end) {
            return;
          }
          if (end > m_buffer_size) {
            this->write_vram_all();
          }
          else {
            fan::graphics::core::edit_glbuffer(m_vbo, &m_buffer[begin], begin, end - begin);
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
          #if fan_debug >= fan_debug_soft
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
        #if fan_debug >= fan_debug_soft
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

        fan::graphics::core::vao_t m_vao;
        
        fan::hector_t<uint8_t> m_buffer;

      };

    }
  }
}

inline void fan::graphics::core::queue_helper_t::open() {

	m_edit_index = fan::uninitialized;

	m_min_edit = fan::uninitialized;
	m_max_edit = 0;
}

inline void fan::graphics::core::queue_helper_t::close(fan::opengl::context_t* context) {
	if (is_queued()) {
		context->m_write_queue.erase(m_edit_index);
		reset_edit();
	}
}

inline bool fan::graphics::core::queue_helper_t::is_queued() const {
  return m_edit_index != fan::uninitialized;
}

inline void fan::graphics::core::queue_helper_t::edit(fan::opengl::context_t* context, uint32_t begin, uint32_t end, glsl_buffer_t* buffer) {

	m_min_edit = std::min(m_min_edit, begin);
	m_max_edit = std::max(m_max_edit, end);

	if (is_queued()) {
    return;
  }

  m_edit_index = context->m_write_queue.push_back(buffer_queue_t{this, buffer});
}

inline void fan::graphics::core::queue_helper_t::on_edit(fan::opengl::context_t* context) {
  context->m_write_queue.erase(m_edit_index);

	m_min_edit = fan::uninitialized;
	m_max_edit = 0;

	m_edit_index = fan::uninitialized;
}

inline void fan::graphics::core::queue_helper_t::reset_edit() {
	m_min_edit = fan::uninitialized;
	m_max_edit = 0;

	m_edit_index = fan::uninitialized;
}

inline void fan::opengl::context_t::init() {

  #if fan_debug >= fan_debug_soft
    std::memset(m_init, 0, sizeof(m_init));
    m_init[0] = 1;
  #endif

	if (glewInit() != GLEW_OK) {
		fan::throw_error("failed to initialize glew");
	}

  glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  m_flags = 0;
}

inline void fan::opengl::context_t::bind_to_window(fan::window* window) {

  #if fan_debug >= fan_debug_soft
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
  #if fan_debug >= fan_debug_soft
    m_init[2] = 1;
  #endif
	glViewport(viewport_position.x, viewport_position.y, viewport_size_.x, viewport_size_.y);
	viewport_size = viewport_size_;
}

inline void fan::opengl::context_t::process() {
	#if fan_renderer == fan_renderer_opengl

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	#endif

  #if fan_debug >= fan_debug_soft
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

inline void fan::opengl::context_t::render(fan::window* window) {
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
    if (m_flags & fan::graphics::render_flags::depth_test) {
      glDisable(GL_DEPTH_TEST);
      m_flags &= ~fan::graphics::render_flags::depth_test;
    }
    break;
  }
  default: {
    if (!(m_flags & fan::graphics::render_flags::depth_test)) {
      glEnable(GL_DEPTH_TEST);
      m_flags |= fan::graphics::render_flags::depth_test;
    }
  }
  }

  
}

#endif