#pragma once

#include <fan/graphics/renderer.hpp>

#if fan_renderer == fan_renderer_opengl

#include <fan/types/types.hpp>
#include <fan/types/color.hpp>
#include <fan/graphics/camera.hpp>
#include <fan/graphics/opengl/gl_shader.hpp>
#include <fan/window/window.hpp>

//#define fan_debug

namespace fan {

  namespace opengl {

    struct context_t;

  }

  namespace graphics{

    namespace core {

      struct glsl_buffer_t;

      struct queue_helper_t {

			  void open();

			  void close(fan::opengl::context_t* context);

        bool is_queued() const;

			  void edit(fan::opengl::context_t* context, uint32_t begin, uint32_t end, glsl_buffer_t* buffer);

			  void on_edit();

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

  namespace opengl {

		struct context_t {

    #ifdef fan_debug

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

    #ifdef fan_debug

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

        static constexpr uint32_t default_buffer_size = 0;
        static constexpr f32_t buffer_increment = 1;

        void open() {
          #ifdef fan_debug
          if (m_vbo != -1) {
            fan::throw_error("tried to allocate already allocated vbo - remove() needs to be called before create");
          }
        #endif
          m_element_size = 0;
          m_buffer_size = 0;

          m_vao.open();

          glGenBuffers(1, &m_vbo);
          this->allocate_buffer(default_buffer_size);
          m_buffer.reserve(default_buffer_size);
        }

        void close() {
        #ifdef fan_debug
          if (m_vbo == -1) {
            fan::throw_error("tried to remove non existent vbo");
          }
        #endif
          glDeleteBuffers(1, &m_vbo);

          m_vao.close();
        }

        void init(uint32_t program, uint32_t size) {

          m_element_size = size;

          m_vao.bind();

          this->bind();

          uint32_t element_count = size / sizeof(f32_t) / 4;

          for (int i = 0; i < element_count; i++) {

            int location = glGetAttribLocation(program, (std::string("input") + std::to_string(i)).c_str());
            glEnableVertexAttribArray(location);

            glVertexAttribPointer(
              location, 
              4, 
              GL_FLOAT, 
              GL_FALSE, 
              size,
              (void*)(i * sizeof(fan::vec4))
            );
          }

          if ((size / sizeof(f32_t)) % 4 == 0) {
            return;
          }

          int location = glGetAttribLocation(program, (std::string("input") + std::to_string(element_count)).c_str());
          glEnableVertexAttribArray(location);

          glVertexAttribPointer(
            location, 
            (size / sizeof(f32_t)) % 4, 
            GL_FLOAT, 
            GL_FALSE, 
            size,
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
          return (void*)&m_buffer[offset];
        }

        void push_ram_instance(const void* data) {
          m_buffer.insert(m_buffer.end(), (uint8_t*)data, (uint8_t*)data + m_element_size);
        }

        void insert_ram_instance(uint32_t i, const void* data) {
          m_buffer.insert(m_buffer.begin() + i * m_element_size, (uint8_t*)data, (uint8_t*)data + m_element_size);
        }

        void write_vram_all() {
          
          if (m_buffer.size() > m_buffer_size) {

            m_vao.bind();

            this->bind();

            m_buffer_size = m_buffer.capacity();

            fan::graphics::core::write_glbuffer(m_vbo, m_buffer.data(), m_buffer_size);
          }
          else {
            fan::graphics::core::edit_glbuffer(m_vbo, &m_buffer[0], 0,  m_buffer.size());
          }
        }
        
        void* get_instance(uint32_t i, uint32_t byte_offset) const {
          return get_buffer_data(i * m_element_size + byte_offset);
        }
        void edit_ram_instance(uint32_t i, const void* data, uint32_t byte_offset, uint32_t sizeof_data) {
          std::memcpy(&m_buffer[i * m_element_size + byte_offset], data, sizeof_data);
        }
        void edit_vram_instance(uint32_t i, const void* data, uint32_t byte_offset, uint32_t sizeof_data) {
          fan::graphics::core::edit_glbuffer(m_vbo, data, i * m_element_size + byte_offset, sizeof_data);
        }
        void edit_vram_buffer(uint32_t begin, uint32_t end) {
          fan::graphics::core::edit_glbuffer(m_vbo, &m_buffer[begin], begin, end - begin);
        }
        // moves element from end to x - used for optimized earsing where draw order doesnt matter
        void erase_move_ram_buffer(uint32_t dst, uint32_t src) {
          std::memcpy(&m_buffer[dst * m_element_size], &m_buffer[src * m_element_size], m_element_size);
        }

        void erase_instance(uint32_t i, uint32_t count = 1) {

          m_buffer.erase(m_buffer.begin() + i * m_element_size, m_buffer.begin() + (i * count) * m_element_size);

          m_buffer_size = m_buffer.size();

          fan::graphics::core::write_glbuffer(m_vbo, m_buffer.data(), m_buffer_size);
        }

        void clear() {
          // do we use default init size
          fan::graphics::core::write_glbuffer(m_vbo, nullptr, 0);
        }

        void print_buffer() {
          void* buffer = malloc(m_buffer.size());

          this->bind();

          glGetBufferSubData(GL_ARRAY_BUFFER, 0, m_buffer.size() - m_element_size, buffer);

          for (int i = 0; i < m_buffer.size() / 4; i++) {
            fan::print(((f32_t*)buffer)[i]);
          }

          free(buffer);
        }

        void draw(fan::opengl::context_t* context, fan::shader_t shader) {
          m_vao.bind();
          const fan::vec2 viewport_size = context->viewport_size;

          fan::mat4 projection(1);
		      projection = fan::math::ortho<fan::mat4>(
            (f32_t)viewport_size.x * 0.5,
            ((f32_t)viewport_size.x + (f32_t)viewport_size.x * 0.5), 
            ((f32_t)viewport_size.y + (f32_t)viewport_size.y * 0.5), 
            ((f32_t)viewport_size.y * 0.5), 
            -1, 
            1000.0f
          );

		      fan::mat4 view(1);
		      view = context->camera.get_view_matrix(view.translate(fan::vec3((f_t)viewport_size.x * 0.5, (f_t)viewport_size.y * 0.5, -700.0f)));

		      shader->use();
		      shader->set_projection(projection);
		      shader->set_view(view);

          // possibly disable depth test here
		      glDrawArrays(GL_TRIANGLES, 0, m_buffer.size() / sizeof(f32_t));
        }

        uint32_t m_vbo
        #ifdef fan_debug
          = -1;
        #endif
          ;
        uint32_t m_element_size;
        uint64_t m_buffer_size;

        fan::graphics::core::vao_t m_vao;
        
        std::vector<uint8_t> m_buffer;

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

inline void fan::graphics::core::queue_helper_t::on_edit() {
	m_min_edit = fan::uninitialized;
	m_max_edit = 0;

	m_edit_index = fan::uninitialized;
}

inline void fan::graphics::core::queue_helper_t::reset_edit() {
	m_min_edit = -1;
	m_max_edit = 0;

	m_edit_index = fan::uninitialized;
}

inline void fan::opengl::context_t::init() {

  #ifdef fan_debug
    m_init[0] = 1;
  #endif

	if (glewInit() != GLEW_OK) {
		fan::throw_error("failed to initialize glew");
	}
}

inline void fan::opengl::context_t::bind_to_window(fan::window* window) {

  #ifdef fan_debug
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

}

inline void fan::opengl::context_t::set_viewport(const fan::vec2& viewport_position, const fan::vec2& viewport_size_) {
  #ifdef fan_debug
    m_init[2] = 1;
  #endif
	glViewport(viewport_position.x, viewport_position.y, viewport_size_.x, viewport_size_.y);
	viewport_size = viewport_size_;
}

inline void fan::opengl::context_t::process() {
	#if fan_renderer == fan_renderer_opengl

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	#endif

  #ifdef fan_debug
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
    m_write_queue[it].queue_helper->on_edit();

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

#endif