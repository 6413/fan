#pragma once

#include <fan/graphics/common_context.h>

#include <fan/graphics/opengl/init.h>
#include <fan/graphics/image_load.h>
#include <fan/graphics/camera.h>

template<typename T>
concept not_non_arithmethic_types = !std::is_same_v<T, fan::vec2> &&
!std::is_same_v<T, fan::vec3> &&
!std::is_same_v<T, fan::vec4> &&
!std::is_same_v<T, fan::color>;

namespace fan {
  struct window_t;
}

namespace fan {
  namespace opengl {
    inline std::unordered_map<size_t, int> shader_location_cache;
    struct context_t {

      void print_version();

      struct properties_t {

      };
  
      fan::opengl::opengl_t opengl;
      GLuint current_program = -1;

      static void error_callback(int error, const char* description) {
        if (error == GLFW_NOT_INITIALIZED) {
          return;
        }
        fan::print("window error " + std::to_string(error) + ": " + description);
        __abort();
      }

      void open(const properties_t& p = properties_t());
      void close();
      void internal_close();

      void render(fan::window_t& window);

      void set_depth_test(bool flag);
      void set_blending(bool flag);

      void set_stencil_test(bool flag);
      void set_stencil_op(GLenum sfail, GLenum dpfail, GLenum dppass);

      void set_vsync(fan::window_t* window, bool flag);

      static void message_callback(GLenum source,
        GLenum type,
        GLuint id,
        GLenum severity,
        GLsizei length,
        const GLchar* message,
        const void* userParam);

      void set_error_callback();

      void set_current(fan::window_t* window);

      //-----------------------------shader-----------------------------

      struct view_projection_t {
        fan::mat4 projection;
        fan::mat4 view;
      };

      struct shader_t {
        GLuint id = -1;
        int projection_view[2]{ -1, -1 };
        uint32_t vertex = -1, fragment = -1;
      };

      static constexpr auto shader_validate_error_message = [](const auto str) {
        return "failed to set value for:" + str + " check if variable is used in file so that its not optimized away";
      };

      static constexpr auto validate_error_message = [](const auto str) {
        return "failed to set value for:" + str + " check if variable is used in file so that its not optimized away";
        };

      template <typename T>
      void shader_set_value(fan::graphics::shader_nr_t nr, const fan::string& name, const T& val);
      void shader_set_camera(fan::graphics::shader_nr_t nr, fan::graphics::camera_nr_t camera_nr);

     //-----------------------------shader-----------------------------

      //-----------------------------image-----------------------------

      struct image_t {
        #include <fan/graphics/image_common.h>
        GLuint texture_id;
      };

      struct image_format {
        static constexpr auto b8g8r8a8_unorm = GL_BGRA;
        static constexpr auto r8_unorm = GL_RED;
        static constexpr auto rg8_unorm = GL_RG;
      };

      struct image_sampler_address_mode {
        static constexpr auto repeat = GL_REPEAT;
        static constexpr auto mirrored_repeat = GL_MIRRORED_REPEAT;
        static constexpr auto clamp_to_edge = GL_CLAMP_TO_EDGE;
        static constexpr auto clamp_to_border = GL_CLAMP_TO_BORDER;
        static constexpr auto mirrored_clamp_to_edge = GL_MIRROR_CLAMP_TO_EDGE;
      };

      struct image_filter {
        static constexpr auto nearest = GL_NEAREST;
        static constexpr auto linear = GL_LINEAR;
      };

      struct image_load_properties_defaults {
        static constexpr uint32_t visual_output = GL_REPEAT;
        static constexpr uint32_t internal_format = GL_RGBA;
        static constexpr uint32_t format = GL_RGBA;
        static constexpr uint32_t type = GL_UNSIGNED_BYTE;
        static constexpr uint32_t min_filter = GL_NEAREST;
        static constexpr uint32_t mag_filter = GL_NEAREST;
      };

      struct image_load_properties_t {
        uint32_t            visual_output = image_load_properties_defaults::visual_output;
        uintptr_t           internal_format = image_load_properties_defaults::internal_format;
        uintptr_t           format = image_load_properties_defaults::format;
        uintptr_t           type = image_load_properties_defaults::type;
        uintptr_t           min_filter = image_load_properties_defaults::min_filter;
        uintptr_t           mag_filter = image_load_properties_defaults::mag_filter;
      };

      static constexpr fan::vec4_wrap_t<fan::vec2> default_texture_coordinates = fan::vec4_wrap_t<fan::vec2>(
        fan::vec2(0, 0),
        fan::vec2(1, 0),
        fan::vec2(1, 1),
        fan::vec2(0, 1)
      );
      //-----------------------------image-----------------------------

      //-----------------------------viewport-----------------------------

      //-----------------------------viewport-----------------------------

    };
  }
}

namespace fan {
  namespace opengl {
    namespace core {

      int get_buffer_size(fan::opengl::context_t& context, GLenum target_buffer, GLuint buffer_object);

      void write_glbuffer(fan::opengl::context_t& context, GLuint buffer, const void* data, uintptr_t size, uint32_t usage, GLenum target);
      void get_glbuffer(fan::opengl::context_t& context, void* data, GLuint buffer_id, uintptr_t size, uintptr_t offset, GLenum target);

      void edit_glbuffer(fan::opengl::context_t& context, GLuint buffer, const void* data, uintptr_t offset, uintptr_t size, uintptr_t target);

      // not tested
      int get_bound_buffer(fan::opengl::context_t& context);
      //#pragma pack(push, 1)
      //#pragma pack(pop)
      struct vao_t {

        void open(fan::opengl::context_t& context);
        void close(fan::opengl::context_t& context);

        void bind(fan::opengl::context_t& context) const;
        void unbind(fan::opengl::context_t& context) const;

        GLuint m_buffer = (GLuint)-1;
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

        GLuint m_buffer = (GLuint)-1;
        GLenum m_target = (GLuint)-1;
        uint32_t m_usage = GL_DYNAMIC_DRAW;
      };

      struct framebuffer_t {

        struct properties_t {
          properties_t() {}
          GLenum internalformat = GL_DEPTH_STENCIL_ATTACHMENT;
        };

        void open(fan::opengl::context_t& context);
        void close(fan::opengl::context_t& context);

        void bind(fan::opengl::context_t& context) const;
        void unbind(fan::opengl::context_t& context) const;

        bool ready(fan::opengl::context_t& context) const;

        void bind_to_renderbuffer(fan::opengl::context_t& context, GLenum renderbuffer, const properties_t& p = properties_t());

        // texture must be binded with texture.bind();
        static void bind_to_texture(fan::opengl::context_t& context, GLuint texture, GLenum attatchment);

        GLuint framebuffer;
      };

      struct renderbuffer_t {

        struct properties_t {
          properties_t() {}
          GLenum internalformat = GL_DEPTH24_STENCIL8;
          fan::vec2ui size;
        };

        void open(fan::opengl::context_t& context);
        void close(fan::opengl::context_t& context);
        void bind(fan::opengl::context_t& context) const;
        void set_storage(fan::opengl::context_t& context, const properties_t& p) const;
        void bind_to_renderbuffer(fan::opengl::context_t& context, const properties_t& p = properties_t());

        GLuint renderbuffer;
      };


    }
  }
}