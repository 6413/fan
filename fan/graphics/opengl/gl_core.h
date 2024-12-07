#pragma once

#include <fan/graphics/opengl/gl_init.h>
#include <fan/graphics/image_load.h>
#include <fan/graphics/camera.h>

template<typename T>
concept not_non_arithmethic_types = !std::is_same_v<T, fan::vec2> &&
!std::is_same_v<T, fan::vec3> &&
!std::is_same_v<T, fan::vec4> &&
!std::is_same_v<T, fan::color>;

namespace fan {
  namespace opengl {
    struct context_t {

      void print_version();

      struct properties_t {

      };
  
      fan::opengl::opengl_t opengl;
      fan::opengl::GLuint current_program = -1;

      static void error_callback(int error, const char* description) {
        if (GLFW_NOT_INITIALIZED == error) {
          return;
        }
        fan::print("window error:", description);
        __abort();
      }

      void open(const properties_t& p = properties_t());

      void render(fan::window_t& window) {
        glfwSwapBuffers(window.glfw_window);
      }

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



      static fan::string read_shader(const fan::string& path) {
        fan::string code;
        fan::io::file::read(path, &code);
        return code;
      }

      //-----------------------------shader-----------------------------

      struct camera_t : fan::camera {
        fan::mat4 m_projection = fan::mat4(1);
        fan::mat4 m_view = fan::mat4(1);
        f32_t zfar = 1000.f;
        f32_t znear = 0.1f;

        union {
          struct {
            f32_t left;
            f32_t right;
            f32_t up;
            f32_t down;
          };
          fan::vec4 v;
        }coordinates;
      };

    protected:

      #include <fan/graphics/opengl/camera_list_builder_settings.h>
      #include <BLL/BLL.h>
    public:
      using camera_nr_t = camera_list_NodeReference_t;
      struct shader_t {
        fan::opengl::GLuint id = -1;
        int projection_view[2]{ -1, -1 };
        uint32_t vertex = -1, fragment = -1;
        // can be risky without constructor copy
        std::string svertex, sfragment;

        std::unordered_map<std::string, std::string> uniform_type_table;
      };
    protected:
      #include <fan/graphics/opengl/shader_list_builder_settings.h>
      #include <BLL/BLL.h>
    public:
      shader_list_t shader_list;

      using shader_nr_t = shader_list_NodeReference_t;

      static constexpr auto shader_validate_error_message = [](const auto str) {
        return "failed to set value for:" + str + " check if variable is used in file so that its not optimized away";
      };

      shader_nr_t shader_create();
      shader_t& shader_get(shader_nr_t nr);
      void shader_erase(shader_nr_t nr);

      void shader_use(shader_nr_t nr);

      void shader_set_vertex(shader_nr_t nr, const fan::string& vertex_code);
      void shader_set_fragment(shader_nr_t nr, const fan::string& fragment_code);
      bool shader_compile(shader_nr_t nr);
      bool shader_check_compile_errors(fan::opengl::GLuint nr, const fan::string& type);
      bool shader_check_compile_errors(fan::opengl::context_t::shader_t&, const fan::string& type);

      void shader_set_camera(shader_nr_t nr, camera_nr_t camera_nr);

      static constexpr auto validate_error_message = [](const auto str) {
        return "failed to set value for:" + str + " check if variable is used in file so that its not optimized away";
        };

      inline static std::unordered_map<size_t, int> shader_location_cache;
      template <typename T>
      void shader_set_value(shader_nr_t nr, const fan::string& name, T val) {
        static_assert(std::is_same_v<T, bool> == false || !std::is_same_v<T, int>, "only 4 byte supported");
        static_assert(std::is_same_v<T, double> == false, "only 4 byte supported");
        uint8_t value[sizeof(T)];
        for (uint32_t i = 0; i < sizeof(T); ++i) {
          value[i] = ((uint8_t*)&val)[i];
        }
        shader_use(nr);
        shader_t& shader = shader_get(nr);
        auto found = shader.uniform_type_table.find(name);
        if (found == shader.uniform_type_table.end()) {
          //fan::print("failed to set uniform value");
          return;
          //fan::throw_error("failed to set uniform value");
        }

        size_t hash0 = std::hash<std::string>{}(name);
        size_t hash1 = std::hash<decltype(shader_nr_t::NRI)>{}(nr.NRI);
        auto shader_loc_it = shader_location_cache.find(hash0 ^ hash1);
        if (shader_loc_it == shader_location_cache.end()) {
          fan::opengl::GLint location = opengl.call(opengl.glGetUniformLocation, shader.id, name.c_str());
          if (location == -1) {
            return;
          }
          shader_loc_it = shader_location_cache.emplace(hash0 ^ hash1, location).first;
        }
        fan::opengl::GLint location = shader_loc_it->second;


#if fan_debug >= fan_debug_insanity
        fan_validate_value(location, validate_error_message(name));
#endif

        switch (fan::get_hash(found->second)) {
        case fan::get_hash(std::string_view("bool")): {
          if constexpr (not_non_arithmethic_types<T>) {
            opengl.call(opengl.glUniform1i, location, *(int*)value);
          }
          break;
        }
        case fan::get_hash(std::string_view("sampler2D")):
        case fan::get_hash(std::string_view("int")): {
          if constexpr (not_non_arithmethic_types<T>) {
            opengl.call(opengl.glUniform1i, location, *(int*)value);
          }
          break;
        }
        case fan::get_hash(std::string_view("uint")): {
          if constexpr (not_non_arithmethic_types<T>) {
            opengl.call(opengl.glUniform1ui, location, *(uint32_t*)value);
          }
          break;
        }
        case fan::get_hash(std::string_view("float")): {
          if constexpr (not_non_arithmethic_types<T>) {
            opengl.call(opengl.glUniform1f, location, *(f32_t*)value);
          }
          break;
        }
        case fan::get_hash(std::string_view("vec2")): {
          if constexpr (std::is_same_v<T, fan::vec2> ||
            std::is_same_v<T, fan::vec3>) {
            opengl.call(opengl.glUniform2fv, location, 1, (f32_t*)&value[0]);
          }
          break;
        }
        case fan::get_hash(std::string_view("vec3")): {
          if constexpr (std::is_same_v<T, fan::vec3>) {
            opengl.call(opengl.glUniform3fv, location, 1, (f32_t*)&value[0]);
          }
          break;
        }
        case fan::get_hash(std::string_view("vec4")): {
          if constexpr (std::is_same_v<T, fan::vec4> || std::is_same_v<T, fan::color>) {
            opengl.call(opengl.glUniform4fv, location, 1, (f32_t*)&value[0]);
          }
          break;
        }
        case fan::get_hash(std::string_view("mat4")): {
          opengl.call(opengl.glUniformMatrix4fv, location, 1, fan::opengl::GL_FALSE, (f32_t*)&value[0]);
          break;
        }
        }
      }

      /*
      
        case fan::get_hash(std::string_view("int[]")): {
          shader->set_int(var_name, initial);
          break;
        }
        case fan::get_hash(std::string_view("uint[]")): {
          shader->set_int(var_name, initial);
          break;
        }
        case fan::get_hash(std::string_view("float[]")): {
          shader->set_float(var_name, initial);
          break;
        }
      */

      //-----------------------------shader-----------------------------




      //-----------------------------image-----------------------------

      struct image_t {
        fan::opengl::GLuint texture_id;
        fan::vec2 size;
      };

      struct gl_image_impl {
        #include <fan/graphics/opengl/image_list_builder_settings.h>
        #if defined(loco_opengl)
        #elif defined(loco_vulkan)
        #include <fan/graphics/vulkan/image_list_builder_settings.h>
        #endif
        #include <BLL/BLL.h>
      };

      using image_nr_t = gl_image_impl::image_list_NodeReference_t;

      struct image_format {
        static constexpr auto b8g8r8a8_unorm = fan::opengl::GL_BGRA;
        static constexpr auto r8_unorm = fan::opengl::GL_RED;
        static constexpr auto rg8_unorm = fan::opengl::GL_RG;
      };

      struct image_sampler_address_mode {
        static constexpr auto repeat = fan::opengl::GL_REPEAT;
        static constexpr auto mirrored_repeat = fan::opengl::GL_MIRRORED_REPEAT;
        static constexpr auto clamp_to_edge = fan::opengl::GL_CLAMP_TO_EDGE;
        static constexpr auto clamp_to_border = fan::opengl::GL_CLAMP_TO_BORDER;
        static constexpr auto mirrored_clamp_to_edge = fan::opengl::GL_MIRROR_CLAMP_TO_EDGE;
      };

      struct image_filter {
        static constexpr auto nearest = fan::opengl::GL_NEAREST;
        static constexpr auto linear = fan::opengl::GL_LINEAR;
      };

      struct image_load_properties_defaults {
        static constexpr uint32_t visual_output = fan::opengl::GL_REPEAT;
        static constexpr uint32_t internal_format = fan::opengl::GL_RGBA;
        static constexpr uint32_t format = fan::opengl::GL_RGBA;
        static constexpr uint32_t type = fan::opengl::GL_UNSIGNED_BYTE;
        static constexpr uint32_t min_filter = fan::opengl::GL_NEAREST;
        static constexpr uint32_t mag_filter = fan::opengl::GL_NEAREST;
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

      image_nr_t image_create();
      fan::opengl::GLuint& image_get(image_nr_t nr);
      image_t& image_get_data(image_nr_t nr);

      void image_erase(image_nr_t nr);

      void image_bind(image_nr_t nr);
      void image_unbind(image_nr_t nr);

      void image_set_settings(const image_load_properties_t& p);

      image_nr_t image_load(const fan::image::image_info_t& image_info);
      image_nr_t image_load(const fan::image::image_info_t& image_info, const image_load_properties_t& p);
      image_nr_t image_load(const fan::string& path);
      image_nr_t image_load(const fan::string& path, const image_load_properties_t& p);
      image_nr_t image_load(fan::color* colors, const fan::vec2ui& size_);
      image_nr_t image_load(fan::color* colors, const fan::vec2ui& size_, const image_load_properties_t& p);

      void image_unload(image_nr_t nr);

      image_nr_t create_missing_texture();
      image_nr_t create_transparent_texture();

      void image_reload_pixels(image_nr_t nr, const fan::image::image_info_t& image_info);
      void image_reload_pixels(image_nr_t nr, const fan::image::image_info_t& image_info, const image_load_properties_t& p);

      std::unique_ptr<uint8_t[]> image_get_pixel_data(image_nr_t nr, fan::opengl::GLenum format, fan::vec2 uvp = 0, fan::vec2 uvs = 1);

      image_nr_t create_image(const fan::color& color);
      image_nr_t create_image(const fan::color& color, const fan::opengl::context_t::image_load_properties_t& p);

      gl_image_impl::image_list_t image_list;

      //-----------------------------image-----------------------------

      //-----------------------------camera-----------------------------

      camera_list_t camera_list;

      camera_nr_t camera_create();
      camera_t& camera_get(camera_nr_t nr);
      void camera_erase(camera_nr_t nr);

      //void link(const camera_t& t);

      static constexpr f32_t znearfar = 0xffff;

      camera_nr_t camera_open(const fan::vec2& x, const fan::vec2& y);

      fan::vec3 camera_get_position(camera_nr_t nr);
      void camera_set_position(camera_nr_t nr, const fan::vec3& cp);

      fan::vec2 camera_get_size(camera_nr_t nr);

      void camera_set_ortho(camera_nr_t nr, fan::vec2 x, fan::vec2 y);
      void camera_set_perspective(camera_nr_t nr, f32_t fov, const fan::vec2& window_size);

      void camera_rotate(camera_nr_t nr, const fan::vec2& offset);

      //-----------------------------camera-----------------------------


      //-----------------------------viewport-----------------------------

      struct viewport_t {
        fan::vec2 viewport_position;
        fan::vec2 viewport_size;
      };

    protected:
      #include "viewport_list_builder_settings.h"
      #include <BLL/BLL.h>
    public:

      using viewport_nr_t = viewport_list_NodeReference_t;

      viewport_list_t viewport_list;

      viewport_nr_t viewport_create();
      viewport_t& viewport_get(viewport_nr_t nr);
      void viewport_erase(viewport_nr_t nr);

      fan::vec2 viewport_get_position(viewport_nr_t nr);
      fan::vec2 viewport_get_size(viewport_nr_t nr);


      void viewport_set(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size);
      void viewport_set(viewport_nr_t nr, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size);
      void viewport_zero(viewport_nr_t nr);

      bool inside(viewport_nr_t nr, const fan::vec2& position);
      bool inside_wir(viewport_nr_t nr, const fan::vec2& position);

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

namespace fan {
  struct pixel_format {
    enum {
      undefined,
      yuv420p,
      nv12,
    };

    constexpr static uint8_t get_texture_amount(uint8_t format) {
      switch (format) {
      case undefined: {
        return 0;
      }
      case yuv420p: {
        return 3;
      }
      case nv12: {
        return 2;
      }
      default: {
        fan::throw_error("invalid format");
        return undefined;
      }
      }
    }
    constexpr static std::array<fan::vec2ui, 4> get_image_sizes(uint8_t format, const fan::vec2ui& image_size) {
      switch (format) {
      case yuv420p: {
        return std::array<fan::vec2ui, 4>{image_size, image_size / 2, image_size / 2};
      }
      case nv12: {
        return std::array<fan::vec2ui, 4>{image_size, fan::vec2ui{ image_size.x / 2, image_size.y / 2}};
      }
      default: {
        fan::throw_error("invalid format");
        return std::array<fan::vec2ui, 4>{};
      }
      }
    }
    template <typename T>
    static constexpr std::array<T, 4> get_image_properties(uint8_t format) {
      switch (format) {
      case yuv420p: {
        return std::array<fan::opengl::context_t::image_load_properties_t, 4>{
          fan::opengl::context_t::image_load_properties_t{
            .internal_format = fan::opengl::context_t::image_format::r8_unorm,
            .format = fan::opengl::context_t::image_format::r8_unorm
          },
          fan::opengl::context_t::image_load_properties_t{
              .internal_format = fan::opengl::context_t::image_format::r8_unorm,
              .format = fan::opengl::context_t::image_format::r8_unorm
          },
          fan::opengl::context_t::image_load_properties_t{
              .internal_format = fan::opengl::context_t::image_format::r8_unorm,
              .format = fan::opengl::context_t::image_format::r8_unorm
          }
        };
      }
      case nv12: {
        return std::array<fan::opengl::context_t::image_load_properties_t, 4>{
          fan::opengl::context_t::image_load_properties_t{
            .internal_format = fan::opengl::context_t::image_format::r8_unorm,
            .format = fan::opengl::context_t::image_format::r8_unorm
          },
            fan::opengl::context_t::image_load_properties_t{
              .internal_format = fan::opengl::context_t::image_format::rg8_unorm,
              .format = fan::opengl::context_t::image_format::rg8_unorm
          }
        };
      }
      default: {
        fan::throw_error("invalid format");
        return std::array<fan::opengl::context_t::image_load_properties_t, 4>{};
      }
      }
    }
  };
}