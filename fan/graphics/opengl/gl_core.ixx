module;

#include <fan/utility.h>

#include <fan/graphics/gl_api.h>

export module fan.graphics.opengl.core;

import std;

#if defined(FAN_OPENGL)

import fan.physics.collision.rectangle;

import fan.types;
import fan.types.matrix;
import fan.time;

import fan.types.fstring;
import fan.types.color;
import fan.types.compile_time_string;

import fan.window;
import fan.utility;
import fan.print;
import fan.graphics.image_load;
import fan.graphics.common_context;

#include <fan/graphics/opengl/init.h>

template<typename T>
concept not_non_arithmethic_types = 
  !std::is_same_v<T, fan::vec2> &&
  !std::is_same_v<T, fan::vec3> &&
  !std::is_same_v<T, fan::vec4> &&
  !std::is_same_v<T, fan::color>
;

#define __fan_internal_camera_list (*fan::graphics::ctx().camera_list)
#define __fan_internal_shader_list (*fan::graphics::ctx().shader_list)
#define __fan_internal_image_list (*fan::graphics::ctx().image_list)
#define __fan_internal_viewport_list (*fan::graphics::ctx().viewport_list)

namespace fan::opengl {
  struct opengl_t {
    int major = -1;
    int minor = -1;
    void open();
  };
}

export namespace fan::opengl {
  struct context_t {
    struct properties_t { };

    void print_version();

    static void error_callback(int error, const char* description);

    void open(const properties_t& p = properties_t());
    void close();
    void internal_close();
    void render(fan::window_t& window);
    void set_depth_test(bool flag);
    void set_blending(bool flag);
    void set_stencil_test(bool flag);
    void set_stencil_op(GLenum sfail, GLenum dpfail, GLenum dppass);
    void set_vsync(fan::window_t* window, bool flag);

    struct pbo_t {
      GLuint id = 0;
      std::size_t size = 0;
    };

    pbo_t pbo_create(std::size_t size);
    void pbo_destroy(pbo_t& p);
    std::uint8_t* pbo_map_write(const pbo_t& p);
    void pbo_unmap();
    void pbo_upload_to_texture(
      fan::graphics::image_nr_t nr,
      const pbo_t& p,
      fan::vec2ui size,
      std::uintptr_t global_format,
      GLenum type = GL_UNSIGNED_BYTE
    );

    template<typename fill_fn_t>
    void pbo_write_and_upload(
      const pbo_t& write_pbo,
      const pbo_t& read_pbo,
      fan::graphics::image_nr_t nr,
      fan::vec2ui size,
      std::uintptr_t global_format,  // fan::graphics::image_format_e
      fill_fn_t&& fill,
      GLenum type = GL_UNSIGNED_BYTE
    ) {
      std::uint8_t* ptr = pbo_map_write(write_pbo);
      if (ptr) {
        fill(ptr, write_pbo.size);
        pbo_unmap();
      }
      pbo_upload_to_texture(nr, read_pbo, size, global_format, type);
    }

    static void message_callback(
      GLenum source,
      GLenum type,
      GLuint id,
      GLenum severity,
      GLsizei length,
      const GLchar* message,
      const void* userParam
    );

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
      std::uint32_t vertex = -1, fragment = -1;
    };

    //static constexpr auto shader_validate_error_message = [](const std::string& str) {
    //  return "failed to set value for:" + str + " check if variable is used in file so that its not optimized away";
    //};

    //static constexpr auto validate_error_message = [](const std::string& str) {
    //  return "failed to set value for:" + str + " check if variable is used in file so that its not optimized away";
    //  };


    template <typename T>
    void shader_get_value(fan::graphics::shader_nr_t nr, const std::string_view name, T& val) {
      shader_use(nr);
      shader_t& shader = shader_get(nr);
      auto found = __fan_internal_shader_list[nr].uniform_type_table.find(std::string(name));
      if (found == __fan_internal_shader_list[nr].uniform_type_table.end()) {
        return;
      }

      std::size_t hash0 = std::hash<std::string_view>{}(name);
      std::size_t hash1 = std::hash<decltype(fan::graphics::shader_nr_t::NRI)>{}(nr.NRI);
      auto shader_loc_it = shader_location_cache.find(hash0 ^ hash1);
      if (shader_loc_it == shader_location_cache.end()) {
        GLint location = fan_opengl_call(glGetUniformLocation(shader.id, std::string(name).c_str()));
        if (location == -1) {
          return;
        }
        shader_loc_it = shader_location_cache.emplace(hash0 ^ hash1, location).first;
      }
      GLint location = shader_loc_it->second;

      switch (fan::get_hash(found->second)) {
        case fan::get_hash(std::string_view("bool")): {
          int temp;
          fan_opengl_call(glGetUniformiv(shader.id, location, &temp));
          if constexpr (std::is_same_v<T, bool>) {
            val = temp;
          }
          break;
        }
        case fan::get_hash(std::string_view("sampler2D")):
        case fan::get_hash(std::string_view("int")): {
          fan_opengl_call(glGetUniformiv(shader.id, location, (GLint*)&val));
          break;
        }
        case fan::get_hash(std::string_view("uint")): {
          fan_opengl_call(glGetUniformuiv(shader.id, location, (GLuint*)&val));
          break;
        }
        case fan::get_hash(std::string_view("float")):
        case fan::get_hash(std::string_view("vec2")):
        case fan::get_hash(std::string_view("vec3")):
        case fan::get_hash(std::string_view("vec4")):
        case fan::get_hash(std::string_view("mat4")): {
          fan_opengl_call(glGetUniformfv(shader.id, location, (GLfloat*)&val));
          break;
        }
      }
    }

    template <typename T>
    void shader_set_value(fan::graphics::shader_nr_t nr, const std::string_view name, const T& val) {
      static_assert(!std::is_same_v<T, std::uint8_t>, "only 4 std::byte supported");
      static_assert(!std::is_same_v<T, std::uint16_t>, "only 4 std::byte supported");
      static_assert(std::is_same_v<T, bool> == false || !std::is_same_v<T, int>, "only 4 std::byte supported");
      static_assert(std::is_same_v<T, double> == false, "only 4 std::byte supported");
      std::uint8_t value[sizeof(T)];
      for (std::uint32_t i = 0; i < sizeof(T); ++i) {
        value[i] = ((std::uint8_t*)&val)[i];
      }
      shader_use(nr);
      shader_t& shader = shader_get(nr);
      auto& context = *this;
      auto found = __fan_internal_shader_list[nr].uniform_type_table.find(std::string(name));
      if (found == __fan_internal_shader_list[nr].uniform_type_table.end()) {
        //fan::print("failed to set uniform value");
        return;
        //fan::throw_error("failed to set uniform value");
      }

      std::size_t hash0 = std::hash<std::string_view>{}(name);
      std::size_t hash1 = std::hash<decltype(fan::graphics::shader_nr_t::NRI)>{}(nr.NRI);
      auto shader_loc_it = shader_location_cache.find(hash0 ^ hash1);
      if (shader_loc_it == shader_location_cache.end()) {
        GLint location = fan_opengl_call(glGetUniformLocation(shader.id, std::string(name).c_str()));
        if (location == -1) {
          return;
        }
        shader_loc_it = shader_location_cache.emplace(hash0 ^ hash1, location).first;
      }
      GLint location = shader_loc_it->second;


#if FAN_DEBUG >= fan_debug_insane
      //fan_validate_value(location, validate_error_message(name));
#endif

      switch (fan::get_hash(found->second)) {
      case fan::get_hash(std::string_view("bool")): {
        if constexpr (not_non_arithmethic_types<T>) {
          fan_opengl_call(glUniform1i(location, *(bool*)value ? 1 : 0));
        }
        break;
      }
      case fan::get_hash(std::string_view("sampler2D")):
      case fan::get_hash(std::string_view("int")): {
        if constexpr (not_non_arithmethic_types<T>) {
          fan_opengl_call(glUniform1i(location, *(int*)value));
        }
        break;
      }
      case fan::get_hash(std::string_view("uint")): {
        if constexpr (not_non_arithmethic_types<T>) {
          fan_opengl_call(glUniform1ui(location, *(std::uint32_t*)value));
        }
        break;
      }
      case fan::get_hash(std::string_view("float")): {
        if constexpr (not_non_arithmethic_types<T>) {
          fan_opengl_call(glUniform1f(location, *(f32_t*)value));
        }
        break;
      }
      case fan::get_hash(std::string_view("vec2")): {
        if constexpr (std::is_same_v<T, fan::vec2> ||
          std::is_same_v<T, fan::vec3>) {
          fan_opengl_call(glUniform2fv(location, 1, (f32_t*)&value[0]));
        }
        break;
      }
      case fan::get_hash(std::string_view("vec3")): {
        if constexpr (std::is_same_v<T, fan::vec3>) {
          fan_opengl_call(glUniform3fv(location, 1, (f32_t*)&value[0]));
        }
        break;
      }
      case fan::get_hash(std::string_view("vec4")): {
        if constexpr (std::is_same_v<T, fan::vec4> || std::is_same_v<T, fan::color>) {
          fan_opengl_call(glUniform4fv(location, 1, (f32_t*)&value[0]));
        }
        break;
      }
      case fan::get_hash(std::string_view("mat4")): {
        fan_opengl_call(glUniformMatrix4fv(location, 1, GL_FALSE, (f32_t*)&value[0]));
        break;
      }
      }
    }
    template <typename T>
    void shader_set_value(fan::graphics::shader_nr_t nr, const std::string_view name, const std::vector<T>& val) {
      shader_use(nr);
      auto found = __fan_internal_shader_list[nr].uniform_type_table.find(std::string(name));
      if (found == __fan_internal_shader_list[nr].uniform_type_table.end()) return;

      std::size_t hash0 = std::hash<std::string_view> {}(name);
      std::size_t hash1 = std::hash<decltype(fan::graphics::shader_nr_t::NRI)> {}(nr.NRI);
      auto shader_loc_it = shader_location_cache.find(hash0 ^ hash1);
      if (shader_loc_it == shader_location_cache.end()) {
        GLint location = glGetUniformLocation(shader_get(nr).id, std::string(name).data());
        if (location == -1) return;
        shader_loc_it = shader_location_cache.emplace(hash0 ^ hash1, location).first;
      }
      GLint location = shader_loc_it->second;
      GLsizei count = (GLsizei)val.size();

      switch (fan::get_hash(found->second)) {
      case fan::get_hash(std::string_view("float")):
        glUniform1fv(location, count, (f32_t*)val.data()); break;
      case fan::get_hash(std::string_view("vec2")):
        glUniform2fv(location, count, (f32_t*)val.data()); break;
      case fan::get_hash(std::string_view("vec3")):
        glUniform3fv(location, count, (f32_t*)val.data()); break;
      case fan::get_hash(std::string_view("vec4")):
        glUniform4fv(location, count, (f32_t*)val.data()); break;
      case fan::get_hash(std::string_view("int")):
        glUniform1iv(location, count, (GLint*)val.data()); break;
      }
    }

    //-----------------------------shader-----------------------------

    //-----------------------------image-----------------------------

    struct image_t {
      GLuint texture_id;
    };

    struct image_load_properties_defaults {
      static constexpr std::uint32_t visual_output = GL_REPEAT;
      static constexpr std::uint32_t internal_format = GL_RGBA;
      static constexpr std::uint32_t format = GL_RGBA;
      static constexpr std::uint32_t type = GL_UNSIGNED_BYTE;
      static constexpr std::uint32_t min_filter = GL_LINEAR;
      static constexpr std::uint32_t mag_filter = GL_LINEAR;
    };

    struct image_load_properties_t {
      std::uint32_t            visual_output = image_load_properties_defaults::visual_output;
      std::uintptr_t           internal_format = image_load_properties_defaults::internal_format;
      std::uintptr_t           format = image_load_properties_defaults::format;
      std::uintptr_t           type = image_load_properties_defaults::type;
      std::uintptr_t           min_filter = image_load_properties_defaults::min_filter;
      std::uintptr_t           mag_filter = image_load_properties_defaults::mag_filter;
    };

    struct image_cache_entry_t {
      fan::graphics::image_nr_t nr;
      std::uint32_t ref_count;
    };

    fan::graphics::image_nr_t image_load_internal(
      fan::str_view_t path,
      const fan::opengl::context_t::image_load_properties_t& p,
      const std::source_location& callers_path = std::source_location::current()
    );
    void image_reload_internal(
      fan::graphics::image_nr_t nr,
      fan::str_view_t path,
      const fan::opengl::context_t::image_load_properties_t& p,
      const std::source_location& callers_path
    );
    void image_clear_cache();

    std::unordered_map<std::string, image_cache_entry_t> image_cache;

    GLenum get_format_from_channels(int channels);

    static constexpr fan::vec4_wrap_t<fan::vec2> default_texture_coordinates = fan::vec4_wrap_t<fan::vec2>(
      fan::vec2(0, 0),
      fan::vec2(1, 0),
      fan::vec2(1, 1),
      fan::vec2(0, 1)
    );
    //-----------------------------image-----------------------------

    //-----------------------------viewport-----------------------------

    //-----------------------------viewport-----------------------------

    // draw modes
    struct primitive_topology_t {
      static constexpr std::uint32_t points = GL_POINTS;
      static constexpr std::uint32_t lines = GL_LINES;
      static constexpr std::uint32_t line_strip = GL_LINE_STRIP;
      static constexpr std::uint32_t line_loop = GL_LINE_LOOP;
      static constexpr std::uint32_t triangles = GL_TRIANGLES;
      static constexpr std::uint32_t triangle_strip = GL_TRIANGLE_STRIP;
      static constexpr std::uint32_t triangle_fan = GL_TRIANGLE_FAN;
    #if !defined(__wasm__)
      static constexpr std::uint32_t lines_with_adjacency = GL_LINES_ADJACENCY;
      static constexpr std::uint32_t line_strip_with_adjacency = GL_LINE_STRIP_ADJACENCY;
      static constexpr std::uint32_t triangles_with_adjacency = GL_TRIANGLES_ADJACENCY;
      static constexpr std::uint32_t triangle_strip_with_adjacency = GL_TRIANGLE_STRIP_ADJACENCY;
    #endif
    };


    fan::graphics::shader_nr_t shader_create();
    fan::opengl::context_t::shader_t& shader_get(fan::graphics::shader_nr_t nr);
    void shader_erase(fan::graphics::shader_nr_t nr);

    bool shader_check_compile_errors(GLuint shader, const std::string_view file_path, const std::string& type);
    bool shader_check_compile_errors(fan::graphics::shader_data_t& common_shader, const std::string_view file_path, const std::string& type);

    void shader_use(fan::graphics::shader_nr_t nr);
    void shader_set_vertex(fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& vertex_code);
    void shader_set_fragment(fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& fragment_code);

    static void parse_uniforms(
      const std::string& shaderData,
      std::unordered_map<std::string, std::string>& uniform_type_table
    );

    bool shader_compile(fan::graphics::shader_nr_t nr);

    fan::graphics::context_camera_t& camera_get(fan::graphics::camera_nr_t nr);
    void shader_set_camera(fan::graphics::shader_nr_t nr, fan::graphics::camera_nr_t camera_nr);


    //-----------------------------shader-----------------------------
    //-----------------------------shader-----------------------------
    //-----------------------------shader-----------------------------
    //-----------------------------shader-----------------------------
    //-----------------------------shader-----------------------------
    //-----------------------------shader-----------------------------
    //-----------------------------shader-----------------------------
    //-----------------------------shader-----------------------------
    //-----------------------------shader-----------------------------
    //-----------------------------shader-----------------------------
    //-----------------------------shader-----------------------------
    //-----------------------------shader-----------------------------

    //*************************************************************************
    //*************************************************************************
    //*************************************************************************
    //*************************************************************************

    //-----------------------------image-----------------------------
    //-----------------------------image-----------------------------
    //-----------------------------image-----------------------------
    //-----------------------------image-----------------------------
    //-----------------------------image-----------------------------
    //-----------------------------image-----------------------------
    //-----------------------------image-----------------------------
    //-----------------------------image-----------------------------
    //-----------------------------image-----------------------------
    //-----------------------------image-----------------------------
    //-----------------------------image-----------------------------
    //-----------------------------image-----------------------------

    fan::opengl::context_t::image_t& image_get(fan::graphics::image_nr_t nr);
    GLuint& image_get_handle(fan::graphics::image_nr_t nr);
    fan::graphics::image_nr_t image_create();
    void image_erase(fan::graphics::image_nr_t nr);
    void image_bind(fan::graphics::image_nr_t nr);
    void image_unbind(fan::graphics::image_nr_t nr);
    fan::graphics::image_load_properties_t& image_get_settings(fan::graphics::image_nr_t nr);
    void image_set_settings(fan::graphics::image_nr_t nr, const fan::opengl::context_t::image_load_properties_t& p);
    fan::graphics::image_nr_t image_load(const fan::image::info_t& image_info, const fan::opengl::context_t::image_load_properties_t& lp);
    fan::graphics::image_nr_t create_missing_texture();
    fan::graphics::image_nr_t create_transparent_texture(fan::opengl::context_t& context);
    fan::graphics::image_nr_t image_load(fan::str_view_t path, const fan::opengl::context_t::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current());
    fan::graphics::image_nr_t image_load(const fan::image::info_t& image_info);
    fan::graphics::image_nr_t image_load(fan::color* colors, const fan::vec2ui& size_, const fan::opengl::context_t::image_load_properties_t& p);
    fan::graphics::image_nr_t image_load(fan::color* colors, const fan::vec2ui& size_);
    fan::graphics::image_nr_t image_load(fan::str_view_t path, const std::source_location& callers_path = std::source_location::current());
    void image_unload(fan::graphics::image_nr_t nr);
    void image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info, const fan::opengl::context_t::image_load_properties_t& lp);
    void image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info);
    void image_reload(fan::graphics::image_nr_t nr, fan::str_view_t path, const fan::opengl::context_t::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current());
    void image_reload(fan::graphics::image_nr_t nr, fan::str_view_t path);
    std::vector<std::uint8_t> image_get_pixel_data(fan::graphics::image_nr_t nr, GLenum format, fan::vec2 uvp, fan::vec2 uvs);
    fan::graphics::image_nr_t image_create(const fan::color& color, const fan::opengl::context_t::image_load_properties_t& p);
    fan::graphics::image_nr_t image_create(const fan::color& color);

    //-----------------------------image-----------------------------
    //-----------------------------image-----------------------------
    //-----------------------------image-----------------------------
    //-----------------------------image-----------------------------
    //-----------------------------image-----------------------------
    //-----------------------------image-----------------------------
    //-----------------------------image-----------------------------
    //-----------------------------image-----------------------------
    //-----------------------------image-----------------------------
    //-----------------------------image-----------------------------
    //-----------------------------image-----------------------------
    //-----------------------------image-----------------------------







    //-----------------------------camera-----------------------------
    //-----------------------------camera-----------------------------
    //-----------------------------camera-----------------------------
    //-----------------------------camera-----------------------------
    //-----------------------------camera-----------------------------
    //-----------------------------camera-----------------------------
    //-----------------------------camera-----------------------------
    //-----------------------------camera-----------------------------
    //-----------------------------camera-----------------------------
    //-----------------------------camera-----------------------------
    //-----------------------------camera-----------------------------
    //-----------------------------camera-----------------------------


    fan::graphics::camera_nr_t camera_create();
    void camera_erase(fan::graphics::camera_nr_t nr);
    void camera_set_ortho(fan::graphics::camera_nr_t nr, fan::vec2 x, fan::vec2 y);
    void camera_update_projection(fan::graphics::camera_nr_t nr);
    void camera_update_view(fan::graphics::camera_nr_t nr);
    fan::graphics::camera_nr_t camera_create(const fan::vec2& x, const fan::vec2& y);
    // Returns the raw translation offset of the camera matrix.
    // For an orthographic projection starting at (0,0), this represents the top-left corner.
    // For a symmetric projection (e.g., -width/2 to width/2), this represents the center.
    fan::vec3 camera_get_position(fan::graphics::camera_nr_t nr);
    void camera_set_position(fan::graphics::camera_nr_t nr, const fan::vec3& cp);
    // Returns the true world-space center of the camera's view,
    // regardless of how the projection matrix was initialized.
    fan::vec3 camera_get_center(fan::graphics::camera_nr_t nr);
    void camera_set_center(fan::graphics::camera_nr_t nr, const fan::vec3& cp);
    fan::vec2 camera_get_size(fan::graphics::camera_nr_t nr);
    f32_t camera_get_zoom(fan::graphics::camera_nr_t nr);
    void camera_set_zoom(fan::graphics::camera_nr_t nr, f32_t new_zoom);
    void camera_set_perspective(fan::graphics::camera_nr_t nr, f32_t fov, const fan::vec2& window_size);

    void camera_rotate(fan::graphics::camera_nr_t nr, const fan::vec2& offset);
    void viewport_set(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size);
    fan::graphics::context_viewport_t& viewport_get(fan::graphics::viewport_nr_t nr);
    void viewport_set(fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size);
    fan::graphics::viewport_nr_t viewport_create();
    void viewport_erase(fan::graphics::viewport_nr_t nr);
    fan::vec2 viewport_get_position(fan::graphics::viewport_nr_t nr);
    fan::vec2 viewport_get_size(fan::graphics::viewport_nr_t nr);
    void viewport_zero(fan::graphics::viewport_nr_t nr);
    bool viewport_inside(fan::graphics::viewport_nr_t nr, const fan::vec2& position);
    bool viewport_inside_wir(fan::graphics::viewport_nr_t nr, const fan::vec2& position);

    static std::uint32_t global_to_opengl_format(std::uintptr_t format);
    static std::uint32_t global_to_opengl_type(std::uintptr_t type);
    static std::uint32_t global_to_opengl_address_mode(std::uint32_t mode);
    static std::uint32_t global_to_opengl_filter(std::uintptr_t filter);
    static std::uint32_t opengl_to_global_format(std::uintptr_t format);
    static std::uint32_t opengl_to_global_type(std::uintptr_t type);
    static std::uint32_t opengl_to_global_address_mode(std::uint32_t mode);
    static std::uint32_t opengl_to_global_filter(std::uintptr_t filter);

    void close(fan::opengl::context_t& context);

    static fan::opengl::context_t::image_load_properties_t image_global_to_opengl(const fan::graphics::image_load_properties_t& p);
    static fan::graphics::image_load_properties_t image_opengl_to_global(const fan::opengl::context_t::image_load_properties_t& p);

    std::unordered_map<std::size_t, int> shader_location_cache;
    fan::opengl::opengl_t opengl;
    GLuint current_program = -1;
  };
}

template void fan::opengl::context_t::shader_get_value<bool>(fan::graphics::shader_nr_t nr, const std::string_view name, bool& val);
template void fan::opengl::context_t::shader_get_value<fan::vec2>(fan::graphics::shader_nr_t nr, const std::string_view name, fan::vec2& val);
template void fan::opengl::context_t::shader_get_value<fan::vec3>(fan::graphics::shader_nr_t nr, const std::string_view name, fan::vec3& val);
template void fan::opengl::context_t::shader_get_value<fan::vec4>(fan::graphics::shader_nr_t nr, const std::string_view name, fan::vec4& val);
template void fan::opengl::context_t::shader_get_value<fan::mat4>(fan::graphics::shader_nr_t nr, const std::string_view name, fan::mat4& val);
template void fan::opengl::context_t::shader_get_value<fan::color>(fan::graphics::shader_nr_t nr, const std::string_view name, fan::color& val);
template void fan::opengl::context_t::shader_get_value<std::uint32_t>(fan::graphics::shader_nr_t nr, const std::string_view name, std::uint32_t& val);
template void fan::opengl::context_t::shader_get_value<std::uint64_t>(fan::graphics::shader_nr_t nr, const std::string_view name, std::uint64_t& val);
template void fan::opengl::context_t::shader_get_value<int>(fan::graphics::shader_nr_t nr, const std::string_view name, int& val);
template void fan::opengl::context_t::shader_get_value<f32_t>(fan::graphics::shader_nr_t nr, const std::string_view name, f32_t& val);

template void fan::opengl::context_t::shader_set_value<fan::vec2>(fan::graphics::shader_nr_t nr, const std::string_view name, const fan::vec2& val);
template void fan::opengl::context_t::shader_set_value<fan::vec3>(fan::graphics::shader_nr_t nr, const std::string_view name, const fan::vec3& val);
template void fan::opengl::context_t::shader_set_value<fan::vec4>(fan::graphics::shader_nr_t nr, const std::string_view name, const fan::vec4& val);
template void fan::opengl::context_t::shader_set_value<fan::mat4>(fan::graphics::shader_nr_t nr, const std::string_view name, const fan::mat4& val);
template void fan::opengl::context_t::shader_set_value<fan::color>(fan::graphics::shader_nr_t nr, const std::string_view name, const fan::color& val);
template void fan::opengl::context_t::shader_set_value<std::uint32_t>(fan::graphics::shader_nr_t nr, const std::string_view name, const std::uint32_t& val);
template void fan::opengl::context_t::shader_set_value<std::uint64_t>(fan::graphics::shader_nr_t nr, const std::string_view name, const std::uint64_t& val);
template void fan::opengl::context_t::shader_set_value<int>(fan::graphics::shader_nr_t nr, const std::string_view name, const int& val);
template void fan::opengl::context_t::shader_set_value<f32_t>(fan::graphics::shader_nr_t nr, const std::string_view name, const f32_t& val);
template void fan::opengl::context_t::shader_set_value<fan::vec1_wrap_t<f32_t>>(fan::graphics::shader_nr_t nr, const std::string_view name, const fan::vec1_wrap_t<f32_t>& val);
template void fan::opengl::context_t::shader_set_value<fan::vec_wrap_t<1, f32_t>>(fan::graphics::shader_nr_t nr, const std::string_view name, const fan::vec_wrap_t<1, f32_t>& val);
template void fan::opengl::context_t::shader_set_value<fan::vec_wrap_t<2, f32_t>>(fan::graphics::shader_nr_t nr, const std::string_view name, const fan::vec_wrap_t<2, f32_t>& val);


export namespace fan::opengl::core {
  int get_buffer_size(fan::opengl::context_t& context, GLenum target_buffer, GLuint buffer_object);
  void write_glbuffer(fan::opengl::context_t& context, GLuint buffer, const void* data, std::uintptr_t size, std::uint32_t usage, GLenum target);
  void get_glbuffer(fan::opengl::context_t& context, void* data, GLuint buffer_id, std::uintptr_t size, std::uintptr_t offset, GLenum target);
  void edit_glbuffer(fan::opengl::context_t& context, GLuint buffer, const void* data, std::uintptr_t offset, std::uintptr_t size, std::uintptr_t target);

  // not tested
  int get_bound_buffer(fan::opengl::context_t& context);
  void reserve_glbuffer(
    fan::opengl::context_t& ctx,
    GLuint buffer,
    std::uint32_t& capacity,
    std::uint32_t needed,
    std::uint32_t usage,
    GLenum target
  );
  void append_glbuffer(
    fan::opengl::context_t& ctx,
    GLuint buffer,
    std::uintptr_t& size_bytes,
    std::uintptr_t& capacity_bytes,
    const void* data,
    std::uintptr_t data_size,
    std::uint32_t usage,
    GLenum target
  );
  //#pragma pack(push, 1)
  //#pragma pack(pop)

  struct vao_t {

    void open(fan::opengl::context_t& context);
    void close(fan::opengl::context_t& context);

    void bind(fan::opengl::context_t& context) const;
    void unbind(fan::opengl::context_t& context) const;

    bool is_valid() const;

    GLuint m_buffer = (GLuint)-1;
  };

  struct vbo_t {

    void open(fan::opengl::context_t& context, GLenum target_);
    void close(fan::opengl::context_t& context);

    bool is_valid() const;

    void bind(fan::opengl::context_t& context) const;

    void get_vram_instance(fan::opengl::context_t& context, void* data, std::uintptr_t size, std::uintptr_t offset);

    // only for target GL_UNIFORM_BUFFER
    void bind_buffer_range(fan::opengl::context_t& context, std::uint32_t total_size);

    void edit_buffer(fan::opengl::context_t& context, const void* data, std::uintptr_t offset, std::uintptr_t size);

    void write_buffer(fan::opengl::context_t& context, const void* data, std::uintptr_t size);

    GLuint m_buffer = (GLuint)-1;
    GLenum m_target = (GLuint)-1;
    std::uint32_t m_usage = GL_DYNAMIC_DRAW;
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
    // texture must be bound with texture.bind();
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

  std::uint32_t get_draw_mode(std::uint8_t draw_mode);
}

namespace fan::graphics {
  export fan::graphics::context_functions_t get_gl_context_functions();
}

export namespace fan::graphics {
  fan::opengl::context_t& get_gl_context();
}

#endif