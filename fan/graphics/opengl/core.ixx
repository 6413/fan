module;

#include <fan/graphics/opengl/init.h>

#include <vector>
#include <string>
#include <source_location>
#include <unordered_map>

#ifndef camera_list
  #define __fan_internal_camera_list (*(fan::graphics::camera_list_t*)fan::graphics::get_camera_list((uint8_t*)this))
#endif

#ifndef shader_list
  #define __fan_internal_shader_list (*(fan::graphics::shader_list_t*)fan::graphics::get_shader_list((uint8_t*)this))
#endif

#ifndef image_list
  #define __fan_internal_image_list (*(fan::graphics::image_list_t*)fan::graphics::get_image_list((uint8_t*)this))
#endif

#ifndef viewport_list
  #define __fan_internal_viewport_list (*(fan::graphics::viewport_list_t*)fan::graphics::get_viewport_list((uint8_t*)this))
#endif

export module fan.graphics.opengl.core;

import fan.physics.collision.rectangle;

import fan.types.fstring;
import fan.types.color;

import fan.window;
import fan.utility;
export import fan.print;
export import fan.graphics.image_load;
export import fan.graphics.common_context;

template<typename T>
concept not_non_arithmethic_types = !std::is_same_v<T, fan::vec2> &&
!std::is_same_v<T, fan::vec3> &&
!std::is_same_v<T, fan::vec4> &&
!std::is_same_v<T, fan::color>;

namespace fan {
  namespace opengl {
    struct opengl_t {
      int major = -1;
      int minor = -1;
      void open() {
        static uint8_t init = 1;
        if (init == 0) {
          return;
        }
        init = 0;
        if (GLenum err = glewInit() != GLEW_OK) {
          fan::throw_error(std::string("glew init error:") + std::string((const char*)glewGetErrorString(err)));
        }
      }
    };
  }
}

export namespace fan {
  namespace opengl {
    using ::fan_track_opengl_calls;
    struct context_t {
      std::unordered_map<size_t, int> shader_location_cache;

      void print_version() {
        fan::print("opengl version supported:", fan_opengl_call(glGetString(GL_VERSION)));
      }

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

      void open(const properties_t& p = properties_t()) {
        opengl.open();
      }
      void close();
      void internal_close() {
        fan::opengl::context_t& context = *this;
        {
          fan::graphics::shader_list_t::nrtra_t nrtra;
          fan::graphics::shader_nr_t nr;
          nrtra.Open(&__fan_internal_shader_list, &nr);
          while (nrtra.Loop(&__fan_internal_shader_list, &nr)) {
            delete static_cast<fan::opengl::context_t::shader_t*>(__fan_internal_shader_list[nr].internal);
          }
          nrtra.Close(&__fan_internal_shader_list);
        }
        {
          fan::graphics::image_list_t::nrtra_t nrtra;
          fan::graphics::image_nr_t nr;
          nrtra.Open(&__fan_internal_image_list, &nr);
          while (nrtra.Loop(&__fan_internal_image_list, &nr)) {
            delete static_cast<fan::opengl::context_t::image_t*>(__fan_internal_image_list[nr].internal);
          }
          nrtra.Close(&__fan_internal_image_list);
        }
      }

      void render(fan::window_t& window){
        glfwSwapBuffers(window.glfw_window);
      }

      void set_depth_test(bool flag) {
        if (flag) {
          fan_opengl_call(glEnable(GL_DEPTH_TEST));
        }
        else {
          fan_opengl_call(glDisable(GL_DEPTH_TEST));
        }
      }
      void set_blending(bool flag){
        if (flag) {
          fan_opengl_call(glDisable(GL_BLEND));
        }
        else {
          fan_opengl_call(glEnable(GL_BLEND));
          fan_opengl_call(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
        }
      }

      void set_stencil_test(bool flag){
        if (flag) {
          fan_opengl_call(glEnable(GL_STENCIL_TEST));
        }
        else {
          fan_opengl_call(glDisable(GL_STENCIL_TEST));
        }
      }
      void set_stencil_op(GLenum sfail, GLenum dpfail, GLenum dppass){
        fan_opengl_call(glStencilOp(sfail, dpfail, dppass));
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
        const void* userParam) {
        if (type == 33361 || type == 33360) { // gl_static_draw
          return;
        }
        fan::print_no_space(type == GL_DEBUG_TYPE_ERROR ? "opengl error:" : "", type, ", severity:", severity, ", message:", message);
        fan::debug::print_stacktrace();
      }

      void set_error_callback(){
        fan_opengl_call(glEnable(GL_DEBUG_OUTPUT));
        fan_opengl_call(glDebugMessageCallback(message_callback, (void*)0));
      }

      void set_current(fan::window_t* window) {
        if (window == nullptr) {
          glfwMakeContextCurrent(nullptr);
        }
        else {
          glfwMakeContextCurrent(window->glfw_window);
        }
      }

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

      //static constexpr auto shader_validate_error_message = [](const std::string& str) {
      //  return "failed to set value for:" + str + " check if variable is used in file so that its not optimized away";
      //};

      //static constexpr auto validate_error_message = [](const std::string& str) {
      //  return "failed to set value for:" + str + " check if variable is used in file so that its not optimized away";
      //  };

      template <typename T>
      void shader_set_value(fan::graphics::shader_nr_t nr, const std::string& name, const T& val) {
        static_assert(!std::is_same_v<T, uint8_t>, "only 4 byte supported");
        static_assert(!std::is_same_v<T, uint16_t>, "only 4 byte supported");
        static_assert(std::is_same_v<T, bool> == false || !std::is_same_v<T, int>, "only 4 byte supported");
        static_assert(std::is_same_v<T, double> == false, "only 4 byte supported");
        uint8_t value[sizeof(T)];
        for (uint32_t i = 0; i < sizeof(T); ++i) {
          value[i] = ((uint8_t*)&val)[i];
        }
        shader_use(nr);
        shader_t& shader = shader_get(nr);
        auto& context = *this;
        auto found = __fan_internal_shader_list[nr].uniform_type_table.find(name);
        if (found == __fan_internal_shader_list[nr].uniform_type_table.end()) {
          //fan::print("failed to set uniform value");
          return;
          //fan::throw_error("failed to set uniform value");
        }

        size_t hash0 = std::hash<std::string>{}(name);
        size_t hash1 = std::hash<decltype(fan::graphics::shader_nr_t::NRI)>{}(nr.NRI);
        auto shader_loc_it = shader_location_cache.find(hash0 ^ hash1);
        if (shader_loc_it == shader_location_cache.end()) {
          GLint location = fan_opengl_call(glGetUniformLocation(shader.id, name.c_str()));
          if (location == -1) {
            return;
          }
          shader_loc_it = shader_location_cache.emplace(hash0 ^ hash1, location).first;
        }
        GLint location = shader_loc_it->second;


#if fan_debug >= fan_debug_insane
        fan_validate_value(location, validate_error_message(name));
#endif

        switch (fan::get_hash(found->second)) {
        case fan::get_hash(std::string_view("bool")): {
          if constexpr (not_non_arithmethic_types<T>) {
            fan_opengl_call(glUniform1i(location, *(int*)value));
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
            fan_opengl_call(glUniform1ui(location, *(uint32_t*)value));
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

     //-----------------------------shader-----------------------------

      //-----------------------------image-----------------------------

      struct image_t {
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
        static constexpr auto nearest_mipmap_nearest = GL_NEAREST_MIPMAP_NEAREST;
        static constexpr auto linear_mipmap_nearest = GL_LINEAR_MIPMAP_NEAREST;
        static constexpr auto nearest_mipmap_linear = GL_NEAREST_MIPMAP_LINEAR;
        static constexpr auto linear_mipmap_linear = GL_LINEAR_MIPMAP_LINEAR;
      };

      struct image_load_properties_defaults {
        static constexpr uint32_t visual_output = GL_REPEAT;
        static constexpr uint32_t internal_format = GL_RGBA;
        static constexpr uint32_t format = GL_RGBA;
        static constexpr uint32_t type = GL_UNSIGNED_BYTE;
        static constexpr uint32_t min_filter = GL_LINEAR;
        static constexpr uint32_t mag_filter = GL_LINEAR;
      };

      struct image_load_properties_t {
        uint32_t            visual_output = image_load_properties_defaults::visual_output;
        uintptr_t           internal_format = image_load_properties_defaults::internal_format;
        uintptr_t           format = image_load_properties_defaults::format;
        uintptr_t           type = image_load_properties_defaults::type;
        uintptr_t           min_filter = image_load_properties_defaults::min_filter;
        uintptr_t           mag_filter = image_load_properties_defaults::mag_filter;
      };

      GLenum get_format_from_channels(int channels) {
        switch (channels) {
        case 1: return GL_RED;
        case 2: return GL_RG;
        case 3: return GL_RGB;
        case 4: return GL_RGBA;
        default: return GL_RGBA;
        }
      }

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
        static constexpr uint32_t points = GL_POINTS;
        static constexpr uint32_t lines = GL_LINES;
        static constexpr uint32_t line_strip = GL_LINE_STRIP;
        static constexpr uint32_t line_loop = GL_LINE_LOOP;
        static constexpr uint32_t triangles = GL_TRIANGLES;
        static constexpr uint32_t triangle_strip = GL_TRIANGLE_STRIP;
        static constexpr uint32_t triangle_fan = GL_TRIANGLE_FAN;
        static constexpr uint32_t lines_with_adjacency = GL_LINES_ADJACENCY;
        static constexpr uint32_t line_strip_with_adjacency = GL_LINE_STRIP_ADJACENCY;
        static constexpr uint32_t triangles_with_adjacency = GL_TRIANGLES_ADJACENCY;
        static constexpr uint32_t triangle_strip_with_adjacency = GL_TRIANGLE_STRIP_ADJACENCY;
      };

      //statics

      fan::graphics::shader_nr_t shader_create() {
        auto nr = __fan_internal_shader_list.NewNode();
        __fan_internal_shader_list[nr].internal = new fan::opengl::context_t::shader_t;
        return nr;
      }

      fan::opengl::context_t::shader_t& shader_get(fan::graphics::shader_nr_t nr) {
        return *(fan::opengl::context_t::shader_t*)__fan_internal_shader_list[nr].internal;
      }

      void shader_erase(fan::graphics::shader_nr_t nr) {
        auto& shader = shader_get(nr);
        fan_validate_buffer(shader.id, {
          fan_opengl_call(glValidateProgram(shader.id));
          int status = 0;
          fan_opengl_call(glGetProgramiv(shader.id, GL_VALIDATE_STATUS, &status));
          if (status) {
              fan_opengl_call(glDeleteProgram(shader.id));
          }
          shader.id = fan::uninitialized;
          });
        delete static_cast<fan::opengl::context_t::shader_t*>(__fan_internal_shader_list[nr].internal);
        __fan_internal_shader_list.Recycle(nr);
      }

      bool shader_check_compile_errors(GLuint shader, const std::string& type)
      {
        GLint success;

        bool program = type == "PROGRAM";

        if (program == false) {
          fan_opengl_call(glGetShaderiv(shader, GL_COMPILE_STATUS, &success));
        }
        else {
          fan_opengl_call(glGetProgramiv(shader, GL_LINK_STATUS, &success));
        }

        if (success) {
          return true;
        }

        int buffer_size = 0;
        fan_opengl_call(glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &buffer_size));


        if (buffer_size <= 0) {
          return false;
        }

        std::string buffer;
        buffer.resize(buffer_size);

        if (!success)
        {
          int test;
    #define get_info_log(is_program, program, str_buffer, size) \
                    if (is_program) \
                    fan_opengl_call(glGetProgramInfoLog(program, size, nullptr, buffer.data())); \
                    else \
                    fan_opengl_call(glGetShaderInfoLog(program, size, &test, buffer.data()));

          get_info_log(program, shader, buffer, buffer_size);

          fan::print("failed to compile: " + type, buffer);

          return false;
        }
        return true;
      }

      bool shader_check_compile_errors(fan::graphics::shader_data_t& common_shader, const std::string& type)
      {
        fan::opengl::context_t::shader_t& shader = *(fan::opengl::context_t::shader_t*)common_shader.internal;
        GLint success;

        bool vertex = type == "VERTEX";
        bool program = type == "PROGRAM";

        if (program == false) {
          fan_opengl_call(glGetShaderiv(vertex ? shader.vertex : shader.fragment, GL_COMPILE_STATUS, &success));
        }
        else {
          fan_opengl_call(glGetProgramiv(shader.id, GL_LINK_STATUS, &success));
        }

        if (success) {
          return true;
        }

        int buffer_size = 0;

        if (program == false) {
          fan_opengl_call(glGetShaderiv(vertex ? shader.vertex : shader.fragment, GL_INFO_LOG_LENGTH, &buffer_size));
        }
        else {
          // Use shader.id for program info log length as well
          fan_opengl_call(glGetProgramiv(shader.id, GL_INFO_LOG_LENGTH, &buffer_size));
        }

        if (buffer_size <= 0) {
          return false;
        }

        std::string buffer;
        buffer.resize(buffer_size);

        if (!success)
        {
          int test;
#define get_info_log(is_program, program, str_buffer, size) \
              if (is_program) \
              fan_opengl_call(glGetProgramInfoLog(program, size, nullptr, buffer.data())); \
              else \
              fan_opengl_call(glGetShaderInfoLog(program, size, &test, buffer.data()));

          get_info_log(program, program ? shader.id : (vertex ? shader.vertex : shader.fragment), buffer, buffer_size);

          fan::print("failed to compile: " + type, buffer, "filenames", common_shader.svertex, common_shader.sfragment);

          return false;
        }
        return true;
      }

      void shader_use(fan::graphics::shader_nr_t nr) {
        auto& shader = shader_get(nr);
        if (shader.id == current_program) {
          return;
        }
        fan_opengl_call(glUseProgram(shader.id));
        current_program = shader.id;
      }

      void shader_set_vertex(fan::graphics::shader_nr_t nr, const std::string& vertex_code) {
        auto& shader = shader_get(nr);

        if (shader.vertex != (uint32_t)fan::uninitialized) {
          fan_opengl_call(glDeleteShader(shader.vertex));
        }

        shader.vertex = fan_opengl_call(glCreateShader(GL_VERTEX_SHADER));
        __fan_internal_shader_list[nr].svertex = vertex_code;

        char* ptr = (char*)vertex_code.c_str();
        GLint length = vertex_code.size();

        fan_opengl_call(glShaderSource(shader.vertex, 1, &ptr, &length));
        fan_opengl_call(glCompileShader(shader.vertex));

        shader_check_compile_errors(__fan_internal_shader_list[nr], "VERTEX");
      }

      void shader_set_fragment(fan::graphics::shader_nr_t nr, const std::string& fragment_code) {
        auto& shader = shader_get(nr);

        if (shader.fragment != (uint32_t)-1) {
          fan_opengl_call(glDeleteShader(shader.fragment));
        }

        shader.fragment = fan_opengl_call(glCreateShader(GL_FRAGMENT_SHADER));
        __fan_internal_shader_list[nr].sfragment = fragment_code;

        char* ptr = (char*)fragment_code.c_str();
        GLint length = fragment_code.size();

        fan_opengl_call(glShaderSource(shader.fragment, 1, &ptr, &length));

        fan_opengl_call(glCompileShader(shader.fragment));
        shader_check_compile_errors(__fan_internal_shader_list[nr], "FRAGMENT");
      }

      static void parse_uniforms(std::string& shaderData, std::unordered_map<std::string, std::string>& uniform_type_table) {
        size_t pos = 0;

        while ((pos = shaderData.find("uniform", pos)) != std::string::npos) {
          size_t endLine = shaderData.find(';', pos);
          if (endLine == std::string::npos) break;

          std::string line = shaderData.substr(pos, endLine - pos + 1);

          line = line.substr(7);

          size_t start = line.find_first_not_of(" \t");
          if (start == std::string::npos) {
            pos = endLine + 1;
            continue;
          }
          line = line.substr(start);

          size_t space1 = line.find_first_of(" \t");
          if (space1 == std::string::npos) {
            pos = endLine + 1;
            continue;
          }

          std::string type = line.substr(0, space1);
          line = line.substr(space1);
          line = line.substr(line.find_first_not_of(" \t"));

          size_t varEnd = line.find_first_of("=;");
          std::string name = line.substr(0, varEnd);

          name.erase(0, name.find_first_not_of(" \t"));
          name.erase(name.find_last_not_of(" \t") + 1);

          uniform_type_table[name] = type;

          pos = endLine + 1;
        }
      }


      bool shader_compile(fan::graphics::shader_nr_t nr) {
        auto& shader = shader_get(nr);

        auto temp_id = fan_opengl_call(glCreateProgram());

        // Verify shaders are valid before attaching
        if (shader.vertex != (uint32_t)-1) {
          GLint compiled = 0;
          fan_opengl_call(glGetShaderiv(shader.vertex, GL_COMPILE_STATUS, &compiled));
          if (!compiled) {
            fan::print("Vertex shader not compiled successfully before linking");
            fan_opengl_call(glDeleteProgram(temp_id));
            return false;
          }
          fan_opengl_call(glAttachShader(temp_id, shader.vertex));
        }
        else {
          fan::print("Warning: No vertex shader attached");
        }

        if (shader.fragment != (uint32_t)-1) {
          GLint compiled = 0;
          fan_opengl_call(glGetShaderiv(shader.fragment, GL_COMPILE_STATUS, &compiled));
          if (!compiled) {
            fan::print("Fragment shader not compiled successfully before linking");
            fan_opengl_call(glDeleteProgram(temp_id));
            return false;
          }
          fan_opengl_call(glAttachShader(temp_id, shader.fragment));
        }
        else {
          fan::print("Warning: No fragment shader attached");
        }

        fan_opengl_call(glLinkProgram(temp_id));

        // Check program linking status
        GLint success;
        fan_opengl_call(glGetProgramiv(temp_id, GL_LINK_STATUS, &success));

        bool ret = true;
        if (!success) {
          fan::print("PROGRAM LINK FAILED - Dumping shader source:");
          fan::print("=== VERTEX SHADER SOURCE ===");
          fan::print(__fan_internal_shader_list[nr].svertex);
          fan::print("=== END VERTEX SHADER ===");

          fan::print("=== FRAGMENT SHADER SOURCE ===");
          fan::print(__fan_internal_shader_list[nr].sfragment);
          fan::print("=== END FRAGMENT SHADER ===");

          int buffer_size = 0;
          fan_opengl_call(glGetProgramiv(temp_id, GL_INFO_LOG_LENGTH, &buffer_size));

          fan::print("Program link failed. Info log length:", buffer_size);

          if (buffer_size > 1) { // OpenGL includes null terminator in length
            std::string buffer;
            buffer.resize(buffer_size);
            fan_opengl_call(glGetProgramInfoLog(temp_id, buffer_size, nullptr, buffer.data()));
            fan::print("Program link error:", buffer);
          }
          else {
            fan::print("Program link failed but no error message available");

            // Additional debugging - check validation
            fan_opengl_call(glValidateProgram(temp_id));
            GLint validate_status;
            fan_opengl_call(glGetProgramiv(temp_id, GL_VALIDATE_STATUS, &validate_status));
            fan::print("Program validation status:", validate_status);

            // Get validation info log
            GLint validate_log_length;
            fan_opengl_call(glGetProgramiv(temp_id, GL_INFO_LOG_LENGTH, &validate_log_length));
            if (validate_log_length > 1) {
              std::string validate_buffer;
              validate_buffer.resize(validate_log_length);
              fan_opengl_call(glGetProgramInfoLog(temp_id, validate_log_length, nullptr, validate_buffer.data()));
              fan::print("Program validation info:", validate_buffer);
            }

            // Check if shaders are still attached
            GLint attached_shader_count;
            fan_opengl_call(glGetProgramiv(temp_id, GL_ATTACHED_SHADERS, &attached_shader_count));
            fan::print("Number of attached shaders:", attached_shader_count);
          }
          ret = false;
        }

        if (ret == false) {
          fan_opengl_call(glDeleteProgram(temp_id));
          return false;
        }

        // Clean up old shaders
        if (shader.vertex != (uint32_t)-1) {
          fan_opengl_call(glDetachShader(temp_id, shader.vertex));
          fan_opengl_call(glDeleteShader(shader.vertex));
          shader.vertex = -1;
        }
        if (shader.fragment != (uint32_t)-1) {
          fan_opengl_call(glDetachShader(temp_id, shader.fragment));
          fan_opengl_call(glDeleteShader(shader.fragment));
          shader.fragment = -1;
        }

        // Clean up old program
        if (shader.id != (uint32_t)-1) {
          fan_opengl_call(glDeleteProgram(shader.id));
        }
        shader.id = temp_id;

        // Get uniform locations
        shader.projection_view[0] = fan_opengl_call(glGetUniformLocation(shader.id, "projection"));
        shader.projection_view[1] = fan_opengl_call(glGetUniformLocation(shader.id, "view"));


        // Parse uniforms
        std::string vertexData = __fan_internal_shader_list[nr].svertex;
        parse_uniforms(vertexData, __fan_internal_shader_list[nr].uniform_type_table);

        std::string fragmentData = __fan_internal_shader_list[nr].sfragment;
        parse_uniforms(fragmentData, __fan_internal_shader_list[nr].uniform_type_table);

        return ret;
      }

      fan::graphics::context_camera_t& camera_get(fan::graphics::camera_nr_t nr) {
        return __fan_internal_camera_list[nr];
      }

      void shader_set_camera(fan::graphics::shader_nr_t nr, fan::graphics::camera_nr_t camera_nr) {
        auto& camera = camera_get(camera_nr);
        fan_opengl_call(glUniformMatrix4fv(shader_get(nr).projection_view[0], 1, GL_FALSE, &camera.m_projection[0][0]));
        fan_opengl_call(glUniformMatrix4fv(shader_get(nr).projection_view[1], 1, GL_FALSE, &camera.m_view[0][0]));
      }


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

      fan::opengl::context_t::image_t& image_get(fan::graphics::image_nr_t nr) {
        return *(fan::opengl::context_t::image_t*)__fan_internal_image_list[nr].internal;
      }

      GLuint& image_get_handle(fan::graphics::image_nr_t nr) {
        return image_get(nr).texture_id;
      }

      fan::graphics::image_nr_t image_create() {
        fan::graphics::image_nr_t nr = __fan_internal_image_list.NewNode();
        __fan_internal_image_list[nr].internal = new fan::opengl::context_t::image_t;
        fan_opengl_call(glGenTextures(1, &image_get_handle(nr)));
        return nr;
      }

      void image_erase(fan::graphics::image_nr_t nr) {
        auto handle = image_get_handle(nr);
        fan_opengl_call(glDeleteTextures(1, (GLuint*)&handle));
        delete static_cast<fan::opengl::context_t::image_t*>(__fan_internal_image_list[nr].internal);
        __fan_internal_image_list.Recycle(nr);
      }

      void image_bind(fan::graphics::image_nr_t nr) {
        fan_opengl_call(glBindTexture(GL_TEXTURE_2D, image_get_handle(nr)));
      }

      void image_unbind(fan::graphics::image_nr_t nr) {
        fan_opengl_call(glBindTexture(GL_TEXTURE_2D, 0));
      }

      fan::graphics::image_load_properties_t& image_get_settings(fan::graphics::image_nr_t nr) {
        return __fan_internal_image_list[nr].image_settings;
      }

      void image_set_settings(fan::graphics::image_nr_t nr, const fan::opengl::context_t::image_load_properties_t& p) {
        image_bind(nr);
    #if fan_debug >= fan_debug_high
        if (p.visual_output < 0xff) {
          fan::throw_error("invalid format");
        }
        if (p.min_filter < 0xff) {
          fan::throw_error("invalid format");
        }
        if (p.mag_filter < 0xff) {
          fan::throw_error("invalid format");
        }
    #endif
        fan_opengl_call(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, p.visual_output));
        fan_opengl_call(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, p.visual_output));
        fan_opengl_call(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, p.min_filter));
        fan_opengl_call(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, p.mag_filter));

        __fan_internal_image_list[nr].image_settings = image_opengl_to_global(p);
      }

      fan::graphics::image_nr_t image_load(const fan::image::info_t& image_info, const fan::opengl::context_t::image_load_properties_t& lp) {

        auto p = lp;

        // If channels is specified in image_info but format is default
        if (image_info.channels > 0 && p.format == image_load_properties_defaults::format) {
          p.format = get_format_from_channels(image_info.channels);
        }
        else if (image_info.channels <= 0 && p.format != image_load_properties_defaults::format) {
          // Use the specified format
        }
        // Both specified - potential conflict
        else if (image_info.channels > 0 && p.format != image_load_properties_defaults::format) {
          // Check if there's a mismatch
          int format_channels = fan::graphics::get_channel_amount(opengl_to_global_format(p.format));
          if (format_channels != image_info.channels) {
            fan::print("Warning: Format/channels mismatch. Format specifies",
              format_channels, "channels but image_info specifies",
              image_info.channels, "channels. Using format specification.");
          }
        }

        fan::graphics::image_nr_t nr = image_create();
        image_bind(nr);
        image_set_settings(nr, p);

        auto& image = image_get(nr);
        auto& image_data = __fan_internal_image_list[nr];
        image_data.size = image_info.size;
        image_data.image_path = "";

        uint32_t bytes_per_row = (int)(image_data.size.x * fan::graphics::get_channel_amount(opengl_to_global_format(p.format)));
        if ((bytes_per_row % 8) == 0) {
          glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
        }
        else if ((bytes_per_row % 4) == 0) {
          glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        }
        else if ((bytes_per_row % 2) == 0) {
          glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
        }
        else {
          glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        }

        fan_opengl_call(glTexImage2D(GL_TEXTURE_2D, 0, p.internal_format, image_data.size.x, image_data.size.y, 0, p.format, p.type, image_info.data));

        switch (p.min_filter) {
        case GL_LINEAR_MIPMAP_LINEAR:
        case GL_NEAREST_MIPMAP_LINEAR:
        case GL_LINEAR_MIPMAP_NEAREST:
        case GL_NEAREST_MIPMAP_NEAREST: {
          break;
        }
        }
        fan_opengl_call(glGenerateMipmap(GL_TEXTURE_2D));

        return nr;
      }

      fan::graphics::image_nr_t create_missing_texture() {
        fan::opengl::context_t::image_load_properties_t p;

        p.visual_output = GL_REPEAT;

        fan::graphics::image_nr_t nr = image_create();
        image_bind(nr);

        image_set_settings(nr, p);
        auto& image = image_get(nr);
        auto& image_data = __fan_internal_image_list[nr];
        image_data.size = fan::vec2i(2, 2);

        fan_opengl_call(
          glTexImage2D(
            GL_TEXTURE_2D,
            0,
            p.internal_format,
            image_data.size.x,
            image_data.size.y,
            0,
            p.format,
            p.type,
            fan::image::missing_texture_pixels
          )
        );

        fan_opengl_call(glGenerateMipmap(GL_TEXTURE_2D));

        __fan_internal_image_list[nr].image_settings = image_opengl_to_global(p);

        return nr;
      }

      fan::graphics::image_nr_t create_transparent_texture(fan::opengl::context_t& context) {
        fan::opengl::context_t::image_load_properties_t p;

        p.visual_output = GL_REPEAT;
        p.min_filter = GL_NEAREST;
        p.mag_filter = GL_NEAREST;

        fan::graphics::image_nr_t nr = image_create();
        image_bind(nr);

        auto& img = image_get(nr);
        auto& image_data = __fan_internal_image_list[nr];

        image_set_settings(nr, p);

        image_data.size = fan::vec2i(2, 2);

        fan_opengl_call(glTexImage2D(GL_TEXTURE_2D, 0, p.internal_format, 2, 2, 0, p.format, p.type, fan::image::transparent_texture_pixels));

        fan_opengl_call(glGenerateMipmap(GL_TEXTURE_2D));
        return nr;
      }

      fan::graphics::image_nr_t image_load(const std::string& path, const fan::opengl::context_t::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current()) {

    #if fan_assert_if_same_path_loaded_multiple_times

        static std::unordered_map<std::string, bool> existing_images;

        if (existing_images.find(path) != existing_images.end()) {
          fan::throw_error("image already existing " + path);
        }

        existing_images[path] = 0;

    #endif

        fan::image::info_t image_info;
        if (fan::image::load(path, &image_info, callers_path)) {
          return create_missing_texture();
        }
        fan::graphics::image_nr_t nr = image_load(image_info, p);
        __fan_internal_image_list[nr].image_path = path;
        fan::image::free(&image_info);
        return nr;
      }

      fan::graphics::image_nr_t image_load(const fan::image::info_t& image_info) {
        return image_load(image_info, fan::opengl::context_t::image_load_properties_t());
      }

      fan::graphics::image_nr_t image_load(fan::color* colors, const fan::vec2ui& size_, const fan::opengl::context_t::image_load_properties_t& p) {

        fan::graphics::image_nr_t nr = image_create();
        image_bind(nr);

        image_set_settings(nr, p);

        auto& image_data = __fan_internal_image_list[nr];

        image_data.size = size_;

        fan_opengl_call(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, image_data.size.x, image_data.size.y, 0, p.format, GL_FLOAT, (uint8_t*)colors));

        return nr;
      }

      fan::graphics::image_nr_t image_load(fan::color* colors, const fan::vec2ui& size_) {
        return image_load(colors, size_, fan::opengl::context_t::image_load_properties_t());
      }

      fan::graphics::image_nr_t image_load(const std::string& path, const std::source_location& callers_path = std::source_location::current()) {
        return image_load(path, fan::opengl::context_t::image_load_properties_t(), callers_path);
      }

      void image_unload(fan::graphics::image_nr_t nr) {
        image_erase(nr);
      }

      void image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info, const fan::opengl::context_t::image_load_properties_t& lp) {

        auto p = lp;

        // If channels is specified in image_info but format is default
        if (image_info.channels > 0 && p.format == image_load_properties_defaults::format) {
          p.format = get_format_from_channels(image_info.channels);
        }
        else if (image_info.channels <= 0 && p.format != image_load_properties_defaults::format) {
          // Use the specified format
        }
        // Both specified - potential conflict
        else if (image_info.channels > 0 && p.format != image_load_properties_defaults::format) {
          // Check if there's a mismatch
          int format_channels = get_format_from_channels(p.format);
          if (format_channels != image_info.channels) {
            fan::print("Warning: Format/channels mismatch. Format specifies",
              format_channels, "channels but image_info specifies",
              image_info.channels, "channels. Using format specification.");
          }
        }

        image_bind(nr);

        image_set_settings(nr, p);

        uint32_t bytes_per_row = (int)(image_info.size.x * fan::graphics::get_channel_amount(opengl_to_global_format(p.format)));
        if ((bytes_per_row % 8) == 0) {
          glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
        }
        else if ((bytes_per_row % 4) == 0) {
          glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        }
        else if ((bytes_per_row % 2) == 0) {
          glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
        }
        else {
          glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        }
        
        auto& image_data = __fan_internal_image_list[nr];
        image_data.size = image_info.size;
        fan_opengl_call(glTexImage2D(GL_TEXTURE_2D, 0, p.internal_format, image_data.size.x, image_data.size.y, 0, p.format, p.type, image_info.data));

        fan_opengl_call(glGenerateMipmap(GL_TEXTURE_2D));
      }

      void image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info) {
        image_reload(nr, image_info, fan::opengl::context_t::image_load_properties_t());
      }

      void image_reload(fan::graphics::image_nr_t nr, const std::string& path, const fan::opengl::context_t::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current()) {
        fan::image::info_t image_info;
        if (fan::image::load(path, &image_info, callers_path)) {
          image_info.data = (void*)fan::image::missing_texture_pixels;
          image_info.size = 2;
          image_info.channels = 4;
          image_info.type = -1; // ignore free
        }
        image_reload(nr, image_info, p);
        __fan_internal_image_list[nr].image_path = path;
        fan::image::free(&image_info);
      }

      void image_reload(fan::graphics::image_nr_t nr, const std::string& path) {
        image_reload(nr, path, fan::opengl::context_t::image_load_properties_t());
      }

      std::vector<uint8_t> image_get_pixel_data(fan::graphics::image_nr_t nr, GLenum format, fan::vec2 uvp, fan::vec2 uvs) {
        auto& image = image_get(nr);
        image_bind(nr);
        auto& image_data = __fan_internal_image_list[nr];
        fan::vec2ui uv_size = {
            (uint32_t)(image_data.size.x * uvs.x),
            (uint32_t)(image_data.size.y * uvs.y)
        };
        std::vector<uint8_t> full_data(image_data.size.x * image_data.size.y * 4); // assuming rgba
        fan_opengl_call(glGetTexImage(GL_TEXTURE_2D,
          0,
          format,
          GL_UNSIGNED_BYTE,
          full_data.data())
        );
        std::vector<uint8_t> result_data(uv_size.x * uv_size.y * 4); // assuming rgba
        for (uint32_t y = 0; y < uv_size.y; ++y) {
          for (uint32_t x = 0; x < uv_size.x; ++x) {
            uint32_t full_index = ((y + uvp.y * image_data.size.y) * image_data.size.x + (x + uvp.x * image_data.size.x)) * 4;
            uint32_t index = (y * uv_size.x + x) * 4;
            result_data[index + 0] = full_data[full_index + 0];
            result_data[index + 1] = full_data[full_index + 1];
            result_data[index + 2] = full_data[full_index + 2];
            result_data[index + 3] = full_data[full_index + 3];
          }
        }
        return result_data;
      }

      // creates single colored text size.x*size.y sized
      fan::graphics::image_nr_t image_create(const fan::color& color, const fan::opengl::context_t::image_load_properties_t& p) {

        uint8_t pixels[4];
        for (uint32_t p = 0; p < fan::color::size(); p++) {
          pixels[p] = color[p] * 255;
        }

        fan::graphics::image_nr_t nr = image_create();
        image_bind(nr);

        image_set_settings(nr, p);

        fan_opengl_call(glTexImage2D(GL_TEXTURE_2D, 0, p.internal_format, 1, 1, 0, p.format, p.type, pixels));

        auto& image_data = __fan_internal_image_list[nr];
        image_data.size = 1;

        fan_opengl_call(glGenerateMipmap(GL_TEXTURE_2D));

        return nr;
      }

      fan::graphics::image_nr_t image_create(const fan::color& color) {
        return image_create(color, fan::opengl::context_t::image_load_properties_t());
      }

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


      fan::graphics::camera_nr_t camera_create() {
        return __fan_internal_camera_list.NewNode();
      }

      void camera_erase(fan::graphics::camera_nr_t nr) {
        __fan_internal_camera_list.Recycle(nr);
      }

      void camera_set_ortho(fan::graphics::camera_nr_t nr, fan::vec2 x, fan::vec2 y) {
        auto& camera = camera_get(nr);

        camera.coordinates.left = x.x;
        camera.coordinates.right = x.y;
        camera.coordinates.down = y.y;
        camera.coordinates.up = y.x;

        camera.m_projection = fan::math::ortho<fan::mat4>(
          camera.coordinates.left,
          camera.coordinates.right,
          camera.coordinates.down,
          camera.coordinates.up,
          0.1,
          fan::graphics::znearfar
        );

        camera.m_view[3][0] = 0;
        camera.m_view[3][1] = 0;
        camera.m_view[3][2] = 0;
        camera.m_view = camera.m_view.translate(camera.position);
        fan::vec3 position = camera.m_view.get_translation();
        constexpr fan::vec3 front(0, 0, 1);

        camera.m_view = fan::math::look_at_left<fan::mat4, fan::vec3>(position, position + front, fan::camera::world_up);
      }

      fan::graphics::camera_nr_t camera_create(const fan::vec2& x, const fan::vec2& y) {
        fan::graphics::camera_nr_t nr = camera_create();
        camera_set_ortho(nr, fan::vec2(x.x, x.y), fan::vec2(y.x, y.y));
        return nr;
      }

      fan::vec3 camera_get_position(fan::graphics::camera_nr_t nr) {
        return camera_get(nr).position;
      }

      void camera_set_position(fan::graphics::camera_nr_t nr, const fan::vec3& cp) {
        auto& camera = camera_get(nr);
        camera.position = cp;

        camera.m_view[3][0] = 0;
        camera.m_view[3][1] = 0;
        camera.m_view[3][2] = 0;
        camera.m_view = camera.m_view.translate(camera.position);
        fan::vec3 position = camera.m_view.get_translation();
        constexpr fan::vec3 front(0, 0, 1);

        camera.m_view = fan::math::look_at_left<fan::mat4, fan::vec3>(position, position + front, fan::camera::world_up);
      }

      fan::vec2 camera_get_size(fan::graphics::camera_nr_t nr) {
        auto& camera = camera_get(nr);
        return fan::vec2(std::abs(camera.coordinates.right - camera.coordinates.left), std::abs(camera.coordinates.down - camera.coordinates.up));
      }

      void camera_set_perspective(fan::graphics::camera_nr_t nr, f32_t fov, const fan::vec2& window_size) {
        auto& camera = camera_get(nr);

        camera.m_projection = fan::math::perspective<fan::mat4>(fan::math::radians(fov), (f32_t)window_size.x / (f32_t)window_size.y, camera.znear, camera.zfar);

        camera.update_view();

        camera.m_view = camera.get_view_matrix();
      }

      void camera_rotate(fan::graphics::camera_nr_t nr, const fan::vec2& offset) {
        auto& camera = camera_get(nr);
        camera.rotate_camera(offset);
        camera.m_view = camera.get_view_matrix();
      }

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


      //-----------------------------viewport-----------------------------
      //-----------------------------viewport-----------------------------
      //-----------------------------viewport-----------------------------
      //-----------------------------viewport-----------------------------
      //-----------------------------viewport-----------------------------
      //-----------------------------viewport-----------------------------
      //-----------------------------viewport-----------------------------
      //-----------------------------viewport-----------------------------
      //-----------------------------viewport-----------------------------
      //-----------------------------viewport-----------------------------
      //-----------------------------viewport-----------------------------
      //-----------------------------viewport-----------------------------

      void viewport_set(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
        fan_opengl_call(glViewport(viewport_position_.x, window_size.y - viewport_size_.y - viewport_position_.y,
          viewport_size_.x, viewport_size_.y
        ));
      }

      fan::graphics::context_viewport_t& viewport_get(fan::graphics::viewport_nr_t nr) {
        return __fan_internal_viewport_list[nr];
      }

      void viewport_set(fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
        auto& viewport = viewport_get(nr);
        viewport.viewport_position = viewport_position_;
        viewport.viewport_size = viewport_size_;

        viewport_set(viewport_position_, viewport_size_, window_size);
      }

      fan::graphics::viewport_nr_t viewport_create() {
        auto nr = __fan_internal_viewport_list.NewNode();

        viewport_set(
          nr,
          0, 0, 0
        );
        return nr;
      }

      void viewport_erase(fan::graphics::viewport_nr_t nr) {
        __fan_internal_viewport_list.Recycle(nr);
      }

      fan::vec2 viewport_get_position(fan::graphics::viewport_nr_t nr) {
        return viewport_get(nr).viewport_position;
      }

      fan::vec2 viewport_get_size(fan::graphics::viewport_nr_t nr) {
        return viewport_get(nr).viewport_size;
      }


      void viewport_zero(fan::graphics::viewport_nr_t nr) {
        auto& viewport = viewport_get(nr);
        viewport.viewport_position = 0;
        viewport.viewport_size = 0;
        fan_opengl_call(glViewport(0, 0, 0, 0));
      }

      bool viewport_inside(fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
        auto& viewport = viewport_get(nr);
        return fan_2d::collision::rectangle::point_inside_no_rotation(position, viewport.viewport_position + viewport.viewport_size / 2, viewport.viewport_size / 2);
      }

      bool viewport_inside_wir(fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
        auto& viewport = viewport_get(nr);
        return fan_2d::collision::rectangle::point_inside_no_rotation(position, viewport.viewport_size / 2, viewport.viewport_size / 2);
      }


      //-----------------------------viewport-----------------------------
      //-----------------------------viewport-----------------------------
      //-----------------------------viewport-----------------------------
      //-----------------------------viewport-----------------------------
      //-----------------------------viewport-----------------------------
      //-----------------------------viewport-----------------------------
      //-----------------------------viewport-----------------------------
      //-----------------------------viewport-----------------------------
      //-----------------------------viewport-----------------------------
      //-----------------------------viewport-----------------------------
      //-----------------------------viewport-----------------------------
      //-----------------------------viewport-----------------------------


      static uint32_t global_to_opengl_format(uintptr_t format) {
        if (format == fan::graphics::image_format::b8g8r8a8_unorm) return GL_BGRA;
        if (format == fan::graphics::image_format::r8b8g8a8_unorm) return GL_RGBA;
        if (format == fan::graphics::image_format::r8_unorm) return GL_RED;
        if (format == fan::graphics::image_format::rg8_unorm) return GL_RG;
        if (format == fan::graphics::image_format::rgb_unorm) return GL_RGB;
        if (format == fan::graphics::image_format::rgba_unorm) return GL_RGBA;
        if (format == fan::graphics::image_format::r8_uint) return GL_RED_INTEGER;
        if (format == fan::graphics::image_format::r8g8b8a8_srgb) return GL_SRGB8_ALPHA8;
        if (format == fan::graphics::image_format::r11f_g11f_b10f) return GL_R11F_G11F_B10F;
    #if fan_debug >= fan_debug_high
        fan::throw_error("invalid format");
    #endif
        return GL_RGBA;
      }

      static uint32_t global_to_opengl_type(uintptr_t type) {
        if (type == fan::graphics::fan_unsigned_byte) return GL_UNSIGNED_BYTE;
        if (type == fan::graphics::fan_unsigned_int) return GL_UNSIGNED_INT;
        if (type == fan::graphics::fan_float) return GL_FLOAT;
    #if fan_debug >= fan_debug_high
        fan::throw_error("invalid format");
    #endif
        return 0;
      }

      static uint32_t global_to_opengl_address_mode(uint32_t mode) {
        if (mode == fan::graphics::image_sampler_address_mode::repeat) return GL_REPEAT;
        if (mode == fan::graphics::image_sampler_address_mode::mirrored_repeat) return GL_MIRRORED_REPEAT;
        if (mode == fan::graphics::image_sampler_address_mode::clamp_to_edge) return GL_CLAMP_TO_EDGE;
        if (mode == fan::graphics::image_sampler_address_mode::clamp_to_border) return GL_CLAMP_TO_BORDER;
        if (mode == fan::graphics::image_sampler_address_mode::mirrored_clamp_to_edge) return GL_MIRROR_CLAMP_TO_EDGE;
    #if fan_debug >= fan_debug_high
        fan::throw_error("invalid format");
    #endif
        return GL_REPEAT;
      }

      static uint32_t global_to_opengl_filter(uintptr_t filter) {
        using filter_t = fan::graphics::image_filter;

        switch (filter) {
        case filter_t::nearest: return GL_NEAREST;
        case filter_t::linear: return GL_LINEAR;
        case filter_t::nearest_mipmap_nearest: return GL_NEAREST_MIPMAP_NEAREST;
        case filter_t::linear_mipmap_nearest: return GL_LINEAR_MIPMAP_NEAREST;
        case filter_t::nearest_mipmap_linear: return GL_NEAREST_MIPMAP_LINEAR;
        case filter_t::linear_mipmap_linear: return GL_LINEAR_MIPMAP_LINEAR;
        default:
#if fan_debug >= fan_debug_high
          fan::throw_error("Invalid image filter value");
#endif
          return GL_NEAREST;
        }
      }


      static uint32_t opengl_to_global_format(uintptr_t format) {
        if (format == GL_BGRA) return fan::graphics::image_format::b8g8r8a8_unorm;
        if (format == GL_RGBA) return fan::graphics::image_format::r8b8g8a8_unorm;
        if (format == GL_RED) return fan::graphics::image_format::r8_unorm;
        if (format == GL_RG) return fan::graphics::image_format::rg8_unorm;
        if (format == GL_RGB) return fan::graphics::image_format::rgb_unorm;
        if (format == GL_RED_INTEGER) return fan::graphics::image_format::r8_uint;
        if (format == GL_SRGB8_ALPHA8) return fan::graphics::image_format::r8g8b8a8_srgb;
        if (format == GL_R11F_G11F_B10F) return fan::graphics::image_format::r11f_g11f_b10f;
    #if fan_debug >= fan_debug_high
        fan::throw_error("invalid format");
    #endif
        return fan::graphics::image_format::rgba_unorm;
      }

      static uint32_t opengl_to_global_type(uintptr_t type) {
        if (type == GL_UNSIGNED_BYTE) return fan::graphics::fan_unsigned_byte;
        if (type == GL_UNSIGNED_INT) return fan::graphics::fan_unsigned_int;
        if (type == GL_FLOAT) return fan::graphics::fan_float;
    #if fan_debug >= fan_debug_high
        fan::throw_error("invalid format");
    #endif
        return 0;
      }

      static uint32_t opengl_to_global_address_mode(uint32_t mode) {
        if (mode == GL_REPEAT) return fan::graphics::image_sampler_address_mode::repeat;
        if (mode == GL_MIRRORED_REPEAT) return fan::graphics::image_sampler_address_mode::mirrored_repeat;
        if (mode == GL_CLAMP_TO_EDGE) return fan::graphics::image_sampler_address_mode::clamp_to_edge;
        if (mode == GL_CLAMP_TO_BORDER) return fan::graphics::image_sampler_address_mode::clamp_to_border;
        if (mode == GL_MIRROR_CLAMP_TO_EDGE) return fan::graphics::image_sampler_address_mode::mirrored_clamp_to_edge;
    #if fan_debug >= fan_debug_high
        fan::throw_error("invalid format");
    #endif
        return fan::graphics::image_sampler_address_mode::repeat;
      }

      static uint32_t opengl_to_global_filter(uintptr_t filter) {
        using namespace fan::graphics;

        if (filter == GL_NEAREST) return fan::graphics::image_filter::nearest;
        if (filter == GL_LINEAR) return fan::graphics::image_filter::linear;
        if (filter == GL_NEAREST_MIPMAP_NEAREST) return fan::graphics::image_filter::nearest_mipmap_nearest;
        if (filter == GL_LINEAR_MIPMAP_NEAREST) return fan::graphics::image_filter::linear_mipmap_nearest;
        if (filter == GL_NEAREST_MIPMAP_LINEAR) return fan::graphics::image_filter::nearest_mipmap_linear;
        if (filter == GL_LINEAR_MIPMAP_LINEAR) return fan::graphics::image_filter::linear_mipmap_linear;

#if fan_debug >= fan_debug_high
        fan::throw_error("Invalid OpenGL filter value.");
#endif

        return fan::graphics::image_filter::nearest;
      }


      void close(fan::opengl::context_t& context) {
        {
          fan::graphics::camera_list_t::nrtra_t nrtra;
          fan::graphics::camera_nr_t nr;
          nrtra.Open(&__fan_internal_camera_list, &nr);
          while (nrtra.Loop(&__fan_internal_camera_list, &nr)) {
            camera_erase(nr);
          }
          nrtra.Close(&__fan_internal_camera_list);
        }
        {
          fan::graphics::shader_list_t::nrtra_t nrtra;
          fan::graphics::shader_nr_t nr;
          nrtra.Open(&__fan_internal_shader_list, &nr);
          while (nrtra.Loop(&__fan_internal_shader_list, &nr)) {
            shader_erase(nr);
          }
          nrtra.Close(&__fan_internal_shader_list);
        }
        {
          fan::graphics::image_list_t::nrtra_t nrtra;
          fan::graphics::image_nr_t nr;
          nrtra.Open(&__fan_internal_image_list, &nr);
          while (nrtra.Loop(&__fan_internal_image_list, &nr)) {
            image_erase(nr);
          }
          nrtra.Close(&__fan_internal_image_list);
        }
        {
          fan::graphics::viewport_list_t::nrtra_t nrtra;
          fan::graphics::viewport_nr_t nr;
          nrtra.Open(&__fan_internal_viewport_list, &nr);
          while (nrtra.Loop(&__fan_internal_viewport_list, &nr)) {
            viewport_erase(nr);
          }
          nrtra.Close(&__fan_internal_viewport_list);
        }
      }
      /*
      struct image_load_properties_t {
              uint32_t            visual_output = image_load_properties_defaults::visual_output;
              uintptr_t           internal_format = image_load_properties_defaults::internal_format;
              uintptr_t           format = image_load_properties_defaults::format;
              uintptr_t           type = image_load_properties_defaults::type;
              uintptr_t           min_filter = image_load_properties_defaults::min_filter;
              uintptr_t           mag_filter = image_load_properties_defaults::mag_filter;
            };
      */
      static fan::opengl::context_t::image_load_properties_t image_global_to_opengl(const fan::graphics::image_load_properties_t& p) {
        return {
          .visual_output = global_to_opengl_address_mode(p.visual_output),
          .internal_format = global_to_opengl_format(p.internal_format),
          .format = global_to_opengl_format(p.format),
          .type = global_to_opengl_type(p.type),
          .min_filter = global_to_opengl_filter(p.min_filter),
          .mag_filter = global_to_opengl_filter(p.mag_filter),
        };
      }

      static fan::graphics::image_load_properties_t image_opengl_to_global(const fan::opengl::context_t::image_load_properties_t& p) {
        return fan::graphics::image_load_properties_t{
          .visual_output = opengl_to_global_address_mode(p.visual_output),
          .internal_format = opengl_to_global_format(p.internal_format),
          .format = opengl_to_global_format(p.format),
          .type = opengl_to_global_type(p.type),
          .min_filter = opengl_to_global_filter(p.min_filter),
          .mag_filter = opengl_to_global_filter(p.mag_filter),
        };
      }
    };
  }
}

template void fan::opengl::context_t::shader_set_value<fan::vec2>(fan::graphics::shader_nr_t nr, const std::string& name, const fan::vec2& val);
template void fan::opengl::context_t::shader_set_value<fan::vec3>(fan::graphics::shader_nr_t nr, const std::string& name, const fan::vec3& val);
template void fan::opengl::context_t::shader_set_value<fan::vec4>(fan::graphics::shader_nr_t nr, const std::string& name, const fan::vec4& val);
template void fan::opengl::context_t::shader_set_value<fan::mat4>(fan::graphics::shader_nr_t nr, const std::string& name, const fan::mat4& val);
template void fan::opengl::context_t::shader_set_value<fan::color>(fan::graphics::shader_nr_t nr, const std::string& name, const fan::color& val);
template void fan::opengl::context_t::shader_set_value<uint32_t>(fan::graphics::shader_nr_t nr, const std::string& name, const uint32_t& val);
template void fan::opengl::context_t::shader_set_value<uint64_t>(fan::graphics::shader_nr_t nr, const std::string& name, const uint64_t& val);
template void fan::opengl::context_t::shader_set_value<int>(fan::graphics::shader_nr_t nr, const std::string& name, const int& val);
template void fan::opengl::context_t::shader_set_value<f32_t>(fan::graphics::shader_nr_t nr, const std::string& name, const f32_t& val);
template void fan::opengl::context_t::shader_set_value<fan::vec1_wrap_t<f32_t>>(fan::graphics::shader_nr_t nr, const std::string& name, const fan::vec1_wrap_t<f32_t>& val);
template void fan::opengl::context_t::shader_set_value<fan::vec_wrap_t<1, f32_t>>(fan::graphics::shader_nr_t nr, const std::string& name, const fan::vec_wrap_t<1, f32_t>& val);
template void fan::opengl::context_t::shader_set_value<fan::vec_wrap_t<2, f32_t>>(fan::graphics::shader_nr_t nr, const std::string& name, const fan::vec_wrap_t<2, f32_t>& val);

export namespace fan {
  namespace opengl {
    namespace core {
      int get_buffer_size(fan::opengl::context_t& context, GLenum target_buffer, GLuint buffer_object) {
        int size = 0;

        fan_opengl_call(glBindBuffer(target_buffer, buffer_object));
        fan_opengl_call(glGetBufferParameteriv(target_buffer, GL_BUFFER_SIZE, &size));

        return size;
      }

      void write_glbuffer(fan::opengl::context_t& context, GLuint buffer, const void* data, uintptr_t size, uint32_t usage, GLenum target) {
        fan_opengl_call(glBindBuffer(target, buffer));

        fan_opengl_call(glBufferData(target, size, data, usage));
        /*if (target == GL_SHADER_STORAGE_BUFFER) {
        glBindBufferBase(target, location, buffer);
        }*/
      }
      void get_glbuffer(fan::opengl::context_t& context, void* data, GLuint buffer_id, uintptr_t size, uintptr_t offset, GLenum target) {
        fan_opengl_call(glBindBuffer(target, buffer_id));
        fan_opengl_call(glGetBufferSubData(target, offset, size, data));
      }

      void edit_glbuffer(fan::opengl::context_t& context, GLuint buffer, const void* data, uintptr_t offset, uintptr_t size, uintptr_t target) {
        fan_opengl_call(glBindBuffer(target, buffer));

        #if fan_debug >= fan_debug_high

        int buffer_size = get_buffer_size(context, target, buffer);

        if ((buffer_size < (int)size) || (int)(offset + size) > buffer_size) {
          fan::throw_error("tried to write more than allocated");
        }

        #endif

        fan_opengl_call(glBufferSubData(target, offset, size, data));
        /* if (target == GL_SHADER_STORAGE_BUFFER) {
        glBindBufferBase(target, location, buffer);
        }*/
      }

      // not tested
      int get_bound_buffer(fan::opengl::context_t& context) {
        int buffer_id;
        fan_opengl_call(glGetIntegerv(GL_VERTEX_BINDING_BUFFER, &buffer_id));
        return buffer_id;
      }
      //#pragma pack(push, 1)
      //#pragma pack(pop)
      struct vao_t {

        void open(fan::opengl::context_t& context) {
          fan_opengl_call(glGenVertexArrays(1, &m_buffer));
        }
        void close(fan::opengl::context_t& context) {
          if (m_buffer == (decltype(m_buffer))-1) {
            return;
          }
          fan_opengl_call(glDeleteVertexArrays(1, &m_buffer));
        }

        void bind(fan::opengl::context_t& context) const {
          fan_opengl_call(glBindVertexArray(m_buffer));
        }
        void unbind(fan::opengl::context_t& context) const {
          fan_opengl_call(glBindVertexArray(0));
        }

        GLuint m_buffer = (GLuint)-1;
      };

      struct vbo_t {

        void open(fan::opengl::context_t& context, GLenum target_) {
          fan_opengl_call(glGenBuffers(1, &m_buffer));
          m_target = target_;
        }

        void close(fan::opengl::context_t& context) {
#if fan_debug >= fan_debug_medium
          if (m_buffer == (GLuint)-1) {
            return;
            //fan::throw_error("tried to remove non existent vbo");
          }
#endif
          fan_opengl_call(glDeleteBuffers(1, &m_buffer));
        }

        void bind(fan::opengl::context_t& context) const {
          fan_opengl_call(glBindBuffer(m_target, m_buffer));
        }

        void get_vram_instance(fan::opengl::context_t& context, void* data, uintptr_t size, uintptr_t offset) {
          fan::opengl::core::get_glbuffer(context, data, m_buffer, size, offset, m_target);
        }

        // only for target GL_UNIFORM_BUFFER
        void bind_buffer_range(fan::opengl::context_t& context, uint32_t total_size) {
          fan_opengl_call(glBindBufferRange(GL_UNIFORM_BUFFER, 0, m_buffer, 0, total_size));
        }

        void edit_buffer(fan::opengl::context_t& context, const void* data, uintptr_t offset, uintptr_t size) {
          fan::opengl::core::edit_glbuffer(context, m_buffer, data, offset, size, m_target);
        }

        void write_buffer(fan::opengl::context_t& context, const void* data, uintptr_t size) {
          fan::opengl::core::write_glbuffer(context, m_buffer, data, size, m_usage, m_target);
        }

        GLuint m_buffer = (GLuint)-1;
        GLenum m_target = (GLuint)-1;
        uint32_t m_usage = GL_DYNAMIC_DRAW;
      };

      struct framebuffer_t {

        struct properties_t {
          properties_t() {}
          GLenum internalformat = GL_DEPTH_STENCIL_ATTACHMENT;
        };

        void open(fan::opengl::context_t& context) {
          fan_opengl_call(glGenFramebuffers(1, &framebuffer));
        }

        void close(fan::opengl::context_t& context) {
          fan_opengl_call(glDeleteFramebuffers(1, &framebuffer));
        }

        void bind(fan::opengl::context_t& context) const {
          fan_opengl_call(glBindFramebuffer(GL_FRAMEBUFFER, framebuffer));
        }

        void unbind(fan::opengl::context_t& context) const {
          fan_opengl_call(glBindFramebuffer(GL_FRAMEBUFFER, 0));
        }

        bool ready(fan::opengl::context_t& context) const {
          return fan_opengl_call(glCheckFramebufferStatus(GL_FRAMEBUFFER)) == GL_FRAMEBUFFER_COMPLETE;
        }

        void bind_to_renderbuffer(fan::opengl::context_t& context, GLenum renderbuffer, const properties_t& p = properties_t()) {
          bind(context);
          fan_opengl_call(glFramebufferRenderbuffer(GL_FRAMEBUFFER, p.internalformat, GL_RENDERBUFFER, renderbuffer));
        }

        // texture must be binded with texture.bind();
        static void bind_to_texture(fan::opengl::context_t& context, GLuint texture, GLenum attatchment) {
          fan_opengl_call(glFramebufferTexture2D(GL_FRAMEBUFFER, attatchment, GL_TEXTURE_2D, texture, 0));
        }

        GLuint framebuffer;
      };

      struct renderbuffer_t {

        struct properties_t {
          properties_t() {}
          GLenum internalformat = GL_DEPTH24_STENCIL8;
          fan::vec2ui size;
        };

        void open(fan::opengl::context_t& context) {
          fan_opengl_call(glGenRenderbuffers(1, &renderbuffer));
          //set_storage(context, p);
        }

        void close(fan::opengl::context_t& context) {
          fan_opengl_call(glDeleteRenderbuffers(1, &renderbuffer));
        }

        void bind(fan::opengl::context_t& context) const {
          fan_opengl_call(glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer));
        }

        void set_storage(fan::opengl::context_t& context, const properties_t& p) const {
          bind(context);
          fan_opengl_call(glRenderbufferStorage(GL_RENDERBUFFER, p.internalformat, p.size.x, p.size.y));
        }

        void bind_to_renderbuffer(fan::opengl::context_t& context, const properties_t& p = properties_t()) {
          bind(context);
          fan_opengl_call(glFramebufferRenderbuffer(GL_FRAMEBUFFER, p.internalformat, GL_RENDERBUFFER, renderbuffer));
        }

        GLuint renderbuffer;
      };

      uint32_t get_draw_mode(uint8_t draw_mode) {
        switch (draw_mode) {
        case fan::graphics::primitive_topology_t::points:
          return fan::opengl::context_t::primitive_topology_t::points;
        case fan::graphics::primitive_topology_t::lines:
          return fan::opengl::context_t::primitive_topology_t::lines;
        case fan::graphics::primitive_topology_t::line_strip:
          return fan::opengl::context_t::primitive_topology_t::line_strip;
        case fan::graphics::primitive_topology_t::line_loop:
          return fan::opengl::context_t::primitive_topology_t::line_loop;
        case fan::graphics::primitive_topology_t::triangles:
          return fan::opengl::context_t::primitive_topology_t::triangles;
        case fan::graphics::primitive_topology_t::triangle_strip:
          return fan::opengl::context_t::primitive_topology_t::triangle_strip;
        case fan::graphics::primitive_topology_t::triangle_fan:
          return fan::opengl::context_t::primitive_topology_t::triangle_fan;
        case fan::graphics::primitive_topology_t::lines_with_adjacency:
          return fan::opengl::context_t::primitive_topology_t::lines_with_adjacency;
        case fan::graphics::primitive_topology_t::line_strip_with_adjacency:
          return fan::opengl::context_t::primitive_topology_t::line_strip_with_adjacency;
        case fan::graphics::primitive_topology_t::triangles_with_adjacency:
          return fan::opengl::context_t::primitive_topology_t::triangles_with_adjacency;
        case fan::graphics::primitive_topology_t::triangle_strip_with_adjacency:
          return fan::opengl::context_t::primitive_topology_t::triangle_strip_with_adjacency;
        default:
          fan::throw_error("invalid draw mode");
          return -1;
        }
      }
    }
  }
}

namespace fan {
  namespace graphics {
    export fan::graphics::context_functions_t get_gl_context_functions() {
      fan::graphics::context_functions_t cf;
      cf.shader_create = [](void* context) {
        return ((fan::opengl::context_t*)context)->shader_create();
        };
      cf.shader_get = [](void* context, fan::graphics::shader_nr_t nr) {
        return (void*)&((fan::opengl::context_t*)context)->shader_get(nr);
        };
      cf.shader_erase = [](void* context, fan::graphics::shader_nr_t nr) {
        ((fan::opengl::context_t*)context)->shader_erase(nr);
        };
      cf.shader_use = [](void* context, fan::graphics::shader_nr_t nr) {
        ((fan::opengl::context_t*)context)->shader_use(nr);
        };
      cf.shader_set_vertex = [](void* context, fan::graphics::shader_nr_t nr, const std::string& vertex_code) {
        ((fan::opengl::context_t*)context)->shader_set_vertex(nr, vertex_code);
        };
      cf.shader_set_fragment = [](void* context, fan::graphics::shader_nr_t nr, const std::string& fragment_code) {
        ((fan::opengl::context_t*)context)->shader_set_fragment(nr, fragment_code);
        };
      cf.shader_compile = [](void* context, fan::graphics::shader_nr_t nr) {
        return ((fan::opengl::context_t*)context)->shader_compile(nr);
        };
      /*image*/
      cf.image_create = [](void* context) {
        return ((fan::opengl::context_t*)context)->image_create();
        };
      cf.image_get_handle = [](void* context, fan::graphics::image_nr_t nr) {
        return (uint64_t)((fan::opengl::context_t*)context)->image_get_handle(nr);
        };
      cf.image_get = [](void* context, fan::graphics::image_nr_t nr) {
        return (void*)&((fan::opengl::context_t*)context)->image_get(nr);
        };
      cf.image_erase = [](void* context, fan::graphics::image_nr_t nr) {
        ((fan::opengl::context_t*)context)->image_erase(nr);
        };
      cf.image_bind = [](void* context, fan::graphics::image_nr_t nr) {
        ((fan::opengl::context_t*)context)->image_bind(nr);
        };
      cf.image_unbind = [](void* context, fan::graphics::image_nr_t nr) {
        ((fan::opengl::context_t*)context)->image_unbind(nr);
        };
      cf.image_get_settings = [](void* context, fan::graphics::image_nr_t nr) -> fan::graphics::image_load_properties_t& {
        return ((fan::opengl::context_t*)context)->image_get_settings(nr);
        };
      cf.image_set_settings = [](void* context, fan::graphics::image_nr_t nr, const fan::graphics::image_load_properties_t& settings) {
        ((fan::opengl::context_t*)context)->image_set_settings(nr, ((fan::opengl::context_t*)context)->image_global_to_opengl(settings));
        };
      cf.image_load_info = [](void* context, const fan::image::info_t& image_info) {
        return ((fan::opengl::context_t*)context)->image_load(image_info);
        };
      cf.image_load_info_props = [](void* context, const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p) {
        return ((fan::opengl::context_t*)context)->image_load(image_info, ((fan::opengl::context_t*)context)->image_global_to_opengl(p));
        };
      cf.image_load_path = [](void* context, const std::string& path, const std::source_location& callers_path = std::source_location::current()) {
        return ((fan::opengl::context_t*)context)->image_load(path, callers_path);
        };
      cf.image_load_path_props = [](void* context, const std::string& path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current()) {
        return ((fan::opengl::context_t*)context)->image_load(path, ((fan::opengl::context_t*)context)->image_global_to_opengl(p));
        };
      cf.image_load_colors = [](void* context, fan::color* colors, const fan::vec2ui& size_) {
        return ((fan::opengl::context_t*)context)->image_load(colors, size_);
        };
      cf.image_load_colors_props = [](void* context, fan::color* colors, const fan::vec2ui& size_, const fan::graphics::image_load_properties_t& p) {
        return ((fan::opengl::context_t*)context)->image_load(colors, size_, ((fan::opengl::context_t*)context)->image_global_to_opengl(p));
        };
      cf.image_unload = [](void* context, fan::graphics::image_nr_t nr) {
        ((fan::opengl::context_t*)context)->image_unload(nr);
        };
      cf.create_missing_texture = [](void* context) {
        return ((fan::opengl::context_t*)context)->create_missing_texture();
        };
      cf.create_transparent_texture = [](void* context) {
        return ((fan::opengl::context_t*)context)->create_transparent_texture(*(fan::opengl::context_t*)context);
        };
      cf.image_reload_image_info = [](void* context, fan::graphics::image_nr_t nr, const fan::image::info_t& image_info) {
        return ((fan::opengl::context_t*)context)->image_reload(nr, image_info);
        };
      cf.image_reload_image_info_props = [](void* context, fan::graphics::image_nr_t nr, const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p) {
        return ((fan::opengl::context_t*)context)->image_reload(nr, image_info, ((fan::opengl::context_t*)context)->image_global_to_opengl(p));
        };
      cf.image_reload_path = [](void* context, fan::graphics::image_nr_t nr, const std::string& path, const std::source_location& callers_path = std::source_location::current()) {
        return ((fan::opengl::context_t*)context)->image_reload(nr, path);
        };
      cf.image_reload_path_props = [](void* context, fan::graphics::image_nr_t nr, const std::string& path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current()) {
        return ((fan::opengl::context_t*)context)->image_reload(nr, path, ((fan::opengl::context_t*)context)->image_global_to_opengl(p));
        };
      cf.image_create_color = [](void* context, const fan::color& color) {
        return ((fan::opengl::context_t*)context)->image_create(color);
        };
      cf.image_create_color_props = [](void* context, const fan::color& color, const fan::graphics::image_load_properties_t& p) {
        return ((fan::opengl::context_t*)context)->image_create(color, ((fan::opengl::context_t*)context)->image_global_to_opengl(p));
        };
      /*camera*/
      cf.camera_create = [](void* context) {
        return ((fan::opengl::context_t*)context)->camera_create();
        };
      cf.camera_get = [](void* context, fan::graphics::camera_nr_t nr) -> decltype(auto) {
        return ((fan::opengl::context_t*)context)->camera_get(nr);
        };
      cf.camera_erase = [](void* context, fan::graphics::camera_nr_t nr) {
        ((fan::opengl::context_t*)context)->camera_erase(nr);
        };
      cf.camera_create_params = [](void* context, const fan::vec2& x, const fan::vec2& y) {
        return ((fan::opengl::context_t*)context)->camera_create(x, y);
        };
      cf.camera_get_position = [](void* context, fan::graphics::camera_nr_t nr) {
        return ((fan::opengl::context_t*)context)->camera_get_position(nr);
        };
      cf.camera_set_position = [](void* context, fan::graphics::camera_nr_t nr, const fan::vec3& cp) {
        ((fan::opengl::context_t*)context)->camera_set_position(nr, cp);
        };
      cf.camera_get_size = [](void* context, fan::graphics::camera_nr_t nr) {
        return ((fan::opengl::context_t*)context)->camera_get_size(nr);
        };
      cf.camera_set_ortho = [](void* context, fan::graphics::camera_nr_t nr, fan::vec2 x, fan::vec2 y) {
        ((fan::opengl::context_t*)context)->camera_set_ortho(nr, x, y);
        };
      cf.camera_set_perspective = [](void* context, fan::graphics::camera_nr_t nr, f32_t fov, const fan::vec2& window_size) {
        ((fan::opengl::context_t*)context)->camera_set_perspective(nr, fov, window_size);
        };
      cf.camera_rotate = [](void* context, fan::graphics::camera_nr_t nr, const fan::vec2& offset) {
        ((fan::opengl::context_t*)context)->camera_rotate(nr, offset);
        };
      /*viewport*/
      cf.viewport_create = [](void* context) {
        return ((fan::opengl::context_t*)context)->viewport_create();
        };
      cf.viewport_create_params = [](void* context, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
        auto gl_context = ((fan::opengl::context_t*)context);
        auto nr = gl_context->viewport_create();
        gl_context->viewport_set(nr, viewport_position_, viewport_size_, window_size);
        return nr;
      };
      cf.viewport_get = [](void* context, fan::graphics::viewport_nr_t nr) -> fan::graphics::context_viewport_t& {
        return ((fan::opengl::context_t*)context)->viewport_get(nr);
        };
      cf.viewport_erase = [](void* context, fan::graphics::viewport_nr_t nr) {
        ((fan::opengl::context_t*)context)->viewport_erase(nr);
        };
      cf.viewport_get_position = [](void* context, fan::graphics::viewport_nr_t nr) {
        return ((fan::opengl::context_t*)context)->viewport_get_position(nr);
        };
      cf.viewport_get_size = [](void* context, fan::graphics::viewport_nr_t nr) {
        return ((fan::opengl::context_t*)context)->viewport_get_size(nr);
        };
      cf.viewport_set = [](void* context, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
        ((fan::opengl::context_t*)context)->viewport_set(viewport_position_, viewport_size_, window_size);
        };
      cf.viewport_set_nr = [](void* context, fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
        ((fan::opengl::context_t*)context)->viewport_set(nr, viewport_position_, viewport_size_, window_size);
        };
      cf.viewport_zero = [](void* context, fan::graphics::viewport_nr_t nr) {
        ((fan::opengl::context_t*)context)->viewport_zero(nr);
        };
      cf.viewport_inside = [](void* context, fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
        return ((fan::opengl::context_t*)context)->viewport_inside(nr, position);
        };
      cf.viewport_inside_wir = [](void* context, fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
        return ((fan::opengl::context_t*)context)->viewport_inside_wir(nr, position);
        };
      cf.image_get_pixel_data = [](void* context, fan::graphics::image_nr_t nr, GLenum format, fan::vec2 uvp, fan::vec2 uvs) {
        return ((fan::opengl::context_t*)context)->image_get_pixel_data(nr, format, uvp, uvs);
      };
      return cf;
    }
  }
}