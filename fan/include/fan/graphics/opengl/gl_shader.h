#pragma once

#include <fan/types/types.h>

#include <fan/graphics/renderer.h>

#if fan_renderer == fan_renderer_opengl

#include <fan/types/types.h>
#include <fan/io/file.h>
#include <fan/types/matrix.h>
#include <fan/time/time.h>

#include <fan/graphics/shared_core.h>

#include <fan/graphics/opengl/gl_core.h>

namespace fan {
  struct shader_t {

    int projection_view[2];

    shader_t() = default;

    void open(fan::opengl::context_t* context) {
      id = fan::uninitialized;

      vertex = fan::uninitialized;
      fragment = fan::uninitialized;
      geometry = fan::uninitialized;
    }

    void close(fan::opengl::context_t* context) {
      this->remove(context);
    }

    void use(fan::opengl::context_t* context) const
    {
      context->opengl.glUseProgram(id);
    }

    void remove(fan::opengl::context_t* context) {
      fan_validate_buffer(id, {
        context->opengl.glValidateProgram(id);
      int status = 0;
      context->opengl.glGetProgramiv(id, fan::opengl::GL_VALIDATE_STATUS, &status);
      if (status) {
        context->opengl.glDeleteProgram(id);
      }
      id = fan::uninitialized;
        });
    }

    void set_vertex(fan::opengl::context_t* context, const std::string& vertex_code) {

      if (vertex != fan::uninitialized) {
        context->opengl.glDeleteShader(vertex);
      }

      vertex = context->opengl.glCreateShader(fan::opengl::GL_VERTEX_SHADER);

      char* ptr = (char*)vertex_code.c_str();

      context->opengl.glShaderSource(vertex, 1, &ptr, NULL);

      context->opengl.glCompileShader(vertex);

      checkCompileErrors(context, vertex, "VERTEX");
    }

    void set_fragment(fan::opengl::context_t* context, const std::string& fragment_code) {

      if (fragment != -1) {
        context->opengl.glDeleteShader(fragment);
      }

      fragment = context->opengl.glCreateShader(fan::opengl::GL_FRAGMENT_SHADER);

      char* ptr = (char*)fragment_code.c_str();

      context->opengl.glShaderSource(fragment, 1, &ptr, NULL);

      context->opengl.glCompileShader(fragment);
      checkCompileErrors(context, fragment, "FRAGMENT");
    }

    void set_geometry(fan::opengl::context_t* context, const std::string& geometry_code) {
      if (geometry != -1) {
        context->opengl.glDeleteShader(geometry);
      }

      geometry = context->opengl.glCreateShader(fan::opengl::GL_GEOMETRY_SHADER);

      char* ptr = (char*)geometry_code.c_str();
      context->opengl.glShaderSource(geometry, 1, &ptr, NULL);

      context->opengl.glCompileShader(geometry);
      checkCompileErrors(context, geometry, "GEOMETRY");
    }

    void compile(fan::opengl::context_t* context) {
      if (id != -1) {
        context->opengl.glDeleteProgram(id);
      }

      id = context->opengl.glCreateProgram();
      if (vertex != -1) {
        context->opengl.glAttachShader(id, vertex);
      }
      if (fragment != -1) {
        context->opengl.glAttachShader(id, fragment);
      }
      if (geometry != -1) {
        context->opengl.glAttachShader(id, geometry);
      }

      context->opengl.glLinkProgram(id);
      checkCompileErrors(context, id, "PROGRAM");

      if (vertex != -1) {
        context->opengl.glDeleteShader(vertex);
        vertex = -1;
      }
      if (fragment != -1) {
        context->opengl.glDeleteShader(fragment);
        fragment = -1;
      }
      if (geometry != -1) {
        context->opengl.glDeleteShader(geometry);
        geometry = -1;
      }

      projection_view[0] = context->opengl.glGetUniformLocation(id, "projection");
      projection_view[1] = context->opengl.glGetUniformLocation(id, "view");
    }

    void enable_draw(fan::opengl::context_t* context, fan_2d::opengl::shape shape, uint32_t first, uint32_t count) {
      uint32_t mode = 0;

      switch (shape) {
      case fan_2d::opengl::shape::line: {
        mode = fan::opengl::GL_LINES;
        break;
      }
      case fan_2d::opengl::shape::line_strip: {
        mode = fan::opengl::GL_LINE_STRIP;
        break;
      }
      case fan_2d::opengl::shape::triangle: {
        mode = fan::opengl::GL_TRIANGLES;
        break;
      }
      case fan_2d::opengl::shape::triangle_strip: {
        mode = fan::opengl::GL_TRIANGLE_STRIP;
        break;
      }
      case fan_2d::opengl::shape::triangle_fan: {
        mode = fan::opengl::GL_TRIANGLE_FAN;
        break;
      }
      default: {
        mode = fan::opengl::GL_TRIANGLES;
        fan::print("fan warning - unset input assembly topology in graphics pipeline");
        break;
      }
      }

      context->opengl.glDrawArrays(mode, first, count);
    }

    static constexpr auto validate_error_message = [](const auto str) {
      return "failed to set value for:" + str + " check if variable is used in file so that its not optimized away";
    };

    void set_bool(fan::opengl::context_t* context, const std::string& name, bool value) const {
      auto location = context->opengl.glGetUniformLocation(id, name.c_str());

      #if fan_debug >= fan_debug_low
      fan_validate_value(location, validate_error_message(name));
      #endif

      context->opengl.glUniform1i(location, value);
    }

    void set_int(fan::opengl::context_t* context, const std::string& name, int value) const
    {
      auto location = context->opengl.glGetUniformLocation(id, name.c_str());
    #if fan_debug >= fan_debug_low
      fan_validate_value(location, validate_error_message(name));
    #endif
      context->opengl.glUniform1i(location, value);
    }

    void set_uint(fan::opengl::context_t* context, const std::string& name, uint32_t value) const {
       auto location = context->opengl.glGetUniformLocation(id, name.c_str());
    #if fan_debug >= fan_debug_low
      fan_validate_value(location, validate_error_message(name));
    #endif
      context->opengl.glUniform1ui(location, value);
    }

    void set_int_array(fan::opengl::context_t* context, const std::string& name, int* values, int size) const {
      auto location = context->opengl.glGetUniformLocation(id, name.c_str());

      fan_validate_value(location, validate_error_message(name));

      context->opengl.glUniform1iv(location, size, values);
    }

    void set_float(fan::opengl::context_t* context, const std::string& name, fan::vec2::value_type value) const
    {
      auto location = context->opengl.glGetUniformLocation(id, name.c_str());
      #if fan_debug >= fan_debug_low
        fan_validate_value(location, validate_error_message(name));
      #endif

      if constexpr (std::is_same<fan::vec2::value_type, f32_t>::value) {
        context->opengl.glUniform1f(location, value);
      }
      else {
        context->opengl.glUniform1d(location, value);
      }
    }

    void set_vec2(fan::opengl::context_t* context, const std::string& name, const fan::vec2& value) const
    {
      auto location = context->opengl.glGetUniformLocation(id, name.c_str());

      #if fan_debug >= fan_debug_low
        fan_validate_value(location, validate_error_message(name));
      #endif

      if constexpr (std::is_same<fan::vec2::value_type, f32_t>::value) {
        context->opengl.glUniform2fv(location, 1, (f32_t*)&value.x);
      }
      else {
        context->opengl.glUniform2dv(location, 1, (f64_t*)&value.x);
      }
    }

    void set_vec2(fan::opengl::context_t* context, const std::string& name, f32_t x, f32_t y) const
    {
      auto location = context->opengl.glGetUniformLocation(id, name.c_str());

      #if fan_debug >= fan_debug_low
      fan_validate_value(location, validate_error_message(name));
      #endif

      if constexpr (std::is_same<fan::vec2::value_type, f32_t>::value) {
        context->opengl.glUniform2f(location, x, y);
      }
      else {
        context->opengl.glUniform2d(location, x, y);
      }
    }

    void set_vec3(fan::opengl::context_t* context, const std::string& name, const fan::vec3& value) const
    {
      auto location = context->opengl.glGetUniformLocation(id, name.c_str());

      #if fan_debug >= fan_debug_low
      fan_validate_value(location, validate_error_message(name));
      #endif

      if constexpr (std::is_same<fan::vec3::value_type, float>::value) {
        context->opengl.glUniform3f(location, value.x, value.y, value.z);
      }
      else {
        context->opengl.glUniform3d(location, value.x, value.y, value.z);
      }
    }

    void set_vec4(fan::opengl::context_t* context, const std::string& name, const fan::color& color) const
    {
      auto location = context->opengl.glGetUniformLocation(id, name.c_str());

      #if fan_debug >= fan_debug_low
      fan_validate_value(location, validate_error_message(name));
      #endif

      if constexpr (std::is_same<fan::vec4::value_type, float>::value) {
        context->opengl.glUniform4f(location, color.r, color.g, color.b, color.a);
      }
      else {
        context->opengl.glUniform4d(location, color.r, color.g, color.b, color.a);
      }
    }

    void set_vec4(fan::opengl::context_t* context, const std::string& name, f32_t x, f32_t y, f32_t z, f32_t w) const
    {
      auto location = context->opengl.glGetUniformLocation(id, name.c_str());

      #if fan_debug >= fan_debug_low
      fan_validate_value(location, validate_error_message(name));
      #endif

      if constexpr (std::is_same<fan::vec4::value_type, float>::value) {
        context->opengl.glUniform4f(location, x, y, z, w);
      }
      else {
        context->opengl.glUniform4d(location, x, y, z, w);
      }
    }

    void set_projection(fan::opengl::context_t* context, fan::mat4 mat) {
      context->opengl.glUniformMatrix4fv(projection_view[0], 1, fan::opengl::GL_FALSE, (f32_t*)&mat[0][0]);
    }
    void set_view(fan::opengl::context_t* context, fan::mat4 mat) {
      context->opengl.glUniformMatrix4fv(projection_view[1], 1, fan::opengl::GL_FALSE, (f32_t*)&mat[0][0]);
    }

    void set_mat4(fan::opengl::context_t* context, const std::string& name, fan::mat4 mat) const {

      auto location = context->opengl.glGetUniformLocation(id, name.c_str());

    #if fan_debug >= fan_debug_low
      fan_validate_value(location, validate_error_message(name));
    #endif
      if constexpr (std::is_same<fan::mat4::value_type::value_type, float>::value) {
        context->opengl.glUniformMatrix4fv(location, 1, fan::opengl::GL_FALSE, (f32_t*)&mat[0][0]);
      }
      else {
        context->opengl.glUniformMatrix4dv(location, 1, fan::opengl::GL_FALSE, (f64_t*)&mat[0][0]);
      }

    }

    void set_mat4(fan::opengl::context_t* context, const std::string& name, f32_t* value, uint32_t count) const {

      auto location = context->opengl.glGetUniformLocation(id, name.c_str());

      fan_validate_value(location, validate_error_message(name));

      if constexpr (std::is_same<fan::mat4::value_type::value_type, float>::value) {
        context->opengl.glUniformMatrix4fv(location, count, fan::opengl::GL_FALSE, value);
      }
      else {
        context->opengl.glUniformMatrix4dv(location, count, fan::opengl::GL_FALSE, (f64_t*)value);
      }
    }

    uint32_t id;

    uint32_t vertex, fragment, geometry;

  private:

    void checkCompileErrors(fan::opengl::context_t* context, fan::opengl::GLuint shader, std::string type)
    {
      fan::opengl::GLint success;

      bool program = type == "PROGRAM";

      if (program == false) {
        context->opengl.glGetShaderiv(shader, fan::opengl::GL_COMPILE_STATUS, &success);
      }
      else {
        context->opengl.glGetProgramiv(shader, fan::opengl::GL_LINK_STATUS, &success);
      }

      if (success) {
        return;
      }

      int buffer_size = 0;
      context->opengl.glGetShaderiv(shader, fan::opengl::GL_INFO_LOG_LENGTH, &buffer_size);


      if (buffer_size <= 0) {
        return;
      }

      std::string buffer;
      buffer.resize(buffer_size);

      if (!success)
      {

      #define get_info_log(is_program, program, str_buffer, size) \
                if (is_program) \
                context->opengl.glGetProgramInfoLog(program, size, nullptr, buffer.data()); \
                else \
                context->opengl.glGetShaderInfoLog(program, size, nullptr, buffer.data());

        get_info_log(program, shader, buffer, buffer_size);

        fan::print("failed to compile type: " + type, buffer);

        throw std::runtime_error("failed to compile shaders");
      }
    }
};
}


#endif