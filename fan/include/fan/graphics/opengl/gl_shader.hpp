#pragma once

#include <fan/graphics/renderer.hpp>

#if fan_renderer == fan_renderer_opengl

#define GLEW_STATIC
#include <GL/glew.h>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <fan/types/types.hpp>
#include <fan/io/file.hpp>
#include <fan/types/matrix.hpp>

#include <fan/graphics/shared_core.hpp>

namespace fan {
	class base_shader {
    public:

        ~base_shader() {
            this->remove();
        }

        void use() const
        {
            glUseProgram(id);
        }

        void remove() {
          fan_validate_buffer(id, {
              glValidateProgram(id);
              int status = 0;
              glGetProgramiv(id, GL_VALIDATE_STATUS, &status);
              if (status) {
                  glDeleteProgram(id);
              }
              id = fan::uninitialized;
          });
        }

        void set_vertex(const std::string& vertex_code) {

          if (vertex != -1) {
            glDeleteShader(vertex);
          }

          vertex = glCreateShader(GL_VERTEX_SHADER);

          char* ptr = (char*)vertex_code.c_str();

          glShaderSource(vertex, 1, &ptr, NULL);

          glCompileShader(vertex);
          checkCompileErrors(vertex, "VERTEX");
        }

        void set_fragment(const std::string& fragment_code) {

          if (fragment != -1) {
            glDeleteShader(fragment);
          }

          fragment = glCreateShader(GL_FRAGMENT_SHADER);

          char* ptr = (char*)fragment_code.c_str();

          glShaderSource(fragment, 1, &ptr, NULL);

          glCompileShader(fragment);
          checkCompileErrors(fragment, "FRAGMENT");
        }

        void set_geometry(const std::string& geometry_code) {
          if (geometry != -1) {
            glDeleteShader(geometry);
          }

          geometry = glCreateShader(GL_GEOMETRY_SHADER);

          char* ptr = (char*)geometry_code.c_str();
          glShaderSource(geometry, 1, &ptr, NULL);

          glCompileShader(geometry);
          checkCompileErrors(geometry, "GEOMETRY");
        }

        void compile() {
          if (id != -1) {
            glDeleteProgram(id);
          }

          id = glCreateProgram();
          if (vertex != -1) {
            glAttachShader(id, vertex);
          }
          if (fragment != -1) {
            glAttachShader(id, fragment);
          }
          if (geometry != -1) {
            glAttachShader(id, geometry);
          }

          glLinkProgram(id);
          checkCompileErrors(id, "PROGRAM");

          if (vertex != -1) {
            glDeleteShader(vertex);
            vertex = -1;
          }
          if (fragment != -1) {
            glDeleteShader(fragment);
            fragment = -1;
          }
          if (geometry != -1) {
            glDeleteShader(geometry);
            geometry = -1;
          }
        }

        void enable_draw(fan_2d::graphics::shape shape, uint32_t first, uint32_t count) {
          uint32_t mode = 0;

	        switch(shape) {
		        case fan_2d::graphics::shape::line: {
			        mode = GL_LINES;
			        break;
		        }
		        case fan_2d::graphics::shape::line_strip: {
			        mode = GL_LINE_STRIP;
			        break;
		        }
		        case fan_2d::graphics::shape::triangle: {
			        mode = GL_TRIANGLES;
			        break;
		        }
		        case fan_2d::graphics::shape::triangle_strip: {
			        mode = GL_TRIANGLE_STRIP;
			        break;
		        }
		        case fan_2d::graphics::shape::triangle_fan: {
			        mode = GL_TRIANGLE_FAN;
			        break;
		        }
		        default: {
			        mode = GL_TRIANGLES;
			        fan::print("fan warning - unset input assembly topology in graphics pipeline");
			        break;
		        }
	        }

          glDrawArrays(mode, first, count);
        }

        static constexpr auto validate_error_message = [](const auto str) {
            return "failed to set value for:" + str + " check if variable is used in file so that its not optimized away";
        };

        void set_bool(const std::string& name, bool value) const {
            auto location = glGetUniformLocation(id, name.c_str());

            fan_validate_value(location, validate_error_message(name));

            glUniform1i(location, value);
        }

        void set_int(const std::string& name, int value) const
        {
            auto location = glGetUniformLocation(id, name.c_str());

            fan_validate_value(location, validate_error_message(name));

            glUniform1i(location, value);
        }

        void set_int_array(const std::string& name, int* values, int size) const {
            auto location = glGetUniformLocation(id, name.c_str());
            
            fan_validate_value(location, validate_error_message(name));

            glUniform1iv(location, size, values);
        }

        void set_float(const std::string& name, fan::vec2::value_type value) const
        {
            auto location = glGetUniformLocation(id, name.c_str());

            fan_validate_value(location, validate_error_message(name));

            if constexpr (std::is_same<fan::vec2::value_type, f32_t>::value) {
                glUniform1f(location, value);
            }
            else {
                glUniform1d(location, value);
            }
        }

        void set_vec2(const std::string& name, const fan::vec2& value) const
        {
            auto location = glGetUniformLocation(id, name.c_str());

            fan_validate_value(location, validate_error_message(name));

            if constexpr (std::is_same<fan::vec2::value_type, f32_t>::value) {
                glUniform2fv(location, 1, (f32_t*)&value.x);
            }
            else {
                glUniform2dv(location, 1, (f64_t*)&value.x);
            }
        }

        void set_vec2(const std::string& name, f32_t x, f32_t y) const
        {
            auto location = glGetUniformLocation(id, name.c_str());

            fan_validate_value(location, validate_error_message(name));

            if constexpr (std::is_same<fan::vec2::value_type, f32_t>::value) {
                glUniform2f(location, x, y);
            }
            else {
                glUniform2d(location, x, y);
            }
        }

        void set_vec3(const std::string& name, const fan::vec3& value) const
        {
            auto location = glGetUniformLocation(id, name.c_str());

            fan_validate_value(location, validate_error_message(name));

            if constexpr (std::is_same<fan::vec3::value_type, float>::value) {
                glUniform3f(location, value.x, value.y, value.z);
            }
            else {
                glUniform3d(location, value.x, value.y, value.z);
            }
        }

        void set_vec4(const std::string& name, const fan::color& color) const
        {
            auto location = glGetUniformLocation(id, name.c_str());

            fan_validate_value(location, validate_error_message(name));

            if constexpr (std::is_same<fan::vec4::value_type, float>::value) {
                glUniform4f(location, color.r, color.g, color.b, color.a);
            }
            else {
                glUniform4d(location, color.r, color.g, color.b, color.a);
            }
        }

        void set_vec4(const std::string& name, f32_t x, f32_t y, f32_t z, f32_t w) const
        {
            auto location = glGetUniformLocation(id, name.c_str());

            fan_validate_value(location, validate_error_message(name));

            if constexpr (std::is_same<fan::vec4::value_type, float>::value) {
                glUniform4f(location, x, y, z, w);
            }
            else {
                glUniform4d(location, x, y, z, w);
            }
        }

        void set_mat4(const std::string& name, fan::mat4 mat) const { // ei saanu kai olla const

            auto location = glGetUniformLocation(id, name.c_str());

            fan_validate_value(location, validate_error_message(name));

            if constexpr (std::is_same<fan::mat4::value_type::value_type, float>::value) {
                glUniformMatrix4fv(location, 1, GL_FALSE, (f32_t*)&mat[0][0]);
            }
            else {
                glUniformMatrix4dv(location, 1, GL_FALSE, (f64_t*)&mat[0][0]);
            }
        }

        unsigned int id = -1;

        uint32_t vertex = -1, fragment = -1, geometry = -1;

        std::string m_vertex_path;
        std::string m_fragment_path;
        std::string m_geometry_path;

    private:

        void checkCompileErrors(GLuint shader, std::string type)
        {
            GLint success;

            bool program = type == "PROGRAM";

            if (program == false) {
              glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            }
            else {
              glGetProgramiv(shader, GL_LINK_STATUS, &success);
            }

            if (success) {
              return;
            }
            
            int buffer_size = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &buffer_size);


            if (buffer_size <= 0) {
              return;
            }

            std::string buffer;
            buffer.resize(buffer_size);

            if (!success)
            {

              #define get_info_log(is_program, program, str_buffer, size) \
                if (is_program) \
                glGetProgramInfoLog(program, size, nullptr, buffer.data()); \
                else \
                glGetShaderInfoLog(program, size, nullptr, buffer.data());

              get_info_log(program, shader, buffer, buffer_size);

              fan::print("failed to compile type: " + type, buffer);

              throw std::runtime_error("failed to compile shaders");
            }
        }
    };

  struct shader_t : public std::shared_ptr<fan::base_shader> {
    shader_t() : std::shared_ptr<fan::base_shader>(std::make_shared<fan::base_shader>(fan::base_shader())) {}
  };

}


#endif