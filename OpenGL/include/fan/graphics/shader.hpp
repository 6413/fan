#pragma once

#define GLEW_STATIC
#include <GL/glew.h>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <fan/types/types.hpp>
#include <fan/io/file.hpp>
#include <fan/math/matrix.hpp>

namespace fan {
	class shader {
    public:
        shader() : m_id(fan::uninitialized) {}

        shader(const std::string& vertex_path, const std::string& fragment_path, const std::string& geometry_path = std::string())
            :  m_id(fan::uninitialized), m_vertex_path(vertex_path), m_fragment_path(fragment_path), m_geometry_path(geometry_path)
        {
            this->generate_shader(vertex_path, fragment_path, geometry_path);
        }

        shader(const shader& s) : m_id(fan::uninitialized) {
            this->operator=(s);
        }
        
        shader(shader&& s) noexcept : m_id(fan::uninitialized) {
            this->operator=(std::move(s));
        }

        ~shader() {
            this->remove();
        }

        shader& operator=(const shader& s) {

            this->remove();

            this->m_vertex_path   = s.m_vertex_path;
            this->m_fragment_path = s.m_fragment_path;
            this->m_geometry_path = s.m_geometry_path;

            this->generate_shader(m_vertex_path, m_fragment_path, m_geometry_path);

            return *this;
        }

        shader& operator=(shader&& s) noexcept {

            this->remove();

            this->m_id = s.m_id;

            this->m_vertex_path   = std::move(s.m_vertex_path);
            this->m_fragment_path = std::move(s.m_fragment_path);
            this->m_geometry_path = std::move(s.m_geometry_path);

            s.m_id = fan::uninitialized;

            return *this;
        }

        void use() const
        {
            glUseProgram(m_id);
        }

        void remove() {
            fan_validate_buffer(m_id, {
                glValidateProgram(m_id);
                int status = 0;
                glGetProgramiv(m_id, GL_VALIDATE_STATUS, &status);
                if (status) {
                    glDeleteProgram(m_id);
                }
                m_id = fan::uninitialized;
            });
        }

        void generate_shader(const std::string& vertex_path, const std::string& fragment_path, const std::string& geometry_path = std::string()) {
            std::string vertexCode;
            std::string fragmentCode;
            std::string geometryCode;
            std::ifstream vShaderFile;
            std::ifstream fShaderFile;
            std::ifstream gShaderFile;

            if (!fan::io::file::exists(vertex_path)) {
                fan::print("vertex shader does not exist:", vertex_path);
                exit(1);
            }
            if (!fan::io::file::exists(fragment_path)) {
                fan::print("fragment shader does not exist:", fragment_path);
                exit(1);
            }
            vShaderFile.open(vertex_path);
            fShaderFile.open(fragment_path);
            std::stringstream vShaderStream, fShaderStream;
            // read file's buffer contents into streams
            vShaderStream << vShaderFile.rdbuf();
            fShaderStream << fShaderFile.rdbuf();
            // close file handlers
            vShaderFile.close();
            fShaderFile.close();
            // convert stream into string
            vertexCode = vShaderStream.str();
            fragmentCode = fShaderStream.str();
            // if geometry shader path is present, also load a geometry shader
            if (!geometry_path.empty())
            {
                if (!fan::io::file::exists(geometry_path)) {
                    fan::print("geometry shader does not exist:", geometry_path);
                    exit(1);
                }
                gShaderFile.open(geometry_path);
                std::stringstream gShaderStream;
                gShaderStream << gShaderFile.rdbuf();
                gShaderFile.close();
                geometryCode = gShaderStream.str();
            }
            const char* vShaderCode = vertexCode.c_str();
            const char* fShaderCode = fragmentCode.c_str();
            // 2. compile shaders
            unsigned int vertex, fragment;
            // vertex shader

            vertex = glCreateShader(GL_VERTEX_SHADER);
            glShaderSource(vertex, 1, &vShaderCode, NULL);
            glCompileShader(vertex);
            checkCompileErrors(vertex, "VERTEX");
            // fragment Shader
            fragment = glCreateShader(GL_FRAGMENT_SHADER);
            glShaderSource(fragment, 1, &fShaderCode, NULL);
            glCompileShader(fragment);
            checkCompileErrors(fragment, "FRAGMENT");
            // if geometry shader is given, compile geometry shader
            unsigned int geometry = -1;
            if (!geometry_path.empty())
            {
                const char* gShaderCode = geometryCode.c_str();
                geometry = glCreateShader(GL_GEOMETRY_SHADER);
                glShaderSource(geometry, 1, &gShaderCode, NULL);
                glCompileShader(geometry);
                checkCompileErrors(geometry, "GEOMETRY");
            }
            // shader Program
            m_id = glCreateProgram();
            glAttachShader(m_id, vertex);
            glAttachShader(m_id, fragment);

            if (!geometry_path.empty())
                glAttachShader(m_id, geometry);

            glLinkProgram(m_id);
            checkCompileErrors(m_id, "PROGRAM");

            glDeleteShader(vertex);
            glDeleteShader(fragment);
            if (!geometry_path.empty())
                glDeleteShader(geometry);
        }

        void set_bool(const std::string& name, bool value) const {
            glUniform1i(glGetUniformLocation(m_id, name.c_str()), value);
        }

        void set_int(const std::string& name, int value) const
        {
            glUniform1i(glGetUniformLocation(m_id, name.c_str()), value);
        }

        void set_int_array(const std::string& name, int* values, int size) const {
            glUniform1iv(glGetUniformLocation(m_id, name.c_str()), size, values);
        }

        void set_float(const std::string& name, fan::vec2::value_type value) const
        {
            if constexpr (std::is_same<fan::vec2::value_type, float>::value) {
                glUniform1f(glGetUniformLocation(m_id, name.c_str()), value);
            }
            else {
                glUniform1d(glGetUniformLocation(m_id, name.c_str()), value);
            }
        }

        void set_vec2(const std::string& name, const fan::vec2& value) const
        {
            if constexpr (std::is_same<fan::vec2::value_type, float>::value) {
                glUniform2fv(glGetUniformLocation(m_id, name.c_str()), 1, (f32_t*)&value.x);
            }
            else {
                glUniform2dv(glGetUniformLocation(m_id, name.c_str()), 1, (f64_t*)&value.x);
            }
        }

        void set_vec2(const std::string& name, float x, float y) const
        {
            if constexpr (std::is_same<fan::vec2::value_type, float>::value) {
                glUniform2f(glGetUniformLocation(m_id, name.c_str()), x, y);
            }
            else {
                glUniform2d(glGetUniformLocation(m_id, name.c_str()), x, y);
            }
        }

        void set_vec3(const std::string& name, const fan::vec3& value) const
        {
            if constexpr (std::is_same<fan::vec3::value_type, float>::value) {
                glUniform3f(glGetUniformLocation(m_id, name.c_str()), value.x, value.y, value.z);
            }
            else {
                glUniform3d(glGetUniformLocation(m_id, name.c_str()), value.x, value.y, value.z);
            }
        }

        void set_vec4(const std::string& name, const fan::color& color) const
        {
            if constexpr (std::is_same<fan::vec4::value_type, float>::value) {
                glUniform4f(glGetUniformLocation(m_id, name.c_str()), color.r, color.g, color.b, color.a);
            }
            else {
                glUniform4d(glGetUniformLocation(m_id, name.c_str()), color.r, color.g, color.b, color.a);
            }
        }

        void set_vec4(const std::string& name, f32_t x, f32_t y, f32_t z, f32_t w) const
        {
            if constexpr (std::is_same<fan::vec4::value_type, float>::value) {
                glUniform4f(glGetUniformLocation(m_id, name.c_str()), x, y, z, w);
            }
            else {
                glUniform4d(glGetUniformLocation(m_id, name.c_str()), x, y, z, w);
            }
        }

        void set_mat4(const std::string& name, fan::mat4 mat) const { // ei saanu kai olla const
            if constexpr (std::is_same<fan::mat4::value_type::value_type, float>::value) {
                glUniformMatrix4fv(glGetUniformLocation(m_id, name.c_str()), 1, GL_FALSE, (f32_t*)&mat[0][0]);
            }
            else {
                glUniformMatrix4dv(glGetUniformLocation(m_id, name.c_str()), 1, GL_FALSE, (f64_t*)&mat[0][0]);
            }
        }

        unsigned int m_id;

        std::string m_vertex_path;
        std::string m_fragment_path;
        std::string m_geometry_path;

    private:

        void checkCompileErrors(GLuint shader, std::string type)
        {
            GLint success;
            GLchar infoLog[1024];
            if (type != "PROGRAM")
            {
                glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
                if (!success)
                {
                    glGetShaderInfoLog(shader, 1024, NULL, infoLog);
                    fan::print("failed to compile for files:", m_vertex_path, "type:", type, '\n', infoLog);
                }
            }
            else
            {
                glGetProgramiv(shader, GL_LINK_STATUS, &success);
                if (!success)
                {
                    glGetProgramInfoLog(shader, 1024, NULL, infoLog);
                    std::cout << "failed to compile type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
                }
            }
        }
    };
}