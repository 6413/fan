#ifndef SHADER_H
#define SHADER_H
#define GLEW_STATIC
#include <GLFW/glfw3.h>
#include <GL/glew.h>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <FAN/t.h>

class Shader
{
public:
    Shader() : ID(-1) {}
    unsigned int ID;
    // constructor generates the shader on the fly
    // ------------------------------------------------------------------------
    Shader(const char* vertexPath, const char* fragmentPath, const char* geometryPath = nullptr)
    {
        // 1. retrieve the vertex/fragment source code from filePath
        std::string vertexCode;
        std::string fragmentCode;
        std::string geometryCode;
        std::ifstream vShaderFile;
        std::ifstream fShaderFile;
        std::ifstream gShaderFile;
        // ensure ifstream objects can throw exceptions:
        vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        gShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        try
        {
            // open files
            vShaderFile.open(vertexPath);
            fShaderFile.open(fragmentPath);
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
            if (geometryPath != nullptr)
            {
                gShaderFile.open(geometryPath);
                std::stringstream gShaderStream;
                gShaderStream << gShaderFile.rdbuf();
                gShaderFile.close();
                geometryCode = gShaderStream.str();
            }
        }
        catch (std::ifstream::failure e)
        {
            std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
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
        unsigned int geometry;
        if (geometryPath != nullptr)
        {
            const char* gShaderCode = geometryCode.c_str();
            geometry = glCreateShader(GL_GEOMETRY_SHADER);
            glShaderSource(geometry, 1, &gShaderCode, NULL);
            glCompileShader(geometry);
            checkCompileErrors(geometry, "GEOMETRY");
        }
        // shader Program
        ID = glCreateProgram();
        glAttachShader(ID, vertex);
        glAttachShader(ID, fragment);
        if (geometryPath != nullptr)
            glAttachShader(ID, geometry);
        glLinkProgram(ID);
        checkCompileErrors(ID, "PROGRAM");
        // delete the shaders as they're linked into our program now and no longer necessery
        glDeleteShader(vertex);
        glDeleteShader(fragment);
        if (geometryPath != nullptr)
            glDeleteShader(geometry);

    }

    ~Shader() {

    }

    void use() const
    {
        glUseProgram(ID);
    }

    void set_int(const std::string& name, int value) const
    {
        glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
    }

    void set_float(const std::string& name, f32_t value) const
    {
        if constexpr (std::is_same<f32_t, float>::value) {
            glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
        }
        else {
            glUniform1d(glGetUniformLocation(ID, name.c_str()), value);
        }
    }

    void set_vec2(const std::string& name, const fan::vec2& value) const
    {
        if constexpr (std::is_same<f32_t, float>::value) {
            glUniform2fv(glGetUniformLocation(ID, name.c_str()), 1, (f32_t*)&value.x);
        }
        else {
            glUniform2dv(glGetUniformLocation(ID, name.c_str()), 1, (f64_t*)&value.x);
        }
    }

    void set_vec2(const std::string& name, float x, float y) const
    {
        if constexpr (std::is_same<f32_t, float>::value) {
            glUniform2f(glGetUniformLocation(ID, name.c_str()), x, y);
        }
        else {
            glUniform2d(glGetUniformLocation(ID, name.c_str()), x, y);
        }
    }

    void set_vec3(const std::string& name, const fan::vec3& value) const
    {
        if constexpr (std::is_same<f32_t, float>::value) {
            glUniform3f(glGetUniformLocation(ID, name.c_str()), value.x, value.y, value.z);
        }
        else {
            glUniform3d(glGetUniformLocation(ID, name.c_str()), value.x, value.y, value.z);
        }
    }

    void set_vec4(const std::string& name, const fan::color& color) const
    {
        if constexpr (std::is_same<f32_t, float>::value) {
            glUniform4f(glGetUniformLocation(ID, name.c_str()), color.r, color.g, color.b, color.a);
        }
        else {
            glUniform4d(glGetUniformLocation(ID, name.c_str()), color.r, color.g, color.b, color.a);
        }
    }

    void set_vec4(const std::string& name, f32_t x, f32_t y, f32_t z, f32_t w) const
    {
        if constexpr (std::is_same<f32_t, float>::value) {
            glUniform4f(glGetUniformLocation(ID, name.c_str()), x, y, z, w);
        }
        else {
            glUniform4d(glGetUniformLocation(ID, name.c_str()), x, y, z, w);
        }
    }

    void set_mat4(const std::string& name, fan::mat4 mat) const { // ei saanu kai olla const
        if constexpr (std::is_same<f32_t, float>::value) {
            glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, (f32_t*)&mat[0][0]);
        }
        else {
            glUniformMatrix4dv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, (f64_t*)&mat[0][0]);
        }
    }

    void upload_data(GLuint buffer, GLsizeiptr size, const void* data, GLenum usage) {
        glBindBuffer(GL_ARRAY_BUFFER, buffer);
        glBufferData(GL_ARRAY_BUFFER, size, data, usage);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

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
                std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
            }
        }
        else
        {
            glGetProgramiv(shader, GL_LINK_STATUS, &success);
            if (!success)
            {
                glGetProgramInfoLog(shader, 1024, NULL, infoLog);
                std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
            }
        }
    }
};
#endif