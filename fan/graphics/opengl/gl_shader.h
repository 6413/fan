struct shader_t {

  shader_list_NodeData_t& get_shader() const {
    return gloco->shader_list[shader_reference];
  }
  shader_list_NodeData_t& get_shader() {
    return gloco->shader_list[shader_reference];
  }

  shader_list_NodeReference_t shader_reference;

  void open() {
    shader_reference = gloco->shader_list.NewNode();
    get_shader().id = fan::uninitialized;
    get_shader().shader = this;

    get_shader().vertex = fan::uninitialized;
    get_shader().fragment = fan::uninitialized;
  }

  void close() {
    auto& context = gloco->get_context();
    this->remove();
    gloco->shader_list.Recycle(shader_reference);
  }

  shader_t() = default;

  shader_t(shader_t&& shader) {
    shader_reference = shader.shader_reference;
    gloco->shader_list[shader_reference].shader = this;
    shader.shader_reference.sic();
  }
  shader_t(const shader_t& shader) {
    open();
    gloco->shader_list[shader_reference].on_activate = gloco->shader_list[shader.shader_reference].on_activate;
  }

  shader_t& operator=(const shader_t& s) {
    if (this != &s) {
      open();
      gloco->shader_list[shader_reference].on_activate = gloco->shader_list[s.shader_reference].on_activate;
    }
    return *this;
  }
  shader_t& operator=(shader_t&& s) {
    if (this != &s) {
      if (!shader_reference.iic()) {
        close();
      }
      shader_reference = s.shader_reference;
      gloco->shader_list[shader_reference].shader = this;
      gloco->shader_list[shader_reference].on_activate = gloco->shader_list[s.shader_reference].on_activate;
      s.shader_reference.sic();
    }
    return *this;
  }
  ~shader_t() {
    if (shader_reference.iic()) {
      return;
    }
    close();
  }

  void use() const {
    auto& context = gloco->get_context();
    if (get_shader().id == context.current_program) {
      return;
    }
    context.opengl.call(context.opengl.glUseProgram, get_shader().id);
    context.current_program = get_shader().id;
  }

  void remove() {
    auto& context = gloco->get_context();
    fan_validate_buffer(get_shader().id, {
      context.opengl.call(context.opengl.glValidateProgram, get_shader().id);
    int status = 0;
    context.opengl.call(context.opengl.glGetProgramiv, get_shader().id, fan::opengl::GL_VALIDATE_STATUS, &status);
    if (status) {
      context.opengl.call(context.opengl.glDeleteProgram, get_shader().id);
    }
    get_shader().id = fan::uninitialized;
    });
  }

  void set_vertex(char* vertex_ptr, fan::opengl::GLint length) {
    auto& context = gloco->get_context();

    if (get_shader().vertex != fan::uninitialized) {
      context.opengl.call(context.opengl.glDeleteShader, get_shader().vertex);
    }

    get_shader().vertex = context.opengl.call(context.opengl.glCreateShader, fan::opengl::GL_VERTEX_SHADER);

    context.opengl.call(context.opengl.glShaderSource, get_shader().vertex, 1, &vertex_ptr, &length);

    context.opengl.call(context.opengl.glCompileShader, get_shader().vertex);

    checkCompileErrors(context, get_shader().vertex, "VERTEX");
  }

  void set_vertex(const fan::string& vertex_code) {
    auto& context = gloco->get_context();

    if (get_shader().vertex != fan::uninitialized) {
      context.opengl.call(context.opengl.glDeleteShader, get_shader().vertex);
    }

    get_shader().vertex = context.opengl.call(context.opengl.glCreateShader, fan::opengl::GL_VERTEX_SHADER);

    char* ptr = (char*)vertex_code.c_str();
    fan::opengl::GLint length = vertex_code.size();

    context.opengl.call(context.opengl.glShaderSource, get_shader().vertex, 1, &ptr, &length);
    context.opengl.call(context.opengl.glCompileShader, get_shader().vertex);

    checkCompileErrors(context, get_shader().vertex, "VERTEX");
  }

  void set_fragment( char* fragment_ptr, fan::opengl::GLint length) {
    auto& context = gloco->get_context();

    if (get_shader().fragment != -1) {
      context.opengl.glDeleteShader(get_shader().fragment);
    }

    get_shader().fragment = context.opengl.glCreateShader(fan::opengl::GL_FRAGMENT_SHADER);
    context.opengl.glShaderSource(get_shader().fragment, 1, &fragment_ptr, &length);

    context.opengl.glCompileShader(get_shader().fragment);
    checkCompileErrors(context, get_shader().fragment, "FRAGMENT");
  }
  void set_fragment(const fan::string& fragment_code) {

    auto& context = gloco->get_context();

    if (get_shader().fragment != -1) {
      context.opengl.call(context.opengl.glDeleteShader, get_shader().fragment);
    }

    get_shader().fragment = context.opengl.call(context.opengl.glCreateShader, fan::opengl::GL_FRAGMENT_SHADER);

    char* ptr = (char*)fragment_code.c_str();
    fan::opengl::GLint length = fragment_code.size();

    context.opengl.call(context.opengl.glShaderSource, get_shader().fragment, 1, &ptr, &length);

    context.opengl.call(context.opengl.glCompileShader, get_shader().fragment);
    checkCompileErrors(context, get_shader().fragment, "FRAGMENT");
  }

  void compile() {
    auto& context = gloco->get_context();

    if (get_shader().id != -1) {
      context.opengl.call(context.opengl.glDeleteProgram, get_shader().id);
    }

    get_shader().id = context.opengl.call(context.opengl.glCreateProgram);
    if (get_shader().vertex != -1) {
      context.opengl.call(context.opengl.glAttachShader, get_shader().id, get_shader().vertex);
    }
    if (get_shader().fragment != -1) {
      context.opengl.call(context.opengl.glAttachShader, get_shader().id, get_shader().fragment);
    }

    context.opengl.call(context.opengl.glLinkProgram, get_shader().id);
    checkCompileErrors(context, get_shader().id, "PROGRAM");

    if (get_shader().vertex != -1) {
      context.opengl.call(context.opengl.glDeleteShader, get_shader().vertex);
      get_shader().vertex = -1;
    }
    if (get_shader().fragment != -1) {
      context.opengl.call(context.opengl.glDeleteShader, get_shader().fragment);
      get_shader().fragment = -1;
    }

    get_shader().projection_view[0] = context.opengl.call(context.opengl.glGetUniformLocation, get_shader().id, "projection");
    get_shader().projection_view[1] = context.opengl.call(context.opengl.glGetUniformLocation, get_shader().id, "view");
  }

  static constexpr auto validate_error_message = [](const auto str) {
    return "failed to set value for:" + str + " check if variable is used in file so that its not optimized away";
  };

  void set_bool(const fan::string& name, bool value) const {
    auto& context = gloco->get_context();
    set_int(name, value);
  }

  void set_int(const fan::string& name, int value) const {
    auto& context = gloco->get_context();
    auto location = context.opengl.call(context.opengl.glGetUniformLocation, get_shader().id, name.c_str());
    #if fan_debug >= fan_debug_insanity
    fan_validate_value(location, validate_error_message(name));
    #endif
    context.opengl.call(context.opengl.glUniform1i, location, value);
  }

  void set_uint(const fan::string& name, uint32_t value) const {
    auto& context = gloco->get_context();
    auto location = context.opengl.call(context.opengl.glGetUniformLocation, get_shader().id, name.c_str());
    #if fan_debug >= fan_debug_insanity
    fan_validate_value(location, validate_error_message(name));
    #endif
    context.opengl.call(context.opengl.glUniform1ui, location, value);
  }

  void set_int_array(const fan::string& name, int* values, int size) const {
    auto& context = gloco->get_context();
    auto location = context.opengl.call(context.opengl.glGetUniformLocation, get_shader().id, name.c_str());
    #if fan_debug >= fan_debug_insanity
    fan_validate_value(location, validate_error_message(name));
    #endif
    context.opengl.call(context.opengl.glUniform1iv, location, size, values);
}

  void set_uint_array(const fan::string& name, uint32_t* values, int size) const {
    auto& context = gloco->get_context();
    auto location = context.opengl.call(context.opengl.glGetUniformLocation, get_shader().id, name.c_str());
    #if fan_debug >= fan_debug_insanity
    fan_validate_value(location, validate_error_message(name));
    #endif
    context.opengl.call(context.opengl.glUniform1uiv, location, size, values);
  }

  void set_float_array(const fan::string& name, f32_t* values, int size) const {
    auto& context = gloco->get_context();
    auto location = context.opengl.call(context.opengl.glGetUniformLocation, get_shader().id, name.c_str());
    #if fan_debug >= fan_debug_insanity
    fan_validate_value(location, validate_error_message(name));
    #endif
    context.opengl.call(context.opengl.glUniform1fv, location, size, values);
  }

  void set_float(const fan::string& name, fan::vec2::value_type value) const {
    auto& context = gloco->get_context();
    auto location = context.opengl.call(context.opengl.glGetUniformLocation, get_shader().id, name.c_str());
    #if fan_debug >= fan_debug_insanity
    fan_validate_value(location, validate_error_message(name));
    #endif
    if constexpr (std::is_same<fan::vec2::value_type, f32_t>::value) {
      context.opengl.call(context.opengl.glUniform1f, location, value);
    }
    else {
      context.opengl.call(context.opengl.glUniform1d, location, value);
    }
}

  void set_vec2(const fan::string& name, const fan::vec2& value) const {
    auto& context = gloco->get_context();
    auto location = context.opengl.call(context.opengl.glGetUniformLocation, get_shader().id, name.c_str());
    #if fan_debug >= fan_debug_insanity
    fan_validate_value(location, validate_error_message(name));
    #endif
    if constexpr (std::is_same<fan::vec2::value_type, f32_t>::value) {
      context.opengl.call(context.opengl.glUniform2fv, location, 1, (f32_t*)&value.x);
    }
    else {
      context.opengl.call(context.opengl.glUniform2dv, location, 1, (f64_t*)&value.x);
    }
  }

  void set_vec2(const fan::string& name, f32_t x, f32_t y) const {
    auto& context = gloco->get_context();
    auto location = context.opengl.call(context.opengl.glGetUniformLocation, get_shader().id, name.c_str());
    #if fan_debug >= fan_debug_insanity
    fan_validate_value(location, validate_error_message(name));
    #endif
    if constexpr (std::is_same<fan::vec2::value_type, f32_t>::value) {
      context.opengl.call(context.opengl.glUniform2f, location, x, y);
  }
    else {
      context.opengl.call(context.opengl.glUniform2d, location, x, y);
    }
}

  void set_vec3(const fan::string& name, const fan::vec3& value) const {
    auto& context = gloco->get_context();
    auto location = context.opengl.call(context.opengl.glGetUniformLocation, get_shader().id, name.c_str());
    #if fan_debug >= fan_debug_insanity
    fan_validate_value(location, validate_error_message(name));
    #endif
    if constexpr (std::is_same<fan::vec3::value_type, float>::value) {
      context.opengl.call(context.opengl.glUniform3f, location, value.x, value.y, value.z);
    }
    else {
      context.opengl.call(context.opengl.glUniform3d, location, value.x, value.y, value.z);
    }
  }

  void set_vec4(const fan::string& name, const fan::color& color) const {
    auto& context = gloco->get_context();
    auto location = context.opengl.call(context.opengl.glGetUniformLocation, get_shader().id, name.c_str());
    #if fan_debug >= fan_debug_insanity
    fan_validate_value(location, validate_error_message(name));
    #endif
    if constexpr (std::is_same<fan::vec4::value_type, float>::value) {
      context.opengl.call(context.opengl.glUniform4f, location, color.r, color.g, color.b, color.a);
    }
    else {
      context.opengl.call(context.opengl.glUniform4d, location, color.r, color.g, color.b, color.a);
    }
  }

  void set_vec4(const fan::string& name, f32_t x, f32_t y, f32_t z, f32_t w) const {
    auto& context = gloco->get_context();
    auto location = context.opengl.call(context.opengl.glGetUniformLocation, get_shader().id, name.c_str());
    #if fan_debug >= fan_debug_insanity
    fan_validate_value(location, validate_error_message(name));
    #endif
    if constexpr (std::is_same<fan::vec4::value_type, float>::value) {
      context.opengl.call(context.opengl.glUniform4f, location, x, y, z, w);
    }
    else {
      context.opengl.call(context.opengl.glUniform4d, location, x, y, z, w);
    }
  }

  void set_camera(auto* camera, auto write_queue, uint32_t flags = 0) {
    auto& context = gloco->get_context();
    context.opengl.call(context.opengl.glUniformMatrix4fv, get_shader().projection_view[0], 1, fan::opengl::GL_FALSE, &camera->m_projection[0][0]);
    context.opengl.call(context.opengl.glUniformMatrix4fv, get_shader().projection_view[1], 1, fan::opengl::GL_FALSE, &camera->m_view[0][0]);
  }

  void set_mat4(const fan::string& name, fan::mat4 mat) const {
    auto& context = gloco->get_context();
    auto location = context.opengl.call(context.opengl.glGetUniformLocation, get_shader().id, name.c_str());
    #if fan_debug >= fan_debug_insanity
    fan_validate_value(location, validate_error_message(name));
    #endif
    if constexpr (std::is_same<fan::mat4::value_type::value_type, float>::value) {
      context.opengl.call(context.opengl.glUniformMatrix4fv, location, 1, fan::opengl::GL_FALSE, (f32_t*)&mat[0][0]);
    }
    else {
      context.opengl.call(context.opengl.glUniformMatrix4dv, location, 1, fan::opengl::GL_FALSE, (f64_t*)&mat[0][0]);
    }
  }

  void set_mat4(const fan::string& name, f32_t* value, uint32_t count) const {
    auto& context = gloco->get_context();
    auto location = context.opengl.call(context.opengl.glGetUniformLocation, get_shader().id, name.c_str());
    fan_validate_value(location, validate_error_message(name));
    if constexpr (std::is_same<fan::mat4::value_type::value_type, float>::value) {
      context.opengl.call(context.opengl.glUniformMatrix4fv, location, count, fan::opengl::GL_FALSE, value);
  }
    else {
      context.opengl.call(context.opengl.glUniformMatrix4dv, location, count, fan::opengl::GL_FALSE, (f64_t*)value);
    }
  }

private:

  void checkCompileErrors(fan::opengl::context_t& context, fan::opengl::GLuint shader, fan::string type)
  {
    fan::opengl::GLint success;

    bool program = type == "PROGRAM";

    if (program == false) {
      context.opengl.call(context.opengl.glGetShaderiv, shader, fan::opengl::GL_COMPILE_STATUS, &success);
    }
    else {
      context.opengl.call(context.opengl.glGetProgramiv, shader, fan::opengl::GL_LINK_STATUS, &success);
    }

    if (success) {
      return;
    }

    int buffer_size = 0;
    context.opengl.glGetShaderiv(shader, fan::opengl::GL_INFO_LOG_LENGTH, &buffer_size);


    if (buffer_size <= 0) {
      return;
    }

    fan::string buffer;
    buffer.resize(buffer_size);

    if (!success)
    {
      int test;
#define get_info_log(is_program, program, str_buffer, size) \
            if (is_program) \
            context.opengl.call(context.opengl.glGetProgramInfoLog, program, size, nullptr, buffer.data()); \
            else \
            context.opengl.call(context.opengl.glGetShaderInfoLog, program, size, &test, buffer.data());

      get_info_log(program, shader, buffer, buffer_size);
          
      fan::print("failed to compile type: " + type, buffer);

      fan::throw_error("failed to compile shaders");
    }
  }
};

static fan::string read_shader(const fan::string& path) {
  fan::string code;
  fan::io::file::read(path, &code);
  return code;
}