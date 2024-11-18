#include <fan/pch.h>
#include "gl_core.h"

#include <fan/physics/collision/rectangle.h>
// for parsing uniform values
#include <regex>

#include <fan/graphics/stb.h>

//-----------------------------context-----------------------------
//-----------------------------context-----------------------------
//-----------------------------context-----------------------------
//-----------------------------context-----------------------------
//-----------------------------context-----------------------------
//-----------------------------context-----------------------------
//-----------------------------context-----------------------------
//-----------------------------context-----------------------------
//-----------------------------context-----------------------------
//-----------------------------context-----------------------------
//-----------------------------context-----------------------------
//-----------------------------context-----------------------------

void fan::opengl::context_t::print_version() {
  fan::print("opengl version supported:", opengl.glGetString(fan::opengl::GL_VERSION));
}

fan::opengl::context_t::context_t(const properties_t&) {
  {
    fan::print("context_t");
    if (!glfwInit()) {
      fan::throw_error("failed to initialize window manager context");
    }
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

    GLFWwindow* dummy_window = glfwCreateWindow(640, 400, "dummy", nullptr, nullptr);
    if (dummy_window == nullptr) {
      fan::throw_error("failed to open dummy window");
    }
    glfwMakeContextCurrent(dummy_window);
    // TODO bad reloads opengl functions twice
    opengl = fan::opengl::opengl_t(true);

    if (major == -1 || minor == -1) {
      opengl.glGetIntegerv(fan::opengl::GL_MAJOR_VERSION, &major);
      opengl.glGetIntegerv(fan::opengl::GL_MINOR_VERSION, &minor);
      fan::print("AAAAA", opengl.glGetIntegerv, major, minor);
    }
    glfwDestroyWindow(dummy_window);
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);

    if (initialized == false) {
#if 1
      glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, major);
      glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minor);
      glfwWindowHint(GLFW_SAMPLES, 0);

      if ((major > 3) || (major == 3 && minor > 2)) {
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
      }

      if ((major > 3) || (major == 3 && minor > 0)) {
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, true);
      }
#else // renderdoc debug
      glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
      glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
      glfwWindowHint(GLFW_SAMPLES, 0);

      glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
      glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, true);
#endif

      glfwSetErrorCallback(error_callback);
      initialized = true;
    }

  }
}

void fan::opengl::context_t::set_depth_test(bool flag) {
  if (flag) {
    opengl.call(opengl.glEnable, fan::opengl::GL_DEPTH_TEST);
  }
  else {
    opengl.call(opengl.glDisable, fan::opengl::GL_DEPTH_TEST);
  }
}

void fan::opengl::context_t::set_blending(bool flag) {
  if (flag) {
    opengl.call(opengl.glDisable, fan::opengl::GL_BLEND);
  }
  else {
    opengl.call(opengl.glEnable, fan::opengl::GL_BLEND);
    opengl.call(opengl.glBlendFunc, fan::opengl::GL_SRC_ALPHA, fan::opengl::GL_ONE_MINUS_SRC_ALPHA);
  }
}

void fan::opengl::context_t::set_stencil_test(bool flag) {
  if (flag) {
    opengl.call(opengl.glEnable, fan::opengl::GL_STENCIL_TEST);
  }
  else {
    opengl.call(opengl.glDisable, fan::opengl::GL_STENCIL_TEST);
  }
}

void fan::opengl::context_t::set_stencil_op(GLenum sfail, GLenum dpfail, GLenum dppass) {
  opengl.call(opengl.glStencilOp, sfail, dpfail, dppass);
}

void fan::opengl::context_t::set_vsync(fan::window_t& window, bool flag) {
    glfwSwapInterval(flag);
}

void fan::opengl::context_t::message_callback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
  if (type == 33361 || type == 33360) { // gl_static_draw
    return;
  }
  //fan::print_no_space(type == GL_DEBUG_TYPE_ERROR ? "opengl error:" : "", type, ", severity:", severity, ", message:", message);
}

void fan::opengl::context_t::set_error_callback() {
  opengl.call(opengl.glEnable, GL_DEBUG_OUTPUT);
  opengl.call(opengl.glDebugMessageCallback, message_callback, (void*)0);
}

void fan::opengl::context_t::set_current(fan::window_t* window)
{
  if (window == nullptr) {
    glfwMakeContextCurrent(nullptr);
  }
  else {
    glfwMakeContextCurrent(window->glfw_window);
  }
}

//-----------------------------context-----------------------------
//-----------------------------context-----------------------------
//-----------------------------context-----------------------------
//-----------------------------context-----------------------------
//-----------------------------context-----------------------------
//-----------------------------context-----------------------------
//-----------------------------context-----------------------------
//-----------------------------context-----------------------------
//-----------------------------context-----------------------------
//-----------------------------context-----------------------------
//-----------------------------context-----------------------------
//-----------------------------context-----------------------------

//**************************************************************
//**************************************************************
//**************************************************************

//-----------------------------core-----------------------------
//-----------------------------core-----------------------------
//-----------------------------core-----------------------------
//-----------------------------core-----------------------------
//-----------------------------core-----------------------------
//-----------------------------core-----------------------------
//-----------------------------core-----------------------------
//-----------------------------core-----------------------------
//-----------------------------core-----------------------------
//-----------------------------core-----------------------------
//-----------------------------core-----------------------------
//-----------------------------core-----------------------------

int fan::opengl::core::get_buffer_size(fan::opengl::context_t& context, GLenum target_buffer, GLuint buffer_object) {
  int size = 0;

  context.opengl.call(context.opengl.glBindBuffer, target_buffer, buffer_object);
  context.opengl.call(context.opengl.glGetBufferParameteriv, target_buffer, fan::opengl::GL_BUFFER_SIZE, &size);

  return size;
}

void fan::opengl::core::write_glbuffer(fan::opengl::context_t& context, GLuint buffer, const void* data, uintptr_t size, uint32_t usage, GLenum target) {
  context.opengl.call(context.opengl.glBindBuffer, target, buffer);

  context.opengl.call(context.opengl.glBufferData, target, size, data, usage);
  /*if (target == GL_SHADER_STORAGE_BUFFER) {
  glBindBufferBase(target, location, buffer);
  }*/
}

void fan::opengl::core::get_glbuffer(fan::opengl::context_t& context, void* data, GLuint buffer_id, uintptr_t size, uintptr_t offset, GLenum target) {
  context.opengl.call(context.opengl.glBindBuffer, target, buffer_id);
  context.opengl.call(context.opengl.glGetBufferSubData, target, offset, size, data);
}

void fan::opengl::core::edit_glbuffer(fan::opengl::context_t& context, GLuint buffer, const void* data, uintptr_t offset, uintptr_t size, uintptr_t target) {
  context.opengl.call(context.opengl.glBindBuffer, target, buffer);

  #if fan_debug >= fan_debug_high

  int buffer_size = get_buffer_size(context, target, buffer);

  if ((buffer_size < (int)size) || (int)(offset + size) > buffer_size) {
    fan::throw_error("tried to write more than allocated");
  }

  #endif

  context.opengl.call(context.opengl.glBufferSubData, target, offset, size, data);
  /* if (target == GL_SHADER_STORAGE_BUFFER) {
  glBindBufferBase(target, location, buffer);
  }*/
}

int fan::opengl::core::get_bound_buffer(fan::opengl::context_t& context) {
  int buffer_id;
  context.opengl.call(context.opengl.glGetIntegerv, fan::opengl::GL_VERTEX_BINDING_BUFFER, &buffer_id);
  return buffer_id;
}

void fan::opengl::core::vao_t::open(fan::opengl::context_t& context) {
  context.opengl.call(context.opengl.glGenVertexArrays, 1, &m_buffer);
}

void fan::opengl::core::vao_t::close(fan::opengl::context_t& context) {
  context.opengl.call(context.opengl.glDeleteVertexArrays, 1, &m_buffer);
}

void fan::opengl::core::vao_t::bind(fan::opengl::context_t& context) const {
  context.opengl.call(context.opengl.glBindVertexArray, m_buffer);
}

void fan::opengl::core::vao_t::unbind(fan::opengl::context_t& context) const {
  context.opengl.call(context.opengl.glBindVertexArray, 0);
}

void fan::opengl::core::vbo_t::open(fan::opengl::context_t& context, GLenum target_) {
  context.opengl.call(context.opengl.glGenBuffers, 1, &m_buffer);
  m_target = target_;
}

void fan::opengl::core::vbo_t::close(fan::opengl::context_t& context) {
  #if fan_debug >= fan_debug_medium
  if (m_buffer == (GLuint)-1) {
    fan::throw_error("tried to remove non existent vbo");
  }
  #endif
  context.opengl.call(context.opengl.glDeleteBuffers, 1, &m_buffer);
}

void fan::opengl::core::vbo_t::bind(fan::opengl::context_t& context) const {
  context.opengl.call(context.opengl.glBindBuffer, m_target, m_buffer);
}

void fan::opengl::core::vbo_t::get_vram_instance(fan::opengl::context_t& context, void* data, uintptr_t size, uintptr_t offset) {
  fan::opengl::core::get_glbuffer(context, data, m_buffer, size, offset, m_target);
}

// only for target GL_UNIFORM_BUFFER

void fan::opengl::core::vbo_t::bind_buffer_range(fan::opengl::context_t& context, uint32_t total_size) {
  context.opengl.call(context.opengl.glBindBufferRange, fan::opengl::GL_UNIFORM_BUFFER, 0, m_buffer, 0, total_size);
}

void fan::opengl::core::vbo_t::edit_buffer(fan::opengl::context_t& context, const void* data, uintptr_t offset, uintptr_t size) {
  fan::opengl::core::edit_glbuffer(context, m_buffer, data, offset, size, m_target);
}

void fan::opengl::core::vbo_t::write_buffer(fan::opengl::context_t& context, const void* data, uintptr_t size) {
  fan::opengl::core::write_glbuffer(context, m_buffer, data, size, m_usage, m_target);
}

void fan::opengl::core::framebuffer_t::open(fan::opengl::context_t& context) {
  context.opengl.call(context.opengl.glGenFramebuffers, 1, &framebuffer);
}

void fan::opengl::core::framebuffer_t::close(fan::opengl::context_t& context) {
  context.opengl.call(context.opengl.glDeleteFramebuffers, 1, &framebuffer);
}

void fan::opengl::core::framebuffer_t::bind(fan::opengl::context_t& context) const {
  context.opengl.call(context.opengl.glBindFramebuffer, fan::opengl::GL_FRAMEBUFFER, framebuffer);
}

void fan::opengl::core::framebuffer_t::unbind(fan::opengl::context_t& context) const {
  context.opengl.call(context.opengl.glBindFramebuffer, fan::opengl::GL_FRAMEBUFFER, 0);
}

bool fan::opengl::core::framebuffer_t::ready(fan::opengl::context_t& context) const {
  return context.opengl.call(context.opengl.glCheckFramebufferStatus, fan::opengl::GL_FRAMEBUFFER) ==
    fan::opengl::GL_FRAMEBUFFER_COMPLETE;
}

void fan::opengl::core::framebuffer_t::bind_to_renderbuffer(fan::opengl::context_t& context, fan::opengl::GLenum renderbuffer, const properties_t& p) {
  bind(context);
  context.opengl.call(context.opengl.glFramebufferRenderbuffer, GL_FRAMEBUFFER, p.internalformat, GL_RENDERBUFFER, renderbuffer);
}

// texture must be binded with texture.bind();

void fan::opengl::core::framebuffer_t::bind_to_texture(fan::opengl::context_t& context, fan::opengl::GLuint texture, fan::opengl::GLenum attatchment) {
  context.opengl.call(context.opengl.glFramebufferTexture2D, GL_FRAMEBUFFER, attatchment, GL_TEXTURE_2D, texture, 0);
}

void fan::opengl::core::renderbuffer_t::open(fan::opengl::context_t& context) {
  context.opengl.call(context.opengl.glGenRenderbuffers, 1, &renderbuffer);
  //set_storage(context, p);
}

void fan::opengl::core::renderbuffer_t::close(fan::opengl::context_t& context) {
  context.opengl.call(context.opengl.glDeleteRenderbuffers, 1, &renderbuffer);
}

void fan::opengl::core::renderbuffer_t::bind(fan::opengl::context_t& context) const {
  context.opengl.call(context.opengl.glBindRenderbuffer, fan::opengl::GL_RENDERBUFFER, renderbuffer);
}

void fan::opengl::core::renderbuffer_t::set_storage(fan::opengl::context_t& context, const properties_t& p) const {
  bind(context);
  context.opengl.call(context.opengl.glRenderbufferStorage, fan::opengl::GL_RENDERBUFFER, p.internalformat, p.size.x, p.size.y);
}

void fan::opengl::core::renderbuffer_t::bind_to_renderbuffer(fan::opengl::context_t& context, const properties_t& p) {
  bind(context);
  context.opengl.call(context.opengl.glFramebufferRenderbuffer, GL_FRAMEBUFFER, p.internalformat, GL_RENDERBUFFER, renderbuffer);
}

//-----------------------------core-----------------------------
//-----------------------------core-----------------------------
//-----------------------------core-----------------------------
//-----------------------------core-----------------------------
//-----------------------------core-----------------------------
//-----------------------------core-----------------------------
//-----------------------------core-----------------------------
//-----------------------------core-----------------------------
//-----------------------------core-----------------------------
//-----------------------------core-----------------------------
//-----------------------------core-----------------------------
//-----------------------------core-----------------------------


fan::opengl::context_t::shader_nr_t fan::opengl::context_t::shader_create() {
  return shader_list.NewNode();
}

fan::opengl::context_t::shader_t& fan::opengl::context_t::shader_get(shader_nr_t nr) {
  return shader_list[nr];
}

void fan::opengl::context_t::shader_erase(shader_nr_t nr) {
  shader_t& shader = shader_get(nr);
  fan_validate_buffer(shader.id, {
     opengl.call(opengl.glValidateProgram, shader.id);
      int status = 0;
      opengl.call(opengl.glGetProgramiv, shader.id, fan::opengl::GL_VALIDATE_STATUS, &status);
      if (status) {
          opengl.call(opengl.glDeleteProgram, shader.id);
      }
      shader.id = fan::uninitialized;
  });
  shader_list.Recycle(nr);
}

void fan::opengl::context_t::shader_use(shader_nr_t nr) {
  shader_t& shader = shader_get(nr);
  if (shader.id == current_program) {
    return;
  }
  opengl.call(opengl.glUseProgram, shader.id);
  current_program = shader.id;
}

void fan::opengl::context_t::shader_set_vertex(shader_nr_t nr, const fan::string& vertex_code) {
  auto& shader = shader_get(nr);

  if (shader.vertex != (uint32_t)fan::uninitialized) {
    opengl.call(opengl.glDeleteShader, shader.vertex);
  }

  shader.vertex = opengl.call(opengl.glCreateShader, fan::opengl::GL_VERTEX_SHADER);
  shader.svertex = vertex_code;

  char* ptr = (char*)vertex_code.c_str();
  fan::opengl::GLint length = vertex_code.size();

  opengl.call(opengl.glShaderSource, shader.vertex, 1, &ptr, &length);
  opengl.call(opengl.glCompileShader, shader.vertex);

  shader_check_compile_errors(shader, "VERTEX");
}

void fan::opengl::context_t::shader_set_fragment(shader_nr_t nr, const fan::string& fragment_code) {
  shader_t& shader = shader_get(nr);

  if (shader.fragment != (uint32_t)-1) {
    opengl.call(opengl.glDeleteShader, shader.fragment);
  }

  shader.fragment = opengl.call(opengl.glCreateShader, fan::opengl::GL_FRAGMENT_SHADER);
  shader.sfragment = fragment_code;

  char* ptr = (char*)fragment_code.c_str();
  fan::opengl::GLint length = fragment_code.size();

  opengl.call(opengl.glShaderSource, shader.fragment, 1, &ptr, &length);

  opengl.call(opengl.glCompileShader, shader.fragment);
  shader_check_compile_errors(shader, "FRAGMENT");
}

bool fan::opengl::context_t::shader_compile(shader_nr_t nr) {
  shader_t& shader = shader_get(nr);

  auto temp_id = opengl.call(opengl.glCreateProgram);
  if (shader.vertex != (uint32_t)-1) {
    opengl.call(opengl.glAttachShader, temp_id, shader.vertex);
  }
  if (shader.fragment != (uint32_t)-1) {
    opengl.call(opengl.glAttachShader, temp_id, shader.fragment);
  }

  opengl.call(opengl.glLinkProgram, temp_id);
  bool ret = shader_check_compile_errors(temp_id, "PROGRAM");

  if (shader.vertex != (uint32_t)-1) {
    opengl.call(opengl.glDeleteShader, shader.vertex);
    shader.vertex = -1;
  }
  if (shader.fragment != (uint32_t)-1) {
    opengl.call(opengl.glDeleteShader, shader.fragment);
    shader.fragment = -1;
  }

  if (ret == false) {
    return ret;
  }

  if (shader.id != (uint32_t)-1) {
    opengl.call(opengl.glDeleteProgram, shader.id);
  }
  shader.id = temp_id;

  shader.projection_view[0] = opengl.call(opengl.glGetUniformLocation, shader.id, "projection");
  shader.projection_view[1] = opengl.call(opengl.glGetUniformLocation, shader.id, "view");

  std::regex uniformRegex(R"(uniform\s+(\w+)\s+(\w+)(\s*=\s*[\d\.]+)?;)");

  // Read vertex shader source code
  fan::string vertexData = shader.svertex;

  // Extract uniforms from vertex shader
  std::smatch match;
  while (std::regex_search(vertexData, match, uniformRegex)) {
      shader.uniform_type_table[match[2]] = match[1];
      vertexData = match.suffix().str();
  }

  // Read fragment shader source code
  fan::string fragmentData = shader.sfragment;

  // Extract uniforms from fragment shader
  while (std::regex_search(fragmentData, match, uniformRegex)) {
      shader.uniform_type_table[match[2]] = match[1];
      fragmentData = match.suffix().str();
  }




  return ret;
}

bool fan::opengl::context_t::shader_check_compile_errors(fan::opengl::GLuint shader, const fan::string& type)
{
  fan::opengl::GLint success;

  bool program = type == "PROGRAM";

  if (program == false) {
    opengl.call(opengl.glGetShaderiv, shader, fan::opengl::GL_COMPILE_STATUS, &success);
  }
  else {
    opengl.call(opengl.glGetProgramiv, shader, fan::opengl::GL_LINK_STATUS, &success);
  }

  if (success) {
    return true;
  }

  int buffer_size = 0;
  opengl.glGetShaderiv(shader, fan::opengl::GL_INFO_LOG_LENGTH, &buffer_size);


  if (buffer_size <= 0) {
    return false;
  }

  fan::string buffer;
  buffer.resize(buffer_size);

  if (!success)
  {
    int test;
#define get_info_log(is_program, program, str_buffer, size) \
                if (is_program) \
                opengl.call(opengl.glGetProgramInfoLog, program, size, nullptr, buffer.data()); \
                else \
                opengl.call(opengl.glGetShaderInfoLog, program, size, &test, buffer.data());

    get_info_log(program, shader, buffer, buffer_size);

    fan::print("failed to compile: " + type, buffer);

    return false;
  }
  return true;
}

bool fan::opengl::context_t::shader_check_compile_errors(fan::opengl::context_t::shader_t& shader, const fan::string& type)
{
  fan::opengl::GLint success;

  bool vertex = type == "VERTEX";
  bool program = type == "PROGRAM";

  if (program == false) {
    opengl.call(opengl.glGetShaderiv, vertex ? shader.vertex : shader.fragment, fan::opengl::GL_COMPILE_STATUS, &success);
  }
  else {
    opengl.call(opengl.glGetProgramiv, vertex ? shader.vertex : shader.fragment, fan::opengl::GL_LINK_STATUS, &success);
  }

  if (success) {
    return true;
  }

  int buffer_size = 0;
  opengl.glGetShaderiv(vertex ? shader.vertex : shader.fragment, fan::opengl::GL_INFO_LOG_LENGTH, &buffer_size);


  if (buffer_size <= 0) {
    return false;
  }

  fan::string buffer;
  buffer.resize(buffer_size);

  if (!success)
  {
    int test;
#define get_info_log(is_program, program, str_buffer, size) \
                if (is_program) \
                opengl.call(opengl.glGetProgramInfoLog, program, size, nullptr, buffer.data()); \
                else \
                opengl.call(opengl.glGetShaderInfoLog, program, size, &test, buffer.data());

    get_info_log(program, vertex ? shader.vertex : shader.fragment, buffer, buffer_size);

    fan::print("failed to compile: " + type, "filenames", shader.svertex, shader.sfragment, buffer);

    return false;
  }
  return true;
}

void fan::opengl::context_t::shader_set_camera(shader_nr_t nr, void* camera_nr) {
  camera_t& camera = camera_get(**(camera_nr_t**)&camera_nr);
  opengl.call(opengl.glUniformMatrix4fv, shader_get(nr).projection_view[0], 1, fan::opengl::GL_FALSE, &camera.m_projection[0][0]);
  opengl.call(opengl.glUniformMatrix4fv, shader_get(nr).projection_view[1], 1, fan::opengl::GL_FALSE, &camera.m_view[0][0]);
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

fan::opengl::context_t::image_nr_t fan::opengl::context_t::image_create() {
  image_nr_t texture_reference = image_list.NewNode();
  //gloco->image_list[texture_reference].image = this;
  opengl.call(opengl.glGenTextures, 1, &image_get(texture_reference));
  return texture_reference;
}

fan::opengl::GLuint& fan::opengl::context_t::image_get(image_nr_t nr) {
  return image_list[nr].texture_id;
}

fan::opengl::context_t::image_t& fan::opengl::context_t::image_get_data(image_nr_t nr) {
  return image_list[nr];
}

void fan::opengl::context_t::image_erase(image_nr_t nr) {
  opengl.glDeleteTextures(1, &image_get(nr));
  image_list.Recycle(nr);
}

void fan::opengl::context_t::image_bind(image_nr_t nr) {
  opengl.call(opengl.glBindTexture, fan::opengl::GL_TEXTURE_2D, image_get(nr));
}

void fan::opengl::context_t::image_unbind(image_nr_t nr) {
  opengl.call(opengl.glBindTexture, fan::opengl::GL_TEXTURE_2D, 0);
}

void fan::opengl::context_t::image_set_settings(const image_load_properties_t& p) {
  opengl.call(opengl.glTexParameteri, fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_WRAP_S, p.visual_output);
  opengl.call(opengl.glTexParameteri, fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_WRAP_T, p.visual_output);
  opengl.call(opengl.glTexParameteri, fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_MIN_FILTER, p.min_filter);
  opengl.call(opengl.glTexParameteri, fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_MAG_FILTER, p.mag_filter);
}

fan::opengl::context_t::image_nr_t fan::opengl::context_t::image_load(const fan::image::image_info_t& image_info) {
  return image_load(image_info, image_load_properties_t());
}

fan::opengl::context_t::image_nr_t fan::opengl::context_t::image_load(const fan::image::image_info_t& image_info, const image_load_properties_t& p) {

  image_nr_t nr = image_create();
  image_bind(nr);

  image_set_settings(p);

  image_t& image = image_get_data(nr);
  image.size = image_info.size;

  opengl.call(opengl.glTexImage2D, fan::opengl::GL_TEXTURE_2D, 0, p.internal_format, image.size.x, image.size.y, 0, p.format, p.type, image_info.data);

  switch (p.min_filter) {
  case fan::opengl::GL_LINEAR_MIPMAP_LINEAR:
  case fan::opengl::GL_NEAREST_MIPMAP_LINEAR:
  case fan::opengl::GL_LINEAR_MIPMAP_NEAREST:
  case fan::opengl::GL_NEAREST_MIPMAP_NEAREST: {
    opengl.call(opengl.glGenerateMipmap, fan::opengl::GL_TEXTURE_2D);
    break;
  }
  }

  return nr;
}

fan::opengl::context_t::image_nr_t fan::opengl::context_t::image_load(const fan::string& path) {
  return image_load(path, image_load_properties_t());
}

fan::opengl::context_t::image_nr_t fan::opengl::context_t::image_load(const fan::string& path, const image_load_properties_t& p) {

#if fan_assert_if_same_path_loaded_multiple_times

  static std::unordered_map<fan::string, bool> existing_images;

  if (existing_images.find(path) != existing_images.end()) {
    fan::throw_error("image already existing " + path);
  }

  existing_images[path] = 0;

#endif

  fan::image::image_info_t image_info;
  if (fan::image::load(path, &image_info)) {
    return create_missing_texture();
  }
  image_nr_t nr = image_load(image_info, p);
  fan::image::free(&image_info);
  return nr;
}

fan::opengl::context_t::image_nr_t fan::opengl::context_t::image_load(fan::color* colors, const fan::vec2ui& size_) {
  return image_load(colors, size_, image_load_properties_t());
}

fan::opengl::context_t::image_nr_t fan::opengl::context_t::image_load(fan::color* colors, const fan::vec2ui& size_, const image_load_properties_t& p) {

  image_nr_t nr = image_create();
  image_bind(nr);

  image_set_settings(p);

  image_t& image = image_get_data(nr);
  image.size = size_;

  opengl.call(opengl.glTexImage2D, fan::opengl::GL_TEXTURE_2D, 0, fan::opengl::GL_RGBA32F, image.size.x, image.size.y, 0, p.format, fan::opengl::GL_FLOAT, (uint8_t*)colors);

  return nr;
}

void fan::opengl::context_t::image_unload(image_nr_t nr) {
  image_erase(nr);
}

fan::opengl::context_t::image_nr_t fan::opengl::context_t::create_missing_texture() {
  image_load_properties_t p;

  p.visual_output = fan::opengl::GL_REPEAT;

  image_nr_t nr = image_create();
  image_bind(nr);

  image_set_settings(p);
  image_t& image = image_get_data(nr);
  image.size = fan::vec2i(2, 2);

  uint8_t pixels[2*2*4];

  uint32_t pixel = 0;

  pixels[pixel++] = 0;
  pixels[pixel++] = 0;
  pixels[pixel++] = 0;
  pixels[pixel++] = 255;

  pixels[pixel++] = 255;
  pixels[pixel++] = 0;
  pixels[pixel++] = 220;
  pixels[pixel++] = 255;

  pixels[pixel++] = 255;
  pixels[pixel++] = 0;
  pixels[pixel++] = 220;
  pixels[pixel++] = 255;

  pixels[pixel++] = 0;
  pixels[pixel++] = 0;
  pixels[pixel++] = 0;
  pixels[pixel++] = 255;

  opengl.call(opengl.glTexImage2D, fan::opengl::GL_TEXTURE_2D, 0, p.internal_format, image.size.x, image.size.y, 0, p.format, p.type, pixels);

  opengl.call(opengl.glGenerateMipmap, fan::opengl::GL_TEXTURE_2D);

  return nr;
}

fan::opengl::context_t::image_nr_t fan::opengl::context_t::create_transparent_texture() {
    image_load_properties_t p;

    uint8_t* pixels = (uint8_t*)malloc(sizeof(uint8_t) * (2 * 2 * fan::color::size()));
    uint32_t pixel = 0;

    pixels[pixel++] = 60;
    pixels[pixel++] = 60;
    pixels[pixel++] = 60;
    pixels[pixel++] = 255;

    pixels[pixel++] = 40;
    pixels[pixel++] = 40;
    pixels[pixel++] = 40;
    pixels[pixel++] = 255;

    pixels[pixel++] = 40;
    pixels[pixel++] = 40;
    pixels[pixel++] = 40;
    pixels[pixel++] = 255;

    pixels[pixel++] = 60;
    pixels[pixel++] = 60;
    pixels[pixel++] = 60;
    pixels[pixel++] = 255;

    p.visual_output = fan::opengl::GL_REPEAT;

    image_nr_t nr = image_create();
    image_bind(nr);

    auto& img = image_get_data(nr);

    image_set_settings(p);

    img.size = fan::vec2i(2, 2);

    opengl.call(opengl.glTexImage2D, fan::opengl::GL_TEXTURE_2D, 0, p.internal_format, 2, 2, 0, p.format, p.type, pixels);

    free(pixels);

    opengl.call(opengl.glGenerateMipmap, fan::opengl::GL_TEXTURE_2D);
    return nr;
}

void fan::opengl::context_t::image_reload_pixels(image_nr_t nr, const fan::image::image_info_t& image_info) {
  image_reload_pixels(nr, image_info, image_load_properties_t());
}

void fan::opengl::context_t::image_reload_pixels(image_nr_t nr, const fan::image::image_info_t& image_info, const image_load_properties_t& p) {

  image_bind(nr);

  image_set_settings(p);

  image_t& image = image_get_data(nr);
  image.size = image_info.size;
  opengl.call(opengl.glTexImage2D, fan::opengl::GL_TEXTURE_2D, 0, p.internal_format, image.size.x, image.size.y, 0, p.format, p.type, image_info.data);
}

std::unique_ptr<uint8_t[]> fan::opengl::context_t::image_get_pixel_data(image_nr_t nr, fan::opengl::GLenum format, fan::vec2 uvp, fan::vec2 uvs) {
  image_t& image = image_get_data(nr);
  image_bind(nr);

  fan::vec2ui uv_size = {
      (uint32_t)(image.size.x * uvs.x),
      (uint32_t)(image.size.y * uvs.y)
  };

  auto full_ptr = std::make_unique<uint8_t[]>(image.size.x * image.size.y * 4); // assuming rgba

  opengl.call(
    opengl.glGetTexImage,
    fan::opengl::GL_TEXTURE_2D,
    0,
    format,
    fan::opengl::GL_UNSIGNED_BYTE,
    full_ptr.get()
  );

  auto ptr = std::make_unique<uint8_t[]>(uv_size.x * uv_size.y * 4); // assuming rgba

  for (uint32_t y = 0; y < uv_size.y; ++y) {
    for (uint32_t x = 0; x < uv_size.x; ++x) {
      uint32_t full_index = ((y + uvp.y * image.size.y) * image.size.x + (x + uvp.x * image.size.x)) * 4;
      uint32_t index = (y * uv_size.x + x) * 4;
      ptr[index + 0] = full_ptr[full_index + 0];
      ptr[index + 1] = full_ptr[full_index + 1];
      ptr[index + 2] = full_ptr[full_index + 2];
      ptr[index + 3] = full_ptr[full_index + 3];
    }
  }

  return ptr;
}

fan::opengl::context_t::image_nr_t fan::opengl::context_t::create_image(const fan::color& color)
{
  return create_image(color, image_load_properties_t());
}

// creates single colored text size.x*size.y sized
fan::opengl::context_t::image_nr_t fan::opengl::context_t::create_image(const fan::color& color, const fan::opengl::context_t::image_load_properties_t& p) {

  uint8_t pixels[4];
  for (uint32_t p = 0; p < fan::color::size(); p++) {
    pixels[p] = color[p] * 255;
  }

  image_nr_t nr = image_create();
  image_bind(nr);

  image_set_settings(p);

  opengl.call(opengl.glTexImage2D, fan::opengl::GL_TEXTURE_2D, 0, p.internal_format, 1, 1, 0, p.format, p.type, pixels);

  image_t& image = image_get_data(nr);
  image.size = 1;

  opengl.call(opengl.glGenerateMipmap, fan::opengl::GL_TEXTURE_2D);

  return nr;
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


fan::opengl::context_t::camera_nr_t fan::opengl::context_t::camera_create()
{
  return camera_list.NewNode();
}

fan::opengl::context_t::camera_t& fan::opengl::context_t::camera_get(camera_nr_t nr) {
  return camera_list[nr];
}

void fan::opengl::context_t::camera_erase(camera_nr_t nr) {
  camera_list.Recycle(nr);
}

fan::opengl::context_t::camera_nr_t fan::opengl::context_t::camera_open(const fan::vec2& x, const fan::vec2& y) {
  camera_nr_t nr = camera_create();
  camera_set_ortho(nr, fan::vec2(x.x, x.y), fan::vec2(y.x, y.y));
  return nr;
}

fan::vec3 fan::opengl::context_t::camera_get_position(camera_nr_t nr) {
  return camera_get(nr).position;
}

void fan::opengl::context_t::camera_set_position(camera_nr_t nr, const fan::vec3& cp) {
  camera_t& camera = camera_get(nr);
  camera.position = cp;


  camera.m_view[3][0] = 0;
  camera.m_view[3][1] = 0;
  camera.m_view[3][2] = 0;
  camera.m_view = camera.m_view.translate(camera.position);
  fan::vec3 position = camera.m_view.get_translation();
  constexpr fan::vec3 front(0, 0, 1);

  camera.m_view = fan::math::look_at_left<fan::mat4, fan::vec3>(position, position + front, fan::camera::world_up);
}

fan::vec2 fan::opengl::context_t::camera_get_size(camera_nr_t nr) {
  camera_t& camera = camera_get(nr);
  return fan::vec2(std::abs(camera.coordinates.right - camera.coordinates.left), std::abs(camera.coordinates.down - camera.coordinates.up));
}

void fan::opengl::context_t::camera_set_ortho(camera_nr_t nr, fan::vec2 x, fan::vec2 y) {
  camera_t& camera = camera_get(nr);

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
    znearfar
  );

  camera.m_view[3][0] = 0;
  camera.m_view[3][1] = 0;
  camera.m_view[3][2] = 0;
  camera.m_view = camera.m_view.translate(camera.position);
  fan::vec3 position = camera.m_view.get_translation();
  constexpr fan::vec3 front(0, 0, 1);

  camera.m_view = fan::math::look_at_left<fan::mat4, fan::vec3>(position, position + front, fan::camera::world_up);

  //auto it = gloco->m_viewport_resize_callback.GetNodeFirst();

  //while (it != gloco->m_viewport_resize_callback.dst) {

  //  gloco->m_viewport_resize_callback.StartSafeNext(it);

  //  resize_cb_data_t cbd;
  //  cbd.camera = this;
  //  cbd.position = get_position();
  //  cbd.size = get_camera_size();
  //  gloco->m_viewport_resize_callback[it].data(cbd);

  //  it = gloco->m_viewport_resize_callback.EndSafeNext();
  //}
}

void fan::opengl::context_t::camera_set_perspective(camera_nr_t nr, f32_t fov, const fan::vec2& window_size) {
  camera_t& camera = camera_get(nr);

  camera.m_projection = fan::math::perspective<fan::mat4>(fan::math::radians(fov), (f32_t)window_size.x / (f32_t)window_size.y, 0.1f, 1000.0f);

  camera.update_view();

  camera.m_view = camera.get_view_matrix();

  //auto it = gloco->m_viewport_resize_callback.GetNodeFirst();

  //while (it != gloco->m_viewport_resize_callback.dst) {

  //  gloco->m_viewport_resize_callback.StartSafeNext(it);

  //  resize_cb_data_t cbd;
  //  cbd.camera = this;
  //  cbd.position = get_position();
  //  cbd.size = get_camera_size();
  //  gloco->m_viewport_resize_callback[it].data(cbd);

  //  it = gloco->m_viewport_resize_callback.EndSafeNext();
  //}
}

void fan::opengl::context_t::camera_rotate(camera_nr_t nr, const fan::vec2& offset) {
  camera_t& camera = camera_get(nr);
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


fan::opengl::context_t::viewport_nr_t fan::opengl::context_t::viewport_create()
{
  auto nr = viewport_list.NewNode();

  viewport_set(
    nr,
    0, 0, 0
  );
  return nr;
}

fan::opengl::context_t::viewport_t& fan::opengl::context_t::viewport_get(viewport_nr_t nr) {
  return viewport_list[nr];
}

void fan::opengl::context_t::viewport_erase(viewport_nr_t nr) {
  viewport_list.Recycle(nr);
}

fan::vec2 fan::opengl::context_t::viewport_get_position(viewport_nr_t nr) {
  return viewport_list[nr].viewport_position;
}

fan::vec2 fan::opengl::context_t::viewport_get_size(viewport_nr_t nr) {
  return viewport_list[nr].viewport_size;
}


void fan::opengl::context_t::viewport_set(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
  opengl.call(
    opengl.glViewport,
    viewport_position_.x, window_size.y - viewport_size_.y - viewport_position_.y,
    viewport_size_.x, viewport_size_.y
  );
}

void fan::opengl::context_t::viewport_set(viewport_nr_t nr, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
  viewport_t& viewport = viewport_get(nr);
  viewport.viewport_position = viewport_position_;
  viewport.viewport_size = viewport_size_;

  viewport_set(viewport_position_, viewport_size_, window_size);
}

void fan::opengl::context_t::viewport_zero(viewport_nr_t nr) {
  viewport_t& viewport = viewport_get(nr);
  viewport.viewport_position = 0;
  viewport.viewport_size = 0;
  opengl.call(opengl.glViewport, 0, 0, 0, 0);
}

bool fan::opengl::context_t::inside(viewport_nr_t nr, const fan::vec2& position) {
  viewport_t& viewport = viewport_get(nr);
  return fan_2d::collision::rectangle::point_inside_no_rotation(position, viewport.viewport_position + viewport.viewport_size / 2, viewport.viewport_size / 2);
}

bool fan::opengl::context_t::inside_wir(viewport_nr_t nr, const fan::vec2& position) {
  viewport_t& viewport = viewport_get(nr);
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