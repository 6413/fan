#include "core.h"

#include <fan/physics/collision/rectangle.h>
// for parsing uniform values
#include <regex>

#include <fan/graphics/stb.h>

#include <fan/window/window.h>

using namespace fan::graphics;

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

//std::unordered_map<size_t, int> fan::opengl::shader_location_cache;

void fan::opengl::context_t::print_version() {
  fan::print("opengl version supported:", fan_opengl_call(glGetString(GL_VERSION)));
}

void fan::opengl::context_t::open(const properties_t&) {
  opengl.open();
}
void fan::opengl::context_t::render(fan::window_t& window) {
  glfwSwapBuffers(window.glfw_window);
}

void fan::opengl::context_t::set_depth_test(bool flag) {
  if (flag) {
    fan_opengl_call(glEnable(GL_DEPTH_TEST));
  }
  else {
    fan_opengl_call(glDisable(GL_DEPTH_TEST));
  }
}

void fan::opengl::context_t::set_blending(bool flag) {
  if (flag) {
    fan_opengl_call(glDisable(GL_BLEND));
  }
  else {
    fan_opengl_call(glEnable(GL_BLEND));
    fan_opengl_call(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
  }
}

void fan::opengl::context_t::set_stencil_test(bool flag) {
  if (flag) {
    fan_opengl_call(glEnable(GL_STENCIL_TEST));
  }
  else {
    fan_opengl_call(glDisable(GL_STENCIL_TEST));
  }
}

void fan::opengl::context_t::set_stencil_op(GLenum sfail, GLenum dpfail, GLenum dppass) {
  fan_opengl_call(glStencilOp(sfail, dpfail, dppass));
}

void fan::opengl::context_t::set_vsync(fan::window_t* window, bool flag) {
  glfwSwapInterval(flag);
}

void fan::opengl::context_t::message_callback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
  if (type == 33361 || type == 33360) { // gl_static_draw
    return;
  }
  fan::print_no_space(type == GL_DEBUG_TYPE_ERROR ? "opengl error:" : "", type, ", severity:", severity, ", message:", message);
}

void fan::opengl::context_t::set_error_callback() {
  fan_opengl_call(glEnable(GL_DEBUG_OUTPUT));
  fan_opengl_call(glDebugMessageCallback(message_callback, (void*)0));
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

  fan_opengl_call(glBindBuffer(target_buffer, buffer_object));
  fan_opengl_call(glGetBufferParameteriv(target_buffer, GL_BUFFER_SIZE, &size));

  return size;
}

void fan::opengl::core::write_glbuffer(fan::opengl::context_t& context, GLuint buffer, const void* data, uintptr_t size, uint32_t usage, GLenum target) {
  fan_opengl_call(glBindBuffer(target, buffer));

  fan_opengl_call(glBufferData(target, size, data, usage));
  /*if (target == GL_SHADER_STORAGE_BUFFER) {
  glBindBufferBase(target, location, buffer);
  }*/
}

void fan::opengl::core::get_glbuffer(fan::opengl::context_t& context, void* data, GLuint buffer_id, uintptr_t size, uintptr_t offset, GLenum target) {
  fan_opengl_call(glBindBuffer(target, buffer_id));
  fan_opengl_call(glGetBufferSubData(target, offset, size, data));
}

void fan::opengl::core::edit_glbuffer(fan::opengl::context_t& context, GLuint buffer, const void* data, uintptr_t offset, uintptr_t size, uintptr_t target) {
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

int fan::opengl::core::get_bound_buffer(fan::opengl::context_t& context) {
  int buffer_id;
  fan_opengl_call(glGetIntegerv(GL_VERTEX_BINDING_BUFFER, &buffer_id));
  return buffer_id;
}

void fan::opengl::core::vao_t::open(fan::opengl::context_t& context) {
  fan_opengl_call(glGenVertexArrays(1, &m_buffer));
}

void fan::opengl::core::vao_t::close(fan::opengl::context_t& context) {
  if (m_buffer == (decltype(m_buffer))-1) {
    return;
  }
  fan_opengl_call(glDeleteVertexArrays(1, &m_buffer));
}

void fan::opengl::core::vao_t::bind(fan::opengl::context_t& context) const {
  fan_opengl_call(glBindVertexArray(m_buffer));
}

void fan::opengl::core::vao_t::unbind(fan::opengl::context_t& context) const {
  fan_opengl_call(glBindVertexArray(0));
}

void fan::opengl::core::vbo_t::open(fan::opengl::context_t& context, GLenum target_) {
  fan_opengl_call(glGenBuffers(1, &m_buffer));
  m_target = target_;
}

void fan::opengl::core::vbo_t::close(fan::opengl::context_t& context) {
  #if fan_debug >= fan_debug_medium
  if (m_buffer == (GLuint)-1) {
    return;
    //fan::throw_error("tried to remove non existent vbo");
  }
  #endif
  fan_opengl_call(glDeleteBuffers(1, &m_buffer));
}

void fan::opengl::core::vbo_t::bind(fan::opengl::context_t& context) const {
  fan_opengl_call(glBindBuffer(m_target, m_buffer));
}

void fan::opengl::core::vbo_t::get_vram_instance(fan::opengl::context_t& context, void* data, uintptr_t size, uintptr_t offset) {
  fan::opengl::core::get_glbuffer(context, data, m_buffer, size, offset, m_target);
}

// only for target GL_UNIFORM_BUFFER

void fan::opengl::core::vbo_t::bind_buffer_range(fan::opengl::context_t& context, uint32_t total_size) {
  fan_opengl_call(glBindBufferRange(GL_UNIFORM_BUFFER, 0, m_buffer, 0, total_size));
}

void fan::opengl::core::vbo_t::edit_buffer(fan::opengl::context_t& context, const void* data, uintptr_t offset, uintptr_t size) {
  fan::opengl::core::edit_glbuffer(context, m_buffer, data, offset, size, m_target);
}

void fan::opengl::core::vbo_t::write_buffer(fan::opengl::context_t& context, const void* data, uintptr_t size) {
  fan::opengl::core::write_glbuffer(context, m_buffer, data, size, m_usage, m_target);
}

void fan::opengl::core::framebuffer_t::open(fan::opengl::context_t& context) {
  fan_opengl_call(glGenFramebuffers(1, &framebuffer));
}

void fan::opengl::core::framebuffer_t::close(fan::opengl::context_t& context) {
  fan_opengl_call(glDeleteFramebuffers(1, &framebuffer));
}

void fan::opengl::core::framebuffer_t::bind(fan::opengl::context_t& context) const {
  fan_opengl_call(glBindFramebuffer(GL_FRAMEBUFFER, framebuffer));
}

void fan::opengl::core::framebuffer_t::unbind(fan::opengl::context_t& context) const {
  fan_opengl_call(glBindFramebuffer(GL_FRAMEBUFFER, 0));
}

bool fan::opengl::core::framebuffer_t::ready(fan::opengl::context_t& context) const {
  return fan_opengl_call(glCheckFramebufferStatus(GL_FRAMEBUFFER)) ==
    GL_FRAMEBUFFER_COMPLETE;
}

void fan::opengl::core::framebuffer_t::bind_to_renderbuffer(fan::opengl::context_t& context, GLenum renderbuffer, const properties_t& p) {
  bind(context);
  fan_opengl_call(glFramebufferRenderbuffer(GL_FRAMEBUFFER, p.internalformat, GL_RENDERBUFFER, renderbuffer));
}

// texture must be binded with texture.bind();

void fan::opengl::core::framebuffer_t::bind_to_texture(fan::opengl::context_t& context, GLuint texture, GLenum attatchment) {
  fan_opengl_call(glFramebufferTexture2D(GL_FRAMEBUFFER, attatchment, GL_TEXTURE_2D, texture, 0));
}

void fan::opengl::core::renderbuffer_t::open(fan::opengl::context_t& context) {
  fan_opengl_call(glGenRenderbuffers(1, &renderbuffer));
  //set_storage(context, p);
}

void fan::opengl::core::renderbuffer_t::close(fan::opengl::context_t& context) {
  fan_opengl_call(glDeleteRenderbuffers(1, &renderbuffer));
}

void fan::opengl::core::renderbuffer_t::bind(fan::opengl::context_t& context) const {
  fan_opengl_call(glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer));
}

void fan::opengl::core::renderbuffer_t::set_storage(fan::opengl::context_t& context, const properties_t& p) const {
  bind(context);
  fan_opengl_call(glRenderbufferStorage(GL_RENDERBUFFER, p.internalformat, p.size.x, p.size.y));
}

void fan::opengl::core::renderbuffer_t::bind_to_renderbuffer(fan::opengl::context_t& context, const properties_t& p) {
  bind(context);
  fan_opengl_call(glFramebufferRenderbuffer(GL_FRAMEBUFFER, p.internalformat, GL_RENDERBUFFER, renderbuffer));
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


fan::graphics::shader_nr_t shader_create(fan::opengl::context_t& context) {
  auto nr = shader_list.NewNode();
  shader_list[nr].internal = new fan::opengl::context_t::shader_t;
  return nr;
}

fan::opengl::context_t::shader_t& shader_get(fan::opengl::context_t& context, shader_nr_t nr) {
  return *(fan::opengl::context_t::shader_t*)shader_list[nr].internal;
}

void shader_erase(fan::opengl::context_t& context, shader_nr_t nr) {
  auto& shader = shader_get(context, nr);
  fan_validate_buffer(shader.id, {
    fan_opengl_call(glValidateProgram(shader.id));
    int status = 0;
    fan_opengl_call(glGetProgramiv(shader.id, GL_VALIDATE_STATUS, &status));
    if (status) {
        fan_opengl_call(glDeleteProgram(shader.id));
    }
    shader.id = fan::uninitialized;
  });
  delete static_cast<fan::opengl::context_t::shader_t*>(shader_list[nr].internal);
  shader_list.Recycle(nr);
}

bool shader_check_compile_errors(GLuint shader, const fan::string& type)
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

  fan::string buffer;
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

bool shader_check_compile_errors(fan::graphics::shader_data_t& common_shader, const fan::string& type)
{
  fan::opengl::context_t::shader_t& shader = *(fan::opengl::context_t::shader_t*)common_shader.internal;
  GLint success;

  bool vertex = type == "VERTEX";
  bool program = type == "PROGRAM";

  if (program == false) {
    fan_opengl_call(glGetShaderiv(vertex ? shader.vertex : shader.fragment, GL_COMPILE_STATUS, &success));
  }
  else {
    fan_opengl_call(glGetProgramiv(vertex ? shader.vertex : shader.fragment, GL_LINK_STATUS, &success));
  }

  if (success) {
    return true;
  }

  int buffer_size = 0;
  fan_opengl_call(glGetShaderiv(vertex ? shader.vertex : shader.fragment, GL_INFO_LOG_LENGTH, &buffer_size));


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
                fan_opengl_call(glGetProgramInfoLog(program, size, nullptr, buffer.data())); \
                else \
                fan_opengl_call(glGetShaderInfoLog(program, size, &test, buffer.data()));

    get_info_log(program, vertex ? shader.vertex : shader.fragment, buffer, buffer_size);

    fan::print("failed to compile: " + type, "filenames", common_shader.svertex, common_shader.sfragment, buffer);

    return false;
  }
  return true;
}

void shader_use(fan::opengl::context_t& context, shader_nr_t nr) {
  auto& shader = shader_get(context, nr);
  if (shader.id == context.current_program) {
    return;
  }
  fan_opengl_call(glUseProgram(shader.id));
  context.current_program = shader.id;
}

template<typename T>
void fan::opengl::context_t::shader_set_value(shader_nr_t nr, const fan::string& name, const T& val) {
  static_assert(!std::is_same_v<T, uint8_t>, "only 4 byte supported");
  static_assert(!std::is_same_v<T, uint16_t>, "only 4 byte supported");
  static_assert(std::is_same_v<T, bool> == false || !std::is_same_v<T, int>, "only 4 byte supported");
  static_assert(std::is_same_v<T, double> == false, "only 4 byte supported");
  uint8_t value[sizeof(T)];
  for (uint32_t i = 0; i < sizeof(T); ++i) {
    value[i] = ((uint8_t*)&val)[i];
  }
  shader_use(*this, nr);
  shader_t& shader = shader_get(*this, nr);
  auto& context = *this;
  auto found = shader_list[nr].uniform_type_table.find(name);
  if (found == shader_list[nr].uniform_type_table.end()) {
    //fan::print("failed to set uniform value");
    return;
    //fan::throw_error("failed to set uniform value");
  }

  size_t hash0 = std::hash<std::string>{}(name);
  size_t hash1 = std::hash<decltype(shader_nr_t::NRI)>{}(nr.NRI);
  auto shader_loc_it = shader_location_cache.find(hash0 ^ hash1);
  if (shader_loc_it == shader_location_cache.end()) {
    GLint location = fan_opengl_call(glGetUniformLocation(shader.id, name.c_str()));
    if (location == -1) {
      return;
    }
    shader_loc_it = shader_location_cache.emplace(hash0 ^ hash1, location).first;
  }
  GLint location = shader_loc_it->second;


#if fan_debug >= fan_debug_insanity
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

template void fan::opengl::context_t::shader_set_value<fan::vec2>(shader_nr_t nr, const fan::string& name, const fan::vec2& val);
template void fan::opengl::context_t::shader_set_value<fan::vec3>(shader_nr_t nr, const fan::string& name, const fan::vec3& val);
template void fan::opengl::context_t::shader_set_value<fan::vec4>(shader_nr_t nr, const fan::string& name, const fan::vec4& val);
template void fan::opengl::context_t::shader_set_value<fan::mat4>(shader_nr_t nr, const fan::string& name, const fan::mat4& val);
template void fan::opengl::context_t::shader_set_value<fan::color>(shader_nr_t nr, const fan::string& name, const fan::color& val);
template void fan::opengl::context_t::shader_set_value<uint32_t>(shader_nr_t nr, const fan::string& name, const uint32_t& val);
template void fan::opengl::context_t::shader_set_value<uint64_t>(shader_nr_t nr, const fan::string& name, const uint64_t& val);
template void fan::opengl::context_t::shader_set_value<int>(shader_nr_t nr, const fan::string& name, const int& val);
template void fan::opengl::context_t::shader_set_value<f32_t>(shader_nr_t nr, const fan::string& name, const f32_t& val);
template void fan::opengl::context_t::shader_set_value<fan::vec1_wrap_t<f32_t>>(shader_nr_t nr, const fan::string& name, const fan::vec1_wrap_t<f32_t>& val);
template void fan::opengl::context_t::shader_set_value<fan::vec_wrap_t<1, f32_t>>(shader_nr_t nr, const fan::string& name, const fan::vec_wrap_t<1, f32_t>& val);
template void fan::opengl::context_t::shader_set_value<fan::vec_wrap_t<2, f32_t>>(shader_nr_t nr, const fan::string& name, const fan::vec_wrap_t<2, f32_t>& val);


void shader_set_vertex(fan::opengl::context_t& context, shader_nr_t nr, const fan::string& vertex_code) {
  auto& shader = shader_get(context, nr);

  if (shader.vertex != (uint32_t)fan::uninitialized) {
    fan_opengl_call(glDeleteShader(shader.vertex));
  }

  shader.vertex = fan_opengl_call(glCreateShader(GL_VERTEX_SHADER));
  shader_list[nr].svertex = vertex_code;

  char* ptr = (char*)vertex_code.c_str();
  GLint length = vertex_code.size();

  fan_opengl_call(glShaderSource(shader.vertex, 1, &ptr, &length));
  fan_opengl_call(glCompileShader(shader.vertex));

  shader_check_compile_errors(shader_list[nr], "VERTEX");
}

void shader_set_fragment(fan::opengl::context_t& context, shader_nr_t nr, const fan::string& fragment_code) {
  auto& shader = shader_get(context, nr);

  if (shader.fragment != (uint32_t)-1) {
    fan_opengl_call(glDeleteShader(shader.fragment));
  }

  shader.fragment = fan_opengl_call(glCreateShader(GL_FRAGMENT_SHADER));
  shader_list[nr].sfragment = fragment_code;

  char* ptr = (char*)fragment_code.c_str();
  GLint length = fragment_code.size();

  fan_opengl_call(glShaderSource(shader.fragment, 1, &ptr, &length));

  fan_opengl_call(glCompileShader(shader.fragment));
  shader_check_compile_errors(shader_list[nr], "FRAGMENT");
}

bool shader_compile(fan::opengl::context_t& context, shader_nr_t nr) {
  auto& shader = shader_get(context, nr);

  auto temp_id = fan_opengl_call(glCreateProgram());
  if (shader.vertex != (uint32_t)-1) {
    fan_opengl_call(glAttachShader(temp_id, shader.vertex));
  }
  if (shader.fragment != (uint32_t)-1) {
    fan_opengl_call(glAttachShader(temp_id, shader.fragment));
  }

  fan_opengl_call(glLinkProgram(temp_id));
  bool ret = shader_check_compile_errors(temp_id, "PROGRAM");

  if (ret == false) {
    fan_opengl_call(glDeleteProgram(temp_id));
    return false;
  }

  if (shader.vertex != (uint32_t)-1) {
    fan_opengl_call(glDeleteShader(shader.vertex));
    shader.vertex = -1;
  }
  if (shader.fragment != (uint32_t)-1) {
    fan_opengl_call(glDeleteShader(shader.fragment));
    shader.fragment = -1;
  }

  if (shader.id != (uint32_t)-1) {
    fan_opengl_call(glDeleteProgram(shader.id));
  }
  shader.id = temp_id;

  shader.projection_view[0] = fan_opengl_call(glGetUniformLocation(shader.id, "projection"));
  shader.projection_view[1] = fan_opengl_call(glGetUniformLocation(shader.id, "view"));

  std::regex uniformRegex(R"(uniform\s+(\w+)\s+(\w+)(\s*=\s*[\d\.]+)?;)");

  fan::string vertexData = shader_list[nr].svertex;

  std::smatch match;
  while (std::regex_search(vertexData, match, uniformRegex)) {
      shader_list[nr].uniform_type_table[match[2]] = match[1];
      vertexData = match.suffix().str();
  }

  fan::string fragmentData = shader_list[nr].sfragment;

  while (std::regex_search(fragmentData, match, uniformRegex)) {
      shader_list[nr].uniform_type_table[match[2]] = match[1];
      fragmentData = match.suffix().str();
  }
  return ret;
}

fan::graphics::context_camera_t& camera_get(fan::opengl::context_t& context, camera_nr_t nr) {
  return camera_list[nr];
}

void shader_set_camera(fan::opengl::context_t& context, shader_nr_t nr, fan::graphics::camera_nr_t camera_nr) {
  auto& camera = camera_get(context, camera_nr);
  fan_opengl_call(glUniformMatrix4fv(shader_get(context, nr).projection_view[0], 1, GL_FALSE, &camera.m_projection[0][0]));
  fan_opengl_call(glUniformMatrix4fv(shader_get(context, nr).projection_view[1], 1, GL_FALSE, &camera.m_view[0][0]));
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

fan::opengl::context_t::image_t& image_get(fan::opengl::context_t& context, image_nr_t nr) {
  return *(fan::opengl::context_t::image_t*)image_list[nr].internal;
}

GLuint& image_get_handle(fan::opengl::context_t& context, image_nr_t nr) {
  return image_get(context, nr).texture_id;
}

fan::graphics::image_nr_t image_create(fan::opengl::context_t& context) {
  uint8_t* cptr = (uint8_t*)&context;
  auto* ptr = fan::graphics::get_image_list(cptr);

  image_nr_t nr = image_list.NewNode();
  image_list[nr].internal = new fan::opengl::context_t::image_t;
  fan_opengl_call(glGenTextures(1, &image_get_handle(context, nr)));
  return nr;
}

void image_erase(fan::opengl::context_t& context, image_nr_t nr) {
  auto handle = image_get_handle(context, nr);
  fan_opengl_call(glDeleteTextures(1, (GLuint*)&handle));
  delete static_cast<fan::opengl::context_t::image_t*>(image_list[nr].internal);
  image_list.Recycle(nr);
}

void image_bind(fan::opengl::context_t& context, image_nr_t nr) {
  fan_opengl_call(glBindTexture(GL_TEXTURE_2D, image_get_handle(context, nr)));
}

void image_unbind(fan::opengl::context_t& context, image_nr_t nr) {
  fan_opengl_call(glBindTexture(GL_TEXTURE_2D, 0));
}

fan::graphics::image_load_properties_t& image_get_settings(fan::opengl::context_t& context, image_nr_t nr) {
  return image_list[nr].image_settings;
}

fan::graphics::image_load_properties_t image_opengl_to_global(const fan::opengl::context_t::image_load_properties_t& p);

void image_set_settings(fan::opengl::context_t& context, image_nr_t nr, const fan::opengl::context_t::image_load_properties_t& p) {
  image_bind(context, nr);
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

  image_list[nr].image_settings = image_opengl_to_global(p);
}

fan::graphics::image_nr_t image_load(fan::opengl::context_t& context, const fan::image::image_info_t& image_info, const fan::opengl::context_t::image_load_properties_t& p) {


  image_nr_t nr = image_create(context);
  image_bind(context, nr);
  image_set_settings(context, nr, p);

  auto& image = image_get(context, nr);
  image.size = image_info.size;
  image_list[nr].image_path = "";

  int fmt = 0;
  int internal_fmt = p.internal_format;

  switch(image_info.channels) {
  case 1: {
    fmt = GL_RED;
    break;
  }
  case 2: {
    fmt = GL_RG;
    break;
  }
  case 3: {
    fmt = GL_RGB;
    break;
  }
  case 4: {
    fmt = GL_RGBA;
    break;
  }
  case 0: {
    fmt = p.format;
    break;
  }
  default:{
    fan::throw_error("invalid channels");
    break;
  }
  }

  uint32_t bytes_per_row = (int)(image.size.x * image_info.channels);
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

  fan_opengl_call(glTexImage2D(GL_TEXTURE_2D, 0, internal_fmt, image.size.x, image.size.y, 0, fmt, p.type, image_info.data));

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

fan::graphics::image_nr_t create_missing_texture(fan::opengl::context_t& context) {
  fan::opengl::context_t::image_load_properties_t p;

  p.visual_output = GL_REPEAT;

  image_nr_t nr = image_create(context);
  image_bind(context, nr);

  image_set_settings(context, nr, p);
  auto& image = image_get(context, nr);
  image.size = fan::vec2i(2, 2);

  fan_opengl_call(
    glTexImage2D(
      GL_TEXTURE_2D, 
      0, 
      p.internal_format, 
      image.size.x, 
      image.size.y, 
      0, 
      p.format, 
      p.type, 
      fan::image::missing_texture_pixels
    )
  );

  fan_opengl_call(glGenerateMipmap(GL_TEXTURE_2D));

  return nr;
}

fan::graphics::image_nr_t create_transparent_texture(fan::opengl::context_t& context) {
  fan::opengl::context_t::image_load_properties_t p;

  p.visual_output = GL_REPEAT;

  image_nr_t nr = image_create(context);
  image_bind(context, nr);

  auto& img = image_get(context, nr);

  image_set_settings(context, nr, p);

  img.size = fan::vec2i(2, 2);

  fan_opengl_call(glTexImage2D(GL_TEXTURE_2D, 0, p.internal_format, 2, 2, 0, p.format, p.type, fan::image::transparent_texture_pixels));

  fan_opengl_call(glGenerateMipmap(GL_TEXTURE_2D));
  return nr;
}

fan::graphics::image_nr_t image_load(fan::opengl::context_t& context, const std::string& path, const fan::opengl::context_t::image_load_properties_t& p) {

#if fan_assert_if_same_path_loaded_multiple_times

  static std::unordered_map<fan::string, bool> existing_images;

  if (existing_images.find(path) != existing_images.end()) {
    fan::throw_error("image already existing " + path);
  }

  existing_images[path] = 0;

#endif

  fan::image::image_info_t image_info;
  if (fan::image::load(path, &image_info)) {
    return create_missing_texture(context);
  }
  image_nr_t nr = image_load(context, image_info, p);
  image_list[nr].image_path = path;
  fan::image::free(&image_info);
  return nr;
}

fan::graphics::image_nr_t image_load(fan::opengl::context_t& context, const fan::image::image_info_t& image_info) {
  return image_load(context, image_info, fan::opengl::context_t::image_load_properties_t());
}

fan::graphics::image_nr_t image_load(fan::opengl::context_t& context, fan::color* colors, const fan::vec2ui& size_, const fan::opengl::context_t::image_load_properties_t& p) {

  image_nr_t nr = image_create(context);
  image_bind(context, nr);

  image_set_settings(context, nr, p);

  auto& image = image_get(context, nr);
  image.size = size_;

  fan_opengl_call(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, image.size.x, image.size.y, 0, p.format, GL_FLOAT, (uint8_t*)colors));

  return nr;
}

fan::graphics::image_nr_t image_load(fan::opengl::context_t& context, fan::color* colors, const fan::vec2ui& size_) {
  return image_load(context, colors, size_, fan::opengl::context_t::image_load_properties_t());
}

fan::graphics::image_nr_t image_load(fan::opengl::context_t& context, const std::string& path) {
  return image_load(context, path, fan::opengl::context_t::image_load_properties_t());
}

void image_unload(fan::opengl::context_t& context, image_nr_t nr) {
  image_erase(context, nr);
}

void image_reload(fan::opengl::context_t& context, image_nr_t nr, const fan::image::image_info_t& image_info, const fan::opengl::context_t::image_load_properties_t& p) {

  image_bind(context, nr);

  image_set_settings(context, nr, p);

  auto& image = image_get(context, nr);
  image.size = image_info.size;
  fan_opengl_call(glTexImage2D(GL_TEXTURE_2D, 0, p.internal_format, image.size.x, image.size.y, 0, p.format, p.type, image_info.data));
}

void image_reload(fan::opengl::context_t& context, image_nr_t nr, const fan::image::image_info_t& image_info) {
  image_reload(context, nr, image_info, fan::opengl::context_t::image_load_properties_t());
}

void image_reload(fan::opengl::context_t& context, image_nr_t nr, const std::string& path, const fan::opengl::context_t::image_load_properties_t& p) {
  fan::image::image_info_t image_info;
  if (fan::image::load(path, &image_info)) {
    image_info.data = (void*)fan::image::missing_texture_pixels;
    image_info.size = 2;
    image_info.channels = 4;
    image_info.type = -1; // ignore free
  }
  image_reload(context, nr, image_info, p);
  image_list[nr].image_path = path;
  fan::image::free(&image_info);
}

void image_reload(fan::opengl::context_t& context, image_nr_t nr, const std::string& path) {
  image_reload(context, nr, path, fan::opengl::context_t::image_load_properties_t());
}

std::unique_ptr<uint8_t[]> image_get_pixel_data(fan::opengl::context_t& context, image_nr_t nr, GLenum format, fan::vec2 uvp, fan::vec2 uvs) {
  auto& image = image_get(context, nr);
  image_bind(context, nr);

  fan::vec2ui uv_size = {
      (uint32_t)(image.size.x * uvs.x),
      (uint32_t)(image.size.y * uvs.y)
  };

  auto full_ptr = std::make_unique<uint8_t[]>(image.size.x * image.size.y * 4); // assuming rgba

  fan_opengl_call(glGetTexImage(GL_TEXTURE_2D,
    0,
    format,
    GL_UNSIGNED_BYTE,
    full_ptr.get())
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

// creates single colored text size.x*size.y sized
fan::graphics::image_nr_t image_create(fan::opengl::context_t& context, const fan::color& color, const fan::opengl::context_t::image_load_properties_t& p) {

  uint8_t pixels[4];
  for (uint32_t p = 0; p < fan::color::size(); p++) {
    pixels[p] = color[p] * 255;
  }

  image_nr_t nr = image_create(context);
  image_bind(context, nr);

  image_set_settings(context, nr, p);

  fan_opengl_call(glTexImage2D(GL_TEXTURE_2D, 0, p.internal_format, 1, 1, 0, p.format, p.type, pixels));

  auto& image = image_get(context, nr);
  image.size = 1;

  fan_opengl_call(glGenerateMipmap(GL_TEXTURE_2D));

  return nr;
}

fan::graphics::image_nr_t image_create(fan::opengl::context_t& context, const fan::color& color) {
  return image_create(context, color, fan::opengl::context_t::image_load_properties_t());
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


fan::graphics::camera_nr_t camera_create(fan::opengl::context_t& context) {
  return camera_list.NewNode();
}

void camera_erase(fan::opengl::context_t& context, camera_nr_t nr) {
  camera_list.Recycle(nr);
}

void camera_set_ortho(fan::opengl::context_t& context, camera_nr_t nr, fan::vec2 x, fan::vec2 y) {
  auto& camera = camera_get(context, nr);

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
}

fan::graphics::camera_nr_t camera_open(fan::opengl::context_t& context, const fan::vec2& x, const fan::vec2& y) {
  camera_nr_t nr = camera_create(context);
  camera_set_ortho(context, nr, fan::vec2(x.x, x.y), fan::vec2(y.x, y.y));
  return nr;
}

fan::vec3 camera_get_position(fan::opengl::context_t& context, camera_nr_t nr) {
  return camera_get(context, nr).position;
}

void camera_set_position(fan::opengl::context_t& context, camera_nr_t nr, const fan::vec3& cp) {
  auto& camera = camera_get(context, nr);
  camera.position = cp;

  camera.m_view[3][0] = 0;
  camera.m_view[3][1] = 0;
  camera.m_view[3][2] = 0;
  camera.m_view = camera.m_view.translate(camera.position);
  fan::vec3 position = camera.m_view.get_translation();
  constexpr fan::vec3 front(0, 0, 1);

  camera.m_view = fan::math::look_at_left<fan::mat4, fan::vec3>(position, position + front, fan::camera::world_up);
}

fan::vec2 camera_get_size(fan::opengl::context_t& context, camera_nr_t nr) {
  auto& camera = camera_get(context, nr);
  return fan::vec2(std::abs(camera.coordinates.right - camera.coordinates.left), std::abs(camera.coordinates.down - camera.coordinates.up));
}

void camera_set_perspective(fan::opengl::context_t& context, camera_nr_t nr, f32_t fov, const fan::vec2& window_size) {
  auto& camera = camera_get(context, nr);

  camera.m_projection = fan::math::perspective<fan::mat4>(fan::math::radians(fov), (f32_t)window_size.x / (f32_t)window_size.y, camera.znear, camera.zfar);

  camera.update_view();

  camera.m_view = camera.get_view_matrix();
}

void camera_rotate(fan::opengl::context_t& context, camera_nr_t nr, const fan::vec2& offset) {
  auto& camera = camera_get(context, nr);
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



void fan::opengl::context_t::shader_set_camera(shader_nr_t nr, fan::graphics::camera_nr_t camera_nr) {
  auto& camera = camera_get(*this, camera_nr);
  fan_opengl_call(glUniformMatrix4fv(shader_get(*this, nr).projection_view[0], 1, GL_FALSE, &camera.m_projection[0][0]));
  fan_opengl_call(glUniformMatrix4fv(shader_get(*this, nr).projection_view[1], 1, GL_FALSE, &camera.m_view[0][0]));
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

void viewport_set(fan::opengl::context_t& context, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
  fan_opengl_call(glViewport(viewport_position_.x, window_size.y - viewport_size_.y - viewport_position_.y,
    viewport_size_.x, viewport_size_.y
  ));
}

fan::graphics::context_viewport_t& viewport_get(fan::opengl::context_t& context, viewport_nr_t nr) {
  return viewport_list[nr];
}

void viewport_set(fan::opengl::context_t& context, viewport_nr_t nr, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
  auto& viewport = viewport_get(context, nr);
  viewport.viewport_position = viewport_position_;
  viewport.viewport_size = viewport_size_;

  viewport_set(context, viewport_position_, viewport_size_, window_size);
}

fan::graphics::viewport_nr_t viewport_create(fan::opengl::context_t& context) {
  auto nr = viewport_list.NewNode();

  viewport_set(
    context,
    nr,
    0, 0, 0
  );
  return nr;
}

void viewport_erase(fan::opengl::context_t& context, viewport_nr_t nr) {
  viewport_list.Recycle(nr);
}

fan::vec2 viewport_get_position(fan::opengl::context_t& context, viewport_nr_t nr) {
  return viewport_get(context, nr).viewport_position;
}

fan::vec2 viewport_get_size(fan::opengl::context_t& context, viewport_nr_t nr) {
  return viewport_get(context, nr).viewport_size;
}


void viewport_zero(fan::opengl::context_t& context, viewport_nr_t nr) {
  auto& viewport = viewport_get(context, nr);
  viewport.viewport_position = 0;
  viewport.viewport_size = 0;
  fan_opengl_call(glViewport(0, 0, 0, 0));
}

bool viewport_inside(fan::opengl::context_t& context, viewport_nr_t nr, const fan::vec2& position) {
  auto& viewport = viewport_get(context, nr);
  return fan_2d::collision::rectangle::point_inside_no_rotation(position, viewport.viewport_position + viewport.viewport_size / 2, viewport.viewport_size / 2);
}

bool viewport_inside_wir(fan::opengl::context_t& context, viewport_nr_t nr, const fan::vec2& position) {
  auto& viewport = viewport_get(context, nr);
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


uint32_t global_to_opengl_format(uintptr_t format) {
  if (format == image_format::b8g8r8a8_unorm) return GL_BGRA;
  if (format == image_format::r8b8g8a8_unorm) return GL_RGBA;
  if (format == image_format::r8_unorm) return GL_RED;
  if (format == image_format::rg8_unorm) return GL_RG;
  if (format == image_format::rgb_unorm) return GL_RGB;
  if (format == image_format::rgba_unorm) return GL_RGBA;
  if (format == image_format::r8_uint) return GL_RED_INTEGER;
  if (format == image_format::r8g8b8a8_srgb) return GL_SRGB8_ALPHA8;
  if (format == image_format::r11f_g11f_b10f) return GL_R11F_G11F_B10F;
#if fan_debug >= fan_debug_high
  fan::throw_error("invalid format");
#endif
  return GL_RGBA;
}

uint32_t global_to_opengl_type(uintptr_t type) {
  if (type == fan_unsigned_byte) return GL_UNSIGNED_BYTE;
  if (type == fan_unsigned_int) return GL_UNSIGNED_INT;
  if (type == fan_float) return GL_FLOAT;
#if fan_debug >= fan_debug_high
  fan::throw_error("invalid format");
#endif
  return 0;
}

uint32_t global_to_opengl_address_mode(uint32_t mode) {
  if (mode == image_sampler_address_mode::repeat) return GL_REPEAT;
  if (mode == image_sampler_address_mode::mirrored_repeat) return GL_MIRRORED_REPEAT;
  if (mode == image_sampler_address_mode::clamp_to_edge) return GL_CLAMP_TO_EDGE;
  if (mode == image_sampler_address_mode::clamp_to_border) return GL_CLAMP_TO_BORDER;
  if (mode == image_sampler_address_mode::mirrored_clamp_to_edge) return GL_MIRROR_CLAMP_TO_EDGE;
#if fan_debug >= fan_debug_high
  fan::throw_error("invalid format");
#endif
  return GL_REPEAT;
}

uint32_t global_to_opengl_filter(uintptr_t filter) {
  if (filter == image_filter::nearest) return GL_NEAREST;
  if (filter == image_filter::linear) return GL_LINEAR;
#if fan_debug >= fan_debug_high
  fan::throw_error("invalid format");
#endif
  return GL_NEAREST;
}

uint32_t opengl_to_global_format(uintptr_t format) {
  if (format == GL_BGRA) return image_format::b8g8r8a8_unorm;
  if (format == GL_RGBA) return image_format::r8b8g8a8_unorm;
  if (format == GL_RED) return image_format::r8_unorm;
  if (format == GL_RG) return image_format::rg8_unorm;
  if (format == GL_RGB) return image_format::rgb_unorm;
  if (format == GL_RED_INTEGER) return image_format::r8_uint;
  if (format == GL_SRGB8_ALPHA8) return image_format::r8g8b8a8_srgb;
  if (format == GL_R11F_G11F_B10F) return image_format::r11f_g11f_b10f;
#if fan_debug >= fan_debug_high
  fan::throw_error("invalid format");
#endif
  return image_format::rgba_unorm;
}

uint32_t opengl_to_global_type(uintptr_t type) {
  if (type == GL_UNSIGNED_BYTE) return fan_unsigned_byte;
  if (type == GL_UNSIGNED_INT) return fan_unsigned_int;
  if (type == GL_FLOAT) return fan_float;
#if fan_debug >= fan_debug_high
  fan::throw_error("invalid format");
#endif
  return 0;
}

uint32_t opengl_to_global_address_mode(uint32_t mode) {
  if (mode == GL_REPEAT) return image_sampler_address_mode::repeat;
  if (mode == GL_MIRRORED_REPEAT) return image_sampler_address_mode::mirrored_repeat;
  if (mode == GL_CLAMP_TO_EDGE) return image_sampler_address_mode::clamp_to_edge;
  if (mode == GL_CLAMP_TO_BORDER) return image_sampler_address_mode::clamp_to_border;
  if (mode == GL_MIRROR_CLAMP_TO_EDGE) return image_sampler_address_mode::mirrored_clamp_to_edge;
#if fan_debug >= fan_debug_high
  fan::throw_error("invalid format");
#endif
  return image_sampler_address_mode::repeat;
}

uint32_t opengl_to_global_filter(uintptr_t filter) {
  if (filter == GL_NEAREST) return image_filter::nearest;
  if (filter == GL_LINEAR) return image_filter::linear;
#if fan_debug >= fan_debug_high
  fan::throw_error("invalid format");
#endif
  return image_filter::nearest;
}


void open(fan::opengl::context_t& context, const fan::opengl::context_t::properties_t&) {
  context.opengl.open();
}

void fan::opengl::context_t::internal_close() {
  fan::opengl::context_t& context = *this;
  {
    fan::graphics::shader_list_t::nrtra_t nrtra;
    fan::graphics::shader_nr_t nr;
    nrtra.Open(&shader_list, &nr);
    while (nrtra.Loop(&shader_list, &nr)) {
      delete static_cast<fan::opengl::context_t::shader_t*>(shader_list[nr].internal);
    }
    nrtra.Close(&shader_list);
  }
  {
    fan::graphics::image_list_t::nrtra_t nrtra;
    fan::graphics::image_nr_t nr;
    nrtra.Open(&image_list, &nr);
    while (nrtra.Loop(&image_list, &nr)) {
      delete static_cast<fan::opengl::context_t::image_t*>(image_list[nr].internal);
    }
    nrtra.Close(&image_list);
  }
}

void close(fan::opengl::context_t& context) {
  {
    fan::graphics::camera_list_t::nrtra_t nrtra;
    fan::graphics::camera_nr_t nr;
    nrtra.Open(&camera_list, &nr);
    while (nrtra.Loop(&camera_list, &nr)) {
      camera_erase(context, nr);
    }
    nrtra.Close(&camera_list);
  }
  {
    fan::graphics::shader_list_t::nrtra_t nrtra;
    fan::graphics::shader_nr_t nr;
    nrtra.Open(&shader_list, &nr);
    while (nrtra.Loop(&shader_list, &nr)) {
      shader_erase(context, nr);
    }
    nrtra.Close(&shader_list);
  }
  {
    fan::graphics::image_list_t::nrtra_t nrtra;
    fan::graphics::image_nr_t nr;
    nrtra.Open(&image_list, &nr);
    while (nrtra.Loop(&image_list, &nr)) {
      image_erase(context, nr);
    }
    nrtra.Close(&image_list);
  }
  {
    fan::graphics::viewport_list_t::nrtra_t nrtra;
    fan::graphics::viewport_nr_t nr;
    nrtra.Open(&viewport_list, &nr);
    while (nrtra.Loop(&viewport_list, &nr)) {
      viewport_erase(context, nr);
    }
    nrtra.Close(&viewport_list);
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
fan::opengl::context_t::image_load_properties_t image_global_to_opengl(const fan::graphics::image_load_properties_t& p) {
  return {
    .visual_output = global_to_opengl_address_mode(p.visual_output),
    .internal_format = global_to_opengl_format(p.internal_format),
    .format = global_to_opengl_format(p.format),
    .type = global_to_opengl_type(p.type),
    .min_filter = global_to_opengl_filter(p.min_filter),
    .mag_filter = global_to_opengl_filter(p.mag_filter),
  };
}

fan::graphics::image_load_properties_t image_opengl_to_global(const fan::opengl::context_t::image_load_properties_t& p) {
  return {
    .visual_output = opengl_to_global_address_mode(p.visual_output),
    .internal_format = opengl_to_global_format(p.internal_format),
    .format = opengl_to_global_format(p.format),
    .type = opengl_to_global_type(p.type),
    .min_filter = opengl_to_global_filter(p.min_filter),
    .mag_filter = opengl_to_global_filter(p.mag_filter),
  };
}

fan::graphics::context_functions_t fan::graphics::get_gl_context_functions() {
	fan::graphics::context_functions_t cf;
  cf.shader_create = [](void* context) { 
    return shader_create(*(fan::opengl::context_t*)context);
  }; 
  cf.shader_get = [](void* context, shader_nr_t nr) { 
    return (void*)&shader_get(*(fan::opengl::context_t*)context, nr);
  }; 
  cf.shader_erase = [](void* context, shader_nr_t nr) { 
    shader_erase(*(fan::opengl::context_t*)context,nr); 
  }; 
  cf.shader_use = [](void* context, shader_nr_t nr) { 
    shader_use(*(fan::opengl::context_t*)context,nr); 
  }; 
  cf.shader_set_vertex = [](void* context, shader_nr_t nr, const std::string& vertex_code) { 
    shader_set_vertex(*(fan::opengl::context_t*)context, nr, vertex_code); 
  }; 
  cf.shader_set_fragment = [](void* context, shader_nr_t nr, const std::string& fragment_code) { 
    shader_set_fragment(*(fan::opengl::context_t*)context,nr, fragment_code); 
  }; 
  cf.shader_compile = [](void* context, shader_nr_t nr) { 
    return shader_compile(*(fan::opengl::context_t*)context,nr); 
  }; 
    /*image*/
  cf.image_create = [](void* context) {
    return image_create(*(fan::opengl::context_t*)context);
  }; 
  cf.image_get_handle = [](void* context, image_nr_t nr) { 
    return (uint64_t)image_get_handle(*(fan::opengl::context_t*)context,nr); 
  }; 
  cf.image_get = [](void* context, image_nr_t nr) {
    return (void*)&image_get(*(fan::opengl::context_t*)context, nr);
  }; 
  cf.image_erase = [](void* context, image_nr_t nr) { 
    image_erase(*(fan::opengl::context_t*)context,nr); 
  }; 
  cf.image_bind = [](void* context, image_nr_t nr) { 
    image_bind(*(fan::opengl::context_t*)context,nr); 
  }; 
  cf.image_unbind = [](void* context, image_nr_t nr) { 
    image_unbind(*(fan::opengl::context_t*)context,nr); 
  }; 
  cf.image_get_settings = [](void* context, fan::graphics::image_nr_t nr) -> fan::graphics::image_load_properties_t& {
    return image_get_settings(*(fan::opengl::context_t*)context, nr);
  };
  cf.image_set_settings = [](void* context, image_nr_t nr, const fan::graphics::image_load_properties_t& settings) {
    image_set_settings(*(fan::opengl::context_t*)context, nr, image_global_to_opengl(settings));
  }; 
  cf.image_load_info = [](void* context, const fan::image::image_info_t& image_info) { 
    return image_load(*(fan::opengl::context_t*)context, image_info);
  }; 
  cf.image_load_info_props = [](void* context, const fan::image::image_info_t& image_info, const fan::graphics::image_load_properties_t& p) { 
    return image_load(*(fan::opengl::context_t*)context, image_info, image_global_to_opengl(p));
  }; 
  cf.image_load_path = [](void* context, const std::string& path) { 
    return image_load(*(fan::opengl::context_t*)context, path);
  }; 
  cf.image_load_path_props = [](void* context, const std::string& path, const fan::graphics::image_load_properties_t& p) { 
    return image_load(*(fan::opengl::context_t*)context, path, image_global_to_opengl(p));
  }; 
  cf.image_load_colors = [](void* context, fan::color* colors, const fan::vec2ui& size_) { 
    return image_load(*(fan::opengl::context_t*)context, colors, size_);
  }; 
  cf.image_load_colors_props = [](void* context, fan::color* colors, const fan::vec2ui& size_, const fan::graphics::image_load_properties_t& p) { 
    return image_load(*(fan::opengl::context_t*)context, colors, size_, image_global_to_opengl(p));
  }; 
  cf.image_unload = [](void* context, image_nr_t nr) { 
    image_unload(*(fan::opengl::context_t*)context, nr); 
  }; 
  cf.create_missing_texture = [](void* context) { 
    return create_missing_texture(*(fan::opengl::context_t*)context);
  }; 
  cf.create_transparent_texture = [](void* context) { 
    return create_transparent_texture(*(fan::opengl::context_t*)context);
  }; 
  cf.image_reload_image_info = [](void* context, image_nr_t nr, const fan::image::image_info_t& image_info) { 
    return image_reload(*(fan::opengl::context_t*)context, nr, image_info); 
  }; 
  cf.image_reload_image_info_props = [](void* context, image_nr_t nr, const fan::image::image_info_t& image_info, const fan::graphics::image_load_properties_t& p) { 
    return image_reload(*(fan::opengl::context_t*)context, nr, image_info, image_global_to_opengl(p)); 
  }; 
  cf.image_reload_path = [](void* context, image_nr_t nr, const std::string& path) { 
    return image_reload(*(fan::opengl::context_t*)context, nr, path); 
  }; 
  cf.image_reload_path_props = [](void* context, image_nr_t nr, const std::string& path, const fan::graphics::image_load_properties_t& p) { 
    return image_reload(*(fan::opengl::context_t*)context, nr, path, image_global_to_opengl(p)); 
  };
  cf.image_create_color = [](void* context, const fan::color& color) { 
    return image_create(*(fan::opengl::context_t*)context, color);
  }; 
  cf.image_create_color_props = [](void* context, const fan::color& color, const fan::graphics::image_load_properties_t& p) { 
    return image_create(*(fan::opengl::context_t*)context, color, image_global_to_opengl(p));
  };
  /*camera*/
  cf.camera_create = [](void* context) {
    return camera_create(*(fan::opengl::context_t*)context);
  };
  cf.camera_get = [](void* context, fan::graphics::camera_nr_t nr) -> decltype(auto) {
    return camera_get(*(fan::opengl::context_t*)context, nr);
  };
  cf.camera_erase = [](void* context, camera_nr_t nr) { 
    camera_erase(*(fan::opengl::context_t*)context, nr); 
  };
  cf.camera_open = [](void* context, const fan::vec2& x, const fan::vec2& y) {
    return camera_open(*(fan::opengl::context_t*)context, x, y);
  };
  cf.camera_get_position = [](void* context, camera_nr_t nr) { 
    return camera_get_position(*(fan::opengl::context_t*)context, nr); 
  };
  cf.camera_set_position = [](void* context, camera_nr_t nr, const fan::vec3& cp) { 
    camera_set_position(*(fan::opengl::context_t*)context, nr, cp); 
  };
  cf.camera_get_size = [](void* context, camera_nr_t nr) { 
    return camera_get_size(*(fan::opengl::context_t*)context, nr); 
  };
  cf.camera_set_ortho = [](void* context, camera_nr_t nr, fan::vec2 x, fan::vec2 y) { 
    camera_set_ortho(*(fan::opengl::context_t*)context, nr, x, y); 
  };
  cf.camera_set_perspective = [](void* context, camera_nr_t nr, f32_t fov, const fan::vec2& window_size) { 
    camera_set_perspective(*(fan::opengl::context_t*)context, nr, fov, window_size); 
  };
  cf.camera_rotate = [](void* context, camera_nr_t nr, const fan::vec2& offset) { 
    camera_rotate(*(fan::opengl::context_t*)context, nr, offset); 
  };
  /*viewport*/
  cf.viewport_create = [](void* context) {
    return viewport_create(*(fan::opengl::context_t*)context);
  };
  cf.viewport_get = [](void* context, viewport_nr_t nr) -> fan::graphics::context_viewport_t&{ 
    return viewport_get(*(fan::opengl::context_t*)context, nr);
  };
  cf.viewport_erase = [](void* context, viewport_nr_t nr) { 
    viewport_erase(*(fan::opengl::context_t*)context, nr); 
  };
  cf.viewport_get_position = [](void* context, viewport_nr_t nr) { 
    return viewport_get_position(*(fan::opengl::context_t*)context, nr); 
  };
  cf.viewport_get_size = [](void* context, viewport_nr_t nr) { 
    return viewport_get_size(*(fan::opengl::context_t*)context, nr); 
  };
  cf.viewport_set = [](void* context, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) { 
    viewport_set(*(fan::opengl::context_t*)context, viewport_position_, viewport_size_, window_size); 
  };
  cf.viewport_set_nr = [](void* context, viewport_nr_t nr, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) { 
    viewport_set(*(fan::opengl::context_t*)context, nr, viewport_position_, viewport_size_, window_size); 
  };
  cf.viewport_zero = [](void* context, viewport_nr_t nr) { 
    viewport_zero(*(fan::opengl::context_t*)context, nr); 
  };
  cf.viewport_inside = [](void* context, viewport_nr_t nr, const fan::vec2& position) { 
    return viewport_inside(*(fan::opengl::context_t*)context, nr, position); 
  };
  cf.viewport_inside_wir = [](void* context, viewport_nr_t nr, const fan::vec2& position) { 
    return viewport_inside_wir(*(fan::opengl::context_t*)context, nr, position); 
  };
  return cf;
}

uint32_t fan::opengl::core::get_draw_mode(uint8_t draw_mode) {
  switch (draw_mode) {
  case primitive_topology_t::points:
    return fan::opengl::context_t::primitive_topology_t::points;
  case primitive_topology_t::lines:
    return fan::opengl::context_t::primitive_topology_t::lines;
  case primitive_topology_t::line_strip:
    return fan::opengl::context_t::primitive_topology_t::line_strip;
  case primitive_topology_t::triangles:
    return fan::opengl::context_t::primitive_topology_t::triangles;
  case primitive_topology_t::triangle_strip:
    return fan::opengl::context_t::primitive_topology_t::triangle_strip;
  case primitive_topology_t::triangle_fan:
    return fan::opengl::context_t::primitive_topology_t::triangle_fan;
  case primitive_topology_t::lines_with_adjacency:
    return fan::opengl::context_t::primitive_topology_t::lines_with_adjacency;
  case primitive_topology_t::line_strip_with_adjacency:
    return fan::opengl::context_t::primitive_topology_t::line_strip_with_adjacency;
  case primitive_topology_t::triangles_with_adjacency:
    return fan::opengl::context_t::primitive_topology_t::triangles_with_adjacency;
  case primitive_topology_t::triangle_strip_with_adjacency:
    return fan::opengl::context_t::primitive_topology_t::triangle_strip_with_adjacency;
  default:
    fan::throw_error("invalid draw mode");
    return -1;
  }
}
