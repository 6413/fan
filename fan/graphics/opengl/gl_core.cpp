#include "gl_core.h"

void fan::opengl::context_t::print_version() {
  fan::print("opengl version supported:", opengl.glGetString(fan::opengl::GL_VERSION));
}

void fan::opengl::context_t::message_callback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
  //if (type == 33361 || type == 33360) { // gl_static_draw
  //  return;
  //}
  fan::print_no_space(type == GL_DEBUG_TYPE_ERROR ? "opengl error:" : "", type, ", severity:", severity, ", message:", message);
}

void fan::opengl::context_t::set_error_callback() {
  opengl.call(opengl.glEnable, GL_DEBUG_OUTPUT);
  opengl.call(opengl.glDebugMessageCallback, message_callback, (void*)0);
}

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

  if (buffer_size < size || (offset + size) > buffer_size) {
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

const void fan::opengl::core::vao_t::bind(fan::opengl::context_t& context) const {
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