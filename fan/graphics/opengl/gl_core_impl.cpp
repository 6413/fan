module;

#include <fan/utility.h>

#if defined(fan_compiler_gcc)
  // fixes collision with GLFW3 headers while doing import std;
  #ifndef _GCC_MAX_ALIGN_T
    #define _GCC_MAX_ALIGN_T
  #endif
#endif

#if defined(FAN_OPENGL)
  #if defined(fan_platform_windows)
    #define GLFW_EXPOSE_NATIVE_WIN32
    #define GLFW_EXPOSE_NATIVE_WGL
    #define GLFW_NATIVE_INCLUDE_NONE
  #endif
  #define GLFW_INCLUDE_NONE
  #include <GLFW/glfw3.h>
  #define __fan_internal_camera_list (*fan::graphics::ctx().camera_list)
  #define __fan_internal_shader_list (*fan::graphics::ctx().shader_list)
  #define __fan_internal_image_list (*fan::graphics::ctx().image_list)
  #define __fan_internal_viewport_list (*fan::graphics::ctx().viewport_list)
  #include <fan/graphics/gl_api.h>
#endif

module fan.graphics.opengl.core;

#if defined(FAN_OPENGL)

import fan.print.error;
import fan.math;
import fan.camera;

#define NO_FUNCS
#include <fan/graphics/opengl/init.h>

namespace fan::opengl {

  void opengl_t::open() {
    static std::uint8_t init = 1;
    if (init == 0) {
      return;
    }
    init = 0;
  #if !defined(__wasm__)
    if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress)) {
      fan::throw_error("glad init error");
    }
  #endif
  }

  void context_t::print_version() {
    fan::print_impl("opengl version supported:", fan_opengl_call(glGetString(GL_VERSION)));
  }

  void context_t::error_callback(int error, const char* description) {
    if (error == GLFW_NOT_INITIALIZED) {
      return;
    }
    fan::print_impl("window error " + std::to_string(error) + ": " + description);
    //__abort();
  }

  void context_t::open(const properties_t&) {
    opengl.open();
  }

  void context_t::close() {
     internal_close();
     __fan_internal_shader_list.Clear();
     __fan_internal_image_list.Clear();
    // std::printf("%p", __fan_internal_image_list);
  }

  void context_t::internal_close() {
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

  void context_t::render(fan::window_t& window) {
    glfwSwapBuffers(window.glfw_window);
  }

  void context_t::set_depth_test(bool flag) {
    if (flag) {
      fan_opengl_call(glEnable(GL_DEPTH_TEST));
    }
    else {
      fan_opengl_call(glDisable(GL_DEPTH_TEST));
    }
  }

  void context_t::set_blending(bool flag) {
    if (flag) {
      fan_opengl_call(glDisable(GL_BLEND));
    }
    else {
      fan_opengl_call(glEnable(GL_BLEND));
      fan_opengl_call(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
    }
  }

  void context_t::set_stencil_test(bool flag) {
    if (flag) {
      fan_opengl_call(glEnable(GL_STENCIL_TEST));
    }
    else {
      fan_opengl_call(glDisable(GL_STENCIL_TEST));
    }
  }

  void context_t::set_stencil_op(GLenum sfail, GLenum dpfail, GLenum dppass) {
    fan_opengl_call(glStencilOp(sfail, dpfail, dppass));
  }

  void context_t::set_vsync(fan::window_t*, bool flag) {
    glfwSwapInterval(flag);
  }

  context_t::pbo_t context_t::pbo_create(std::size_t size) {
    pbo_t p;
    p.size = size;
    glGenBuffers(1, &p.id);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, p.id);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size, nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    return p;
  }
  void context_t::pbo_destroy(context_t::pbo_t& p) {
    if (p.id) { glDeleteBuffers(1, &p.id); p.id = 0; }
  }
  std::uint8_t* context_t::pbo_map_write(const context_t::pbo_t& p) {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, p.id);
    return (std::uint8_t*)glMapBufferRange(
      GL_PIXEL_UNPACK_BUFFER, 0, p.size,
      GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT
    );
  }
  void context_t::pbo_unmap() {
    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  }
  void context_t::pbo_upload_to_texture(
    fan::graphics::image_nr_t nr,
    const context_t::pbo_t& p,
    fan::vec2ui size,
    std::uintptr_t global_format,
    GLenum type
  ) {
    GLenum gl_format = global_to_opengl_format(global_format);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, p.id);
    glBindTexture(GL_TEXTURE_2D, image_get_handle(nr));
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, size.x, size.y, gl_format, type, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  }


  void context_t::message_callback(
    GLenum,
    GLenum type,
    GLuint,
    GLenum severity,
    GLsizei,
    const GLchar* message,
    const void*
  ) {
    if (type == 33361 || type == 33360) {
      return;
    }
  #if !defined(__wasm__)
    fan::print_no_space(type == GL_DEBUG_TYPE_ERROR ? "opengl error:" : "", type, ", severity:", severity, ", message:", message);
  #endif
    fan::debug::print_stacktrace();
  }

  void context_t::set_error_callback() {
  #if !defined(__wasm__)
    fan_opengl_call(glEnable(GL_DEBUG_OUTPUT));
    fan_opengl_call(glDebugMessageCallback(message_callback, (void*)0));
  #endif
  }

  void context_t::set_current(fan::window_t* window) {
    if (window == nullptr) {
      glfwMakeContextCurrent(nullptr);
    }
    else {
      glfwMakeContextCurrent(window->glfw_window);
    }
  }

  GLenum context_t::get_format_from_channels(int channels) {
    switch (channels) {
    case 1: return GL_RED;
    case 2: return GL_RG;
    case 3: return GL_RGB;
    case 4: return GL_RGBA;
    default: return GL_RGBA;
    }
  }


  fan::graphics::shader_nr_t context_t::shader_create() {
    auto nr = __fan_internal_shader_list.NewNode();
    __fan_internal_shader_list[nr].internal = new fan::opengl::context_t::shader_t;
    return nr;
  }
  fan::opengl::context_t::shader_t& context_t::shader_get(fan::graphics::shader_nr_t nr) {
    return *(fan::opengl::context_t::shader_t*)__fan_internal_shader_list[nr].internal;
  }
  void context_t::shader_erase(fan::graphics::shader_nr_t nr) {
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
    operator delete(static_cast<fan::opengl::context_t::shader_t*>(__fan_internal_shader_list[nr].internal));
    __fan_internal_shader_list.Recycle(nr);
  }
  bool context_t::shader_check_compile_errors(GLuint shader, const std::string_view file_path, const std::string& type) {
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
    if (!success) {
      int test;
      if (program) {
        fan_opengl_call(glGetProgramInfoLog(shader, buffer_size, nullptr, buffer.data()));
      }
      else {
        fan_opengl_call(glGetShaderInfoLog(shader, buffer_size, &test, buffer.data()));
      }
      std::string fpath = std::string(file_path);
      fan::print_impl("failed to compile: " + type, buffer, "file_path:", fpath.empty() ? "PATH FILE NOT FOUND" : fpath);
      return false;
    }
    return true;
  }
  bool context_t::shader_check_compile_errors(
    fan::graphics::shader_data_t& common_shader,
    const std::string_view file_path,
    const std::string& type
  ) {
    fan::opengl::context_t::shader_t& shader =
      *(fan::opengl::context_t::shader_t*)common_shader.internal;

    enum class shader_type_e {
      vertex,
      fragment,
      compute,
      program
    };

    shader_type_e shader_type;

    if (type == "VERTEX") {
      shader_type = shader_type_e::vertex;
    }
    else if (type == "FRAGMENT") {
      shader_type = shader_type_e::fragment;
    }
    else if (type == "COMPUTE") {
      shader_type = shader_type_e::compute;
    }
    else {
      shader_type = shader_type_e::program;
    }

    GLuint shader_id = 0;

    switch (shader_type) {
    case shader_type_e::vertex:
      shader_id = shader.vertex;
      break;

    case shader_type_e::fragment:
      shader_id = shader.fragment;
      break;

    case shader_type_e::compute:
      shader_id = shader.compute;
      break;

    default:
      break;
    }

    GLint success = 0;

    if (shader_type != shader_type_e::program) {
      fan_opengl_call(
        glGetShaderiv(shader_id, GL_COMPILE_STATUS, &success)
      );
    }
    else {
      fan_opengl_call(
        glGetProgramiv(shader.id, GL_LINK_STATUS, &success)
      );
    }

    if (success) {
      return true;
    }

    GLint buffer_size = 0;

    if (shader_type != shader_type_e::program) {
      fan_opengl_call(
        glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &buffer_size)
      );
    }
    else {
      fan_opengl_call(
        glGetProgramiv(shader.id, GL_INFO_LOG_LENGTH, &buffer_size)
      );
    }

    if (buffer_size <= 0) {

      std::string fpath = std::string(file_path);

      fan::print_impl(
        "failed to compile/link:",
        type,
        "but no error log was provided by OpenGL",
        "file_path:",
        fpath.empty() ? "PATH FILE NOT FOUND" : fpath
      );

      return false;
    }

    std::string buffer;
    buffer.resize(buffer_size);

    if (shader_type == shader_type_e::program) {

      fan_opengl_call(
        glGetProgramInfoLog(
          shader.id,
          buffer_size,
          nullptr,
          buffer.data()
        )
      );
    }
    else {

      GLint written = 0;

      fan_opengl_call(
        glGetShaderInfoLog(
          shader_id,
          buffer_size,
          &written,
          buffer.data()
        )
      );
    }

    std::string fpath = std::string(file_path);

    fan::print_impl(
      "failed to compile/link:",
      type,
      buffer,
      "file_path:",
      fpath.empty() ? "PATH FILE NOT FOUND" : fpath
    );

    return false;
  }
  void context_t::shader_use(fan::graphics::shader_nr_t nr) {
    auto& shader = shader_get(nr);
    if (shader.id == current_program) {
      return;
    }
    fan_opengl_call(glUseProgram(shader.id));
    current_program = shader.id;
  }
  void context_t::shader_set_vertex(fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& vertex_code) {
    auto& shader = shader_get(nr);
    if (shader.vertex != (std::uint32_t)fan::uninitialized) {
      fan_opengl_call(glDeleteShader(shader.vertex));
    }
    auto& internal_shader = __fan_internal_shader_list[nr];
    shader.vertex = fan_opengl_call(glCreateShader(GL_VERTEX_SHADER));
    internal_shader.path_vertex = file_path;
    internal_shader.svertex = vertex_code;
    char* ptr = (char*)vertex_code.c_str();
    GLint length = vertex_code.size();
    fan_opengl_call(glShaderSource(shader.vertex, 1, &ptr, &length));
    fan_opengl_call(glCompileShader(shader.vertex));
    shader_check_compile_errors(internal_shader, file_path, "VERTEX");
  }
  void context_t::shader_set_fragment(fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& fragment_code) {
    auto& shader = shader_get(nr);
    if (shader.fragment != (std::uint32_t)-1) {
      fan_opengl_call(glDeleteShader(shader.fragment));
    }
    auto& internal_shader = __fan_internal_shader_list[nr];
    shader.fragment = fan_opengl_call(glCreateShader(GL_FRAGMENT_SHADER));
    internal_shader.sfragment = fragment_code;
    internal_shader.path_fragment = file_path;
    char* ptr = (char*)fragment_code.c_str();
    GLint length = fragment_code.size();
    fan_opengl_call(glShaderSource(shader.fragment, 1, &ptr, &length));
    fan_opengl_call(glCompileShader(shader.fragment));
    shader_check_compile_errors(internal_shader, file_path, "FRAGMENT");
  }
  void context_t::shader_set_compute(
    fan::graphics::shader_nr_t nr,
    const std::string_view file_path,
    const std::string& compute_code
  ) {
    auto& shader = shader_get(nr);

    if (shader.compute != (std::uint32_t)-1) {
      fan_opengl_call(glDeleteShader(shader.compute));
    }

    auto& internal_shader = __fan_internal_shader_list[nr];

    shader.compute = fan_opengl_call(glCreateShader(GL_COMPUTE_SHADER));

    internal_shader.scompute = compute_code;
    internal_shader.path_compute = file_path;

    char* ptr = (char*)compute_code.c_str();
    GLint length = compute_code.size();

    fan_opengl_call(glShaderSource(shader.compute, 1, &ptr, &length));
    fan_opengl_call(glCompileShader(shader.compute));

    shader_check_compile_errors(internal_shader, file_path, "COMPUTE");
  }
  void context_t::shader_dispatch_compute(
    fan::graphics::shader_nr_t nr,
    std::uint32_t x,
    std::uint32_t y,
    std::uint32_t z
  ) {
    auto& shader = shader_get(nr);

    if (shader.id == (std::uint32_t)-1) {
      fan::print_impl("attempted to dispatch invalid compute shader");
      return;
    }

    shader_use(nr);

    fan_opengl_call(glDispatchCompute(x, y, z));
  }
  void context_t::parse_uniforms(
    const std::string& shader_data,
    std::unordered_map<std::string, std::string>& uniform_type_table
  ) {
    std::string s = shader_data;
    std::size_t i = 0;

    // remove comments
    while (i < s.size()) {
      if (s[i] == '/' && i + 1 < s.size()) {
        if (s[i + 1] == '/') { s.erase(i, s.find('\n', i) - i); continue; }
        if (s[i + 1] == '*') {
          std::size_t end = s.find("*/", i + 2);
          s.erase(i, (end == std::string::npos ? s.size() : end + 2) - i);
          continue;
        }
      }
      ++i;
    }

    // parse uniforms
    i = 0;
    while ((i = s.find("uniform", i)) != std::string::npos) {
      i += 7;
      while (i < s.size() && std::isspace(s[i])) ++i;
      std::size_t t_end = i; while (t_end < s.size() && (std::isalnum(s[t_end]) || s[t_end] == '_')) ++t_end;
      std::string type = s.substr(i, t_end - i);
      std::size_t n_start = t_end; while (n_start < s.size() && std::isspace(s[n_start])) ++n_start;
      std::size_t n_end = n_start; while (n_end < s.size() && (std::isalnum(s[n_end]) || s[n_end] == '_')) ++n_end;
      std::string name = s.substr(n_start, n_end - n_start);
      
      // Check for array before semicolon
      std::size_t semi = s.find(';', n_end);
      if (s.find('[', n_end) < semi) {
        type += "[]";
      }

      i = (semi == std::string::npos ? s.size() : semi + 1);
      uniform_type_table[name] = type;
    }
  }
  bool context_t::shader_compile(fan::graphics::shader_nr_t nr) {
    auto& shader = shader_get(nr);

    bool has_vertex = shader.vertex != (std::uint32_t)-1;
    bool has_fragment = shader.fragment != (std::uint32_t)-1;
    bool has_compute = shader.compute != (std::uint32_t)-1;

    if (has_compute && (has_vertex || has_fragment)) {
      fan::print_impl("compute shader cannot be linked with graphics shaders");
      return false;
    }

    auto check_shader = [&](GLuint id, const char* name) -> bool {
      GLint compiled = 0;
      fan_opengl_call(glGetShaderiv(id, GL_COMPILE_STATUS, &compiled));

      if (!compiled) {
        fan::print_impl(name, "shader not compiled successfully before linking");
        return false;
      }

      return true;
    };

    auto temp_id = fan_opengl_call(glCreateProgram());

    if (has_vertex) {
      if (!check_shader(shader.vertex, "vertex")) {
        fan_opengl_call(glDeleteProgram(temp_id));
        return false;
      }

      fan_opengl_call(glAttachShader(temp_id, shader.vertex));
    }

    if (has_fragment) {
      if (!check_shader(shader.fragment, "fragment")) {
        fan_opengl_call(glDeleteProgram(temp_id));
        return false;
      }

      fan_opengl_call(glAttachShader(temp_id, shader.fragment));
    }

    if (has_compute) {
      if (!check_shader(shader.compute, "compute")) {
        fan_opengl_call(glDeleteProgram(temp_id));
        return false;
      }

      fan_opengl_call(glAttachShader(temp_id, shader.compute));
    }

    if (!has_vertex && !has_fragment && !has_compute) {
      fan::print_impl("no shaders attached");
      fan_opengl_call(glDeleteProgram(temp_id));
      return false;
    }

    fan_opengl_call(glLinkProgram(temp_id));

    GLint success = 0;
    fan_opengl_call(glGetProgramiv(temp_id, GL_LINK_STATUS, &success));

    if (!success) {
      GLint buffer_size = 0;
      fan_opengl_call(glGetProgramiv(temp_id, GL_INFO_LOG_LENGTH, &buffer_size));

      std::string buffer;
      buffer.resize(buffer_size);

      if (buffer_size > 1) {
        fan_opengl_call(glGetProgramInfoLog(temp_id, buffer_size, nullptr, buffer.data()));
      }

      fan::print_impl("program link failed:", buffer);

      if (has_vertex) {
        fan::print_impl("=== vertex shader ===");
        fan::print_impl(__fan_internal_shader_list[nr].svertex);
      }

      if (has_fragment) {
        fan::print_impl("=== fragment shader ===");
        fan::print_impl(__fan_internal_shader_list[nr].sfragment);
      }

      if (has_compute) {
        fan::print_impl("=== compute shader ===");
        fan::print_impl(__fan_internal_shader_list[nr].scompute);
      }

      fan_opengl_call(glDeleteProgram(temp_id));
      return false;
    }

    auto cleanup_shader = [&](GLuint& id) {
      if (id == (std::uint32_t)-1) {
        return;
      }

      fan_opengl_call(glDetachShader(temp_id, id));
      fan_opengl_call(glDeleteShader(id));

      id = -1;
    };

    cleanup_shader(shader.vertex);
    cleanup_shader(shader.fragment);
    cleanup_shader(shader.compute);

    if (shader.id != (std::uint32_t)-1) {
      fan_opengl_call(glDeleteProgram(shader.id));
    }

    shader.id = temp_id;

    shader.projection_view[0] = fan_opengl_call(glGetUniformLocation(shader.id, "projection"));
    shader.projection_view[1] = fan_opengl_call(glGetUniformLocation(shader.id, "view"));

    auto& internal = __fan_internal_shader_list[nr];

    internal.uniform_type_table.clear();

    parse_uniforms(internal.svertex, internal.uniform_type_table);
    parse_uniforms(internal.sfragment, internal.uniform_type_table);
    parse_uniforms(internal.scompute, internal.uniform_type_table);

    return true;
  }
  fan::graphics::context_camera_t& context_t::camera_get(fan::graphics::camera_nr_t nr) {
    return __fan_internal_camera_list[nr];
  }
  void context_t::shader_set_camera(fan::graphics::shader_nr_t nr, fan::graphics::camera_nr_t camera_nr) {
    auto& camera = camera_get(camera_nr);
    fan_opengl_call(glUniformMatrix4fv(shader_get(nr).projection_view[0], 1, GL_FALSE, &camera.projection[0][0]));
    fan_opengl_call(glUniformMatrix4fv(shader_get(nr).projection_view[1], 1, GL_FALSE, &camera.view[0][0]));
  }

  fan::graphics::image_nr_t context_t::image_load_internal(
    fan::str_view_t path,
    const fan::opengl::context_t::image_load_properties_t& p,
    const std::source_location& callers_path
  ){
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

  void context_t::image_reload_internal(
    fan::graphics::image_nr_t nr,
    fan::str_view_t path,
    const fan::opengl::context_t::image_load_properties_t& p,
    const std::source_location& callers_path
  ){
    fan::image::info_t image_info;
    if (fan::image::load(path, &image_info, callers_path)) {
      image_info.data = (void*)fan::image::missing_texture_pixels;
      image_info.size = 2;
      image_info.channels = 4;
      image_info.type = -1;
    }
    image_reload(nr, image_info, p);
    __fan_internal_image_list[nr].image_path = path;
    fan::image::free(&image_info);
  }

  void context_t::image_clear_cache(){
    for (auto& [path, entry] : image_cache) {
      auto handle = image_get_handle(entry.nr);
      fan_opengl_call(glDeleteTextures(1, (GLuint*)&handle));
      delete static_cast<fan::opengl::context_t::image_t*>(__fan_internal_image_list[entry.nr].internal);
      __fan_internal_image_list.Recycle(entry.nr);
    }
    image_cache.clear();
  }

  fan::opengl::context_t::image_t& context_t::image_get(fan::graphics::image_nr_t nr) {
    return *(fan::opengl::context_t::image_t*)__fan_internal_image_list[nr].internal;
  }

  GLuint& context_t::image_get_handle(fan::graphics::image_nr_t nr) {
    return image_get(nr).texture_id;
  }

  fan::graphics::image_nr_t context_t::image_create() {
    fan::graphics::image_nr_t nr = __fan_internal_image_list.NewNode();
    __fan_internal_image_list[nr].internal = new fan::opengl::context_t::image_t;
    fan_opengl_call(glGenTextures(1, &image_get_handle(nr)));
    return nr;
  }

  void context_t::image_erase(fan::graphics::image_nr_t nr){
    auto& image_data = __fan_internal_image_list[nr];
    if (!image_data.image_path.empty()) {
      auto it = image_cache.find(image_data.image_path);
      if (it != image_cache.end()) {
        if (--it->second.ref_count > 0) {
          return;
        }
        image_cache.erase(it);
      }
    }
    auto handle = image_get_handle(nr);
    fan_opengl_call(glDeleteTextures(1, (GLuint*)&handle));
    delete static_cast<fan::opengl::context_t::image_t*>(__fan_internal_image_list[nr].internal);
    __fan_internal_image_list.Recycle(nr);
  }

  void context_t::image_bind(fan::graphics::image_nr_t nr) {
    fan_opengl_call(glBindTexture(GL_TEXTURE_2D, image_get_handle(nr)));
  }

  void context_t::image_bind(fan::graphics::image_nr_t nr, std::uint32_t unit) {
    fan_opengl_call(glActiveTexture(GL_TEXTURE0 + unit));
    fan_opengl_call(glBindTexture(GL_TEXTURE_2D, image_get_handle(nr)));
  }

  void context_t::image_bind(
    fan::graphics::image_t nr,
    uint32_t unit,
    GLenum access,
    GLenum format
  ) {
    fan_opengl_call(
      glBindImageTexture(
        unit,
        image_get_handle(nr),
        0,
        GL_FALSE,
        0,
        access,
        format
      )
    );
  }

  void context_t::image_unbind(fan::graphics::image_nr_t nr) {
    fan_opengl_call(glBindTexture(GL_TEXTURE_2D, 0));
  }

  fan::graphics::image_load_properties_t& context_t::image_get_settings(fan::graphics::image_nr_t nr) {
    return __fan_internal_image_list[nr].image_settings;
  }

  void context_t::image_set_settings(fan::graphics::image_nr_t nr, const fan::opengl::context_t::image_load_properties_t& p) {
    image_bind(nr);
  #if FAN_DEBUG >= fan_debug_high
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

  fan::graphics::image_nr_t context_t::image_create(void* data, const fan::vec2ui& size, const fan::opengl::context_t::image_load_properties_t& p) {
    fan::graphics::image_nr_t nr = image_create();
    image_bind(nr);
    image_set_settings(nr, p);

    auto& image_data = __fan_internal_image_list[nr];
    image_data.size = size;
    image_data.image_path = "";

    std::uint32_t bytes_per_row = (int)(image_data.size.x * fan::graphics::get_channel_amount(opengl_to_global_format(p.format)));
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

    fan_opengl_call(glTexImage2D(GL_TEXTURE_2D, 0, p.internal_format, image_data.size.x, image_data.size.y, 0, p.format, p.type, data));

    switch (p.min_filter) {
    case GL_LINEAR_MIPMAP_LINEAR:
    case GL_NEAREST_MIPMAP_LINEAR:
    case GL_LINEAR_MIPMAP_NEAREST:
    case GL_NEAREST_MIPMAP_NEAREST: {
      fan_opengl_call(glGenerateMipmap(GL_TEXTURE_2D));
      break;
    }
    }

    return nr;
  }

  fan::graphics::image_nr_t context_t::image_load(const fan::image::info_t& image_info, const fan::opengl::context_t::image_load_properties_t& lp) {
    auto p = lp;
    if (image_info.channels > 0 && p.format == image_load_properties_defaults::format) {
      p.format = get_format_from_channels(image_info.channels);
    }
    else if (image_info.channels <= 0 && p.format != image_load_properties_defaults::format) {
    }
    else if (image_info.channels > 0 && p.format != image_load_properties_defaults::format) {
      int format_channels = fan::graphics::get_channel_amount(opengl_to_global_format(p.format));
      if (format_channels != image_info.channels) {
        fan::print_impl("Warning: Format/channels mismatch. Format specifies",
          format_channels, "channels but image_info specifies",
          image_info.channels, "channels. Using format specification.");
      }
    }
    return image_create(image_info.data, image_info.size, p);
  }

  fan::graphics::image_nr_t context_t::create_missing_texture() {
    fan::opengl::context_t::image_load_properties_t p;
    p.visual_output = GL_REPEAT;
    fan::graphics::image_nr_t nr = image_create((void*)fan::image::missing_texture_pixels, fan::vec2i(2, 2), p);
    __fan_internal_image_list[nr].image_settings = image_opengl_to_global(p);
    return nr;
  }

  fan::graphics::image_nr_t context_t::create_transparent_texture(fan::opengl::context_t& context) {
    fan::opengl::context_t::image_load_properties_t p;
    p.visual_output = GL_REPEAT;
    p.min_filter = GL_NEAREST;
    p.mag_filter = GL_NEAREST;
    return image_create((void*)fan::image::transparent_texture_pixels, fan::vec2i(2, 2), p);
  }

  fan::graphics::image_nr_t context_t::image_load(fan::str_view_t path, const fan::opengl::context_t::image_load_properties_t& p, const std::source_location& callers_path) {
    auto it = image_cache.find(std::string(path));
    if (it != image_cache.end()) {
      it->second.ref_count++;
      return it->second.nr;
    }
    auto nr = image_load_internal(path, p, callers_path);
    image_cache[std::string(path)] = {nr, 1};
    return nr;
  }

  fan::graphics::image_nr_t context_t::image_load(const fan::image::info_t& image_info) {
    return image_load(image_info, fan::opengl::context_t::image_load_properties_t());
  }

  fan::graphics::image_nr_t context_t::image_load(fan::color* colors, const fan::vec2ui& size_, const fan::opengl::context_t::image_load_properties_t& p) {
    fan::opengl::context_t::image_load_properties_t custom_p = p;
    custom_p.internal_format = GL_RGBA32F;
    custom_p.type = GL_FLOAT;
    return image_create((void*)colors, size_, custom_p);
  }

  fan::graphics::image_nr_t context_t::image_load(fan::color* colors, const fan::vec2ui& size_) {
    return image_load(colors, size_, fan::opengl::context_t::image_load_properties_t());
  }

  fan::graphics::image_nr_t context_t::image_load(fan::str_view_t path, const std::source_location& callers_path) {
    return image_load(path, fan::opengl::context_t::image_load_properties_t(), callers_path);
  }

  void context_t::image_unload(fan::graphics::image_nr_t nr) {
    image_erase(nr);
  }

  void context_t::image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info, const fan::opengl::context_t::image_load_properties_t& lp) {
    auto p = lp;
    if (image_info.channels > 0 && p.format == image_load_properties_defaults::format) {
      p.format = get_format_from_channels(image_info.channels);
    }
    else if (image_info.channels <= 0 && p.format != image_load_properties_defaults::format) {
    }
    else if (image_info.channels > 0 && p.format != image_load_properties_defaults::format) {
      int format_channels = get_format_from_channels(p.format);
      if (format_channels != image_info.channels) {
        fan::print_impl("Warning: Format/channels mismatch. Format specifies",
          format_channels, "channels but image_info specifies",
          image_info.channels, "channels. Using format specification.");
      }
    }

    image_bind(nr);
    image_set_settings(nr, p);

    std::uint32_t bytes_per_row = (int)(image_info.size.x * fan::graphics::get_channel_amount(opengl_to_global_format(p.format)));
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

    switch (p.min_filter) {
    case GL_LINEAR_MIPMAP_LINEAR:
    case GL_NEAREST_MIPMAP_LINEAR:
    case GL_LINEAR_MIPMAP_NEAREST:
    case GL_NEAREST_MIPMAP_NEAREST: {
      fan_opengl_call(glGenerateMipmap(GL_TEXTURE_2D));
      break;
    }
    }
  }

  void context_t::image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info) {
    image_reload(nr, image_info, image_global_to_opengl(image_get_settings(nr)));
  }

  void context_t::image_reload(fan::graphics::image_nr_t nr, fan::str_view_t path, const fan::opengl::context_t::image_load_properties_t& p, const std::source_location& callers_path) {
    auto& image_data = __fan_internal_image_list[nr];
    auto it = image_cache.find(image_data.image_path);
    if (it != image_cache.end() && it->second.ref_count > 1) {
      return;
    }
    image_reload_internal(nr, path, p, callers_path);
  }

  void context_t::image_reload(fan::graphics::image_nr_t nr, fan::str_view_t path) {
    image_reload(nr, path, fan::opengl::context_t::image_load_properties_t());
  }

  std::vector<std::uint8_t> context_t::image_get_pixel_data(fan::graphics::image_nr_t nr, GLenum format, fan::vec2 uvp, fan::vec2 uvs) {
#if defined(__wasm__)
    fan::print_impl("glGetTexImage is not supported in WebGL. Use a Framebuffer + glReadPixels instead.");
    return {}; 
#else
    image_bind(nr);
    auto& image_data = __fan_internal_image_list[nr];
    
    std::uint32_t channels = fan::graphics::get_channel_amount(opengl_to_global_format(format));
    
    int px = fan::math::clamp((int)fan::math::round(uvp.x * image_data.size.x), 0, (int)image_data.size.x);
    int py = fan::math::clamp((int)fan::math::round(uvp.y * image_data.size.y), 0, (int)image_data.size.y);
    int pw = fan::math::min((int)fan::math::round(uvs.x * image_data.size.x), (int)image_data.size.x - px);
    int ph = fan::math::min((int)fan::math::round(uvs.y * image_data.size.y), (int)image_data.size.y - py);

    std::vector<std::uint8_t> full_data(image_data.size.x * image_data.size.y * channels);
    fan_opengl_call(glGetTexImage(GL_TEXTURE_2D, 0, format, GL_UNSIGNED_BYTE, full_data.data()));

    std::vector<std::uint8_t> result_data(pw * ph * channels);
    for (int row = 0; row < ph; ++row) {
      std::memcpy(&result_data[row * pw * channels],
                  &full_data[((py + row) * (int)image_data.size.x + px) * channels],
                  pw * channels);
    }
    
    return result_data;
#endif
  }

  fan::graphics::image_nr_t context_t::image_create(const fan::color& color, const fan::opengl::context_t::image_load_properties_t& p) {
    std::uint8_t pixels[4];
    for (std::uint32_t i = 0; i < fan::color::size(); i++) {
      pixels[i] = color[i] * 255;
    }
    return image_create(pixels, fan::vec2ui(1, 1), p);
  }

  fan::graphics::image_nr_t context_t::image_create(const fan::color& color) {
    return image_create(color, fan::opengl::context_t::image_load_properties_t());
  }

  fan::graphics::camera_nr_t context_t::camera_create() { return __fan_internal_camera_list.NewNode(); }
  void context_t::camera_erase(fan::graphics::camera_nr_t nr) { __fan_internal_camera_list.Recycle(nr); }
  void context_t::camera_set_ortho(fan::graphics::camera_nr_t nr, fan::vec2 x, fan::vec2 y) {
    camera_get(nr).coordinates.v = fan::vec4(x, y);
    camera_update_projection(nr);
    camera_update_view(nr);
  }
  void context_t::camera_update_projection(fan::graphics::camera_nr_t nr) {
    auto& camera = camera_get(nr);

    camera.projection = fan::math::ortho<fan::mat4>(
      camera.coordinates.left / camera.zoom,
      camera.coordinates.right / camera.zoom,
      camera.coordinates.bottom / camera.zoom,
      camera.coordinates.top / camera.zoom,
      0.1f,
      fan::graphics::znearfar
    );
  }
  void context_t::camera_update_view(fan::graphics::camera_nr_t nr) {
    auto& camera = camera_get(nr);
    camera.view[3][0] = 0;
    camera.view[3][1] = 0;
    camera.view[3][2] = 0;
    camera.view = camera.view.translate(camera.position);
    fan::vec3 position = camera.view.get_translation();
    constexpr fan::vec3 front(0, 0, 1);
    camera.view = fan::math::look_at_left<fan::mat4, fan::vec3>(position, position + front, fan::camera::world_up);
  }
  fan::graphics::camera_nr_t context_t::camera_create(const fan::vec2& x, const fan::vec2& y) {
    fan::graphics::camera_nr_t nr = camera_create();
    camera_set_ortho(nr, fan::vec2(x.x, x.y), fan::vec2(y.x, y.y));
    return nr;
  }
  fan::vec3 context_t::camera_get_position(fan::graphics::camera_nr_t nr) { return camera_get(nr).position; }
  void context_t::camera_set_position(fan::graphics::camera_nr_t nr, const fan::vec3& cp) {
    auto& camera = camera_get(nr);
    camera.position = cp;
    camera.view[3][0] = 0;
    camera.view[3][1] = 0;
    camera.view[3][2] = 0;
    camera.view = camera.view.translate(camera.position);
    fan::vec3 position = camera.view.get_translation();
    constexpr fan::vec3 front(0, 0, 1);
    camera.view = fan::math::look_at_left<fan::mat4, fan::vec3>(position, position + front, fan::camera::world_up);
  }
  fan::vec3 context_t::camera_get_center(fan::graphics::camera_nr_t nr) {
    auto& c = camera_get(nr);
    fan::vec2 center_offset = fan::vec2(
      c.coordinates.left + c.coordinates.right,
      c.coordinates.top + c.coordinates.bottom
    ) / (2.f * c.zoom);
    return fan::vec2(c.position.x, c.position.y) + center_offset;
  }
  void context_t::camera_set_center(fan::graphics::camera_nr_t nr, const fan::vec3& cp) {
    auto& c = camera_get(nr);
    fan::vec2 center_offset = fan::vec2(
      c.coordinates.left + c.coordinates.right,
      c.coordinates.top + c.coordinates.bottom
    ) / (2.f * c.zoom);

    camera_set_position(nr, fan::vec3(cp.xy() - center_offset, cp.z));
  }
  fan::vec2 context_t::camera_get_size(fan::graphics::camera_nr_t nr) {
    auto& camera = camera_get(nr);
    return fan::vec2(std::abs(camera.coordinates.right - camera.coordinates.left), std::abs(camera.coordinates.bottom - camera.coordinates.top));
  }
  f32_t context_t::camera_get_zoom(fan::graphics::camera_nr_t nr) {
    return camera_get(nr).zoom;
  }
  void context_t::camera_set_zoom(fan::graphics::camera_nr_t nr, f32_t new_zoom) {
    camera_get(nr).zoom = new_zoom;
    camera_update_projection(nr);
    camera_update_view(nr);
  }
  void context_t::camera_set_perspective(fan::graphics::camera_nr_t nr, f32_t fov, const fan::vec2& window_size) {
    auto& camera = camera_get(nr);
    camera.projection = fan::math::perspective<fan::mat4>(fan::math::radians(fov), (f32_t)window_size.x / (f32_t)window_size.y, camera.znear, camera.zfar);
    camera.update_view();
    camera.view = camera.get_view_matrix();
  }
  void context_t::camera_rotate(fan::graphics::camera_nr_t nr, const fan::vec2& offset) {
    auto& camera = camera_get(nr);
    camera.rotate_camera(offset);
    camera.view = camera.get_view_matrix();
  }
  void context_t::viewport_set(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
    fan_opengl_call(glViewport(viewport_position_.x, window_size.y - viewport_size_.y - viewport_position_.y, viewport_size_.x, viewport_size_.y));
  }
  fan::graphics::context_viewport_t& context_t::viewport_get(fan::graphics::viewport_nr_t nr) { return __fan_internal_viewport_list[nr]; }
  void context_t::viewport_set(fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
    auto& viewport = viewport_get(nr);
    viewport.position = viewport_position_;
    viewport.size = viewport_size_;
    viewport_set(viewport_position_, viewport_size_, window_size);
  }
  fan::graphics::viewport_nr_t context_t::viewport_create() {
    auto nr = __fan_internal_viewport_list.NewNode();
    viewport_set(nr, 0, 0, 0);
    return nr;
  }
  void context_t::viewport_erase(fan::graphics::viewport_nr_t nr) { __fan_internal_viewport_list.Recycle(nr); }
  fan::vec2 context_t::viewport_get_position(fan::graphics::viewport_nr_t nr) { return viewport_get(nr).position; }
  fan::vec2 context_t::viewport_get_size(fan::graphics::viewport_nr_t nr) { return viewport_get(nr).size; }
  void context_t::viewport_zero(fan::graphics::viewport_nr_t nr) {
    auto& viewport = viewport_get(nr);
    viewport.position = 0;
    viewport.size = 0;
    fan_opengl_call(glViewport(0, 0, 0, 0));
  }
  bool context_t::viewport_inside(fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
    auto& viewport = viewport_get(nr);
    return fan_2d::collision::rectangle::point_inside_no_rotation(position, viewport.position + viewport.size / 2, viewport.size / 2);
  }
  bool context_t::viewport_inside_wir(fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
    auto& viewport = viewport_get(nr);
    return fan_2d::collision::rectangle::point_inside_no_rotation(position, viewport.size / 2, viewport.size / 2);
  }

  std::uint32_t context_t::global_to_opengl_format(std::uintptr_t format) {
  #if defined(__wasm__)
    // WebGL/GLES does not support BGRA or BGR natively
    if (format == fan::graphics::image_format_e::b8g8r8a8_unorm) return GL_RGBA;
    if (format == fan::graphics::image_format_e::bgr_unorm) return GL_RGB;
  #else
    if (format == fan::graphics::image_format_e::b8g8r8a8_unorm) return GL_BGRA;
    if (format == fan::graphics::image_format_e::bgr_unorm) return GL_BGR;
  #endif

    if (format == fan::graphics::image_format_e::r8b8g8a8_unorm) return GL_RGBA;
    if (format == fan::graphics::image_format_e::r8_unorm) return GL_RED;
    if (format == fan::graphics::image_format_e::r32_float) return GL_R32F;
    if (format == fan::graphics::image_format_e::rg8_unorm) return GL_RG;
    if (format == fan::graphics::image_format_e::rgb_unorm) return GL_RGB;
    if (format == fan::graphics::image_format_e::rgba_unorm) return GL_RGBA;
    if (format == fan::graphics::image_format_e::r8_uint) return GL_RED_INTEGER;
    if (format == fan::graphics::image_format_e::r8g8b8a8_srgb) return GL_SRGB8_ALPHA8;
    if (format == fan::graphics::image_format_e::r11f_g11f_b10f) return GL_R11F_G11F_B10F;

  #if FAN_DEBUG >= fan_debug_high
    fan::throw_error("invalid format");
  #endif

    return GL_RGBA;
  }

  std::uint32_t context_t::global_to_opengl_type(std::uintptr_t type) {
    if (type == fan::graphics::fan_unsigned_byte) return GL_UNSIGNED_BYTE;
    if (type == fan::graphics::fan_unsigned_int) return GL_UNSIGNED_INT;
    if (type == fan::graphics::fan_float) return GL_FLOAT;
  #if FAN_DEBUG >= fan_debug_high
    fan::throw_error("invalid format");
  #endif
    return 0;
  }

  std::uint32_t context_t::global_to_opengl_address_mode(std::uint32_t mode) {
    if (mode == fan::graphics::image_sampler_address_mode_e::repeat) return GL_REPEAT;
    if (mode == fan::graphics::image_sampler_address_mode_e::mirrored_repeat) return GL_MIRRORED_REPEAT;
    if (mode == fan::graphics::image_sampler_address_mode_e::clamp_to_edge) return GL_CLAMP_TO_EDGE;

  #if defined(__wasm__)
    // Fallback for modes not supported in standard WebGL 2 / GLES 3.0
    if (mode == fan::graphics::image_sampler_address_mode_e::clamp_to_border) return GL_CLAMP_TO_EDGE;
    if (mode == fan::graphics::image_sampler_address_mode_e::mirrored_clamp_to_edge) return GL_CLAMP_TO_EDGE;
  #else
    if (mode == fan::graphics::image_sampler_address_mode_e::clamp_to_border) return GL_CLAMP_TO_BORDER;
    if (mode == fan::graphics::image_sampler_address_mode_e::mirrored_clamp_to_edge) return GL_MIRROR_CLAMP_TO_EDGE;
  #endif

  #if FAN_DEBUG >= fan_debug_high
    fan::throw_error("invalid format");
  #endif

    return GL_REPEAT;
  }

  std::uint32_t context_t::global_to_opengl_filter(std::uintptr_t filter) {
    using filter_t = fan::graphics::image_filter_e;
    switch (filter) {
    case filter_t::nearest: return GL_NEAREST;
    case filter_t::linear: return GL_LINEAR;
    case filter_t::nearest_mipmap_nearest: return GL_NEAREST_MIPMAP_NEAREST;
    case filter_t::linear_mipmap_nearest: return GL_LINEAR_MIPMAP_NEAREST;
    case filter_t::nearest_mipmap_linear: return GL_NEAREST_MIPMAP_LINEAR;
    case filter_t::linear_mipmap_linear: return GL_LINEAR_MIPMAP_LINEAR;
    default:
    #if FAN_DEBUG >= fan_debug_high
      fan::throw_error("Invalid image filter value");
    #endif
      return GL_NEAREST;
    }
  }

  std::uint32_t context_t::opengl_to_global_format(std::uintptr_t format) {
  #if !defined(__wasm__)
    if (format == GL_BGRA) return fan::graphics::image_format_e::b8g8r8a8_unorm;
    if (format == GL_BGR) return fan::graphics::image_format_e::bgr_unorm;
  #endif
    if (format == GL_RGBA) return fan::graphics::image_format_e::r8b8g8a8_unorm;
    if (format == GL_RED) return fan::graphics::image_format_e::r8_unorm;
    if (format == GL_R32F) return fan::graphics::image_format_e::r32_float;
    if (format == GL_RG) return fan::graphics::image_format_e::rg8_unorm;
    if (format == GL_RGB) return fan::graphics::image_format_e::rgb_unorm;
    if (format == GL_RED_INTEGER) return fan::graphics::image_format_e::r8_uint;
    if (format == GL_SRGB8_ALPHA8) return fan::graphics::image_format_e::r8g8b8a8_srgb;
    if (format == GL_R11F_G11F_B10F) return fan::graphics::image_format_e::r11f_g11f_b10f;

  #if FAN_DEBUG >= fan_debug_high
    fan::throw_error("invalid format");
  #endif
    return fan::graphics::image_format_e::rgba_unorm;
  }

  std::uint32_t context_t::opengl_to_global_type(std::uintptr_t type) {
    if (type == GL_UNSIGNED_BYTE) return fan::graphics::fan_unsigned_byte;
    if (type == GL_UNSIGNED_INT) return fan::graphics::fan_unsigned_int;
    if (type == GL_FLOAT) return fan::graphics::fan_float;

  #if FAN_DEBUG >= fan_debug_high
    fan::throw_error("invalid format");
  #endif
    return 0;
  }

  std::uint32_t context_t::opengl_to_global_address_mode(std::uint32_t mode) {
    if (mode == GL_REPEAT) return fan::graphics::image_sampler_address_mode_e::repeat;
    if (mode == GL_MIRRORED_REPEAT) return fan::graphics::image_sampler_address_mode_e::mirrored_repeat;
    if (mode == GL_CLAMP_TO_EDGE) return fan::graphics::image_sampler_address_mode_e::clamp_to_edge;

  #if !defined(__wasm__)
    if (mode == GL_CLAMP_TO_BORDER) return fan::graphics::image_sampler_address_mode_e::clamp_to_border;
    if (mode == GL_MIRROR_CLAMP_TO_EDGE) return fan::graphics::image_sampler_address_mode_e::mirrored_clamp_to_edge;
  #endif

  #if FAN_DEBUG >= fan_debug_high
    fan::throw_error("invalid format");
  #endif
    return fan::graphics::image_sampler_address_mode_e::repeat;
  }

  std::uint32_t context_t::opengl_to_global_filter(std::uintptr_t filter) {
    if (filter == GL_NEAREST) return fan::graphics::image_filter_e::nearest;
    if (filter == GL_LINEAR) return fan::graphics::image_filter_e::linear;
    if (filter == GL_NEAREST_MIPMAP_NEAREST) return fan::graphics::image_filter_e::nearest_mipmap_nearest;
    if (filter == GL_LINEAR_MIPMAP_NEAREST) return fan::graphics::image_filter_e::linear_mipmap_nearest;
    if (filter == GL_NEAREST_MIPMAP_LINEAR) return fan::graphics::image_filter_e::nearest_mipmap_linear;
    if (filter == GL_LINEAR_MIPMAP_LINEAR) return fan::graphics::image_filter_e::linear_mipmap_linear;

  #if FAN_DEBUG >= fan_debug_high
    fan::throw_error("Invalid OpenGL filter value.");
  #endif
    return fan::graphics::image_filter_e::nearest;
  }
  fan::opengl::context_t::image_load_properties_t context_t::image_global_to_opengl(const fan::graphics::image_load_properties_t& p) {
    return {
      .visual_output = global_to_opengl_address_mode(p.visual_output),
      .internal_format = global_to_opengl_format(p.internal_format),
      .format = global_to_opengl_format(p.format),
      .type = global_to_opengl_type(p.type),
      .min_filter = global_to_opengl_filter(p.min_filter),
      .mag_filter = global_to_opengl_filter(p.mag_filter),
    };
  }
  fan::graphics::image_load_properties_t context_t::image_opengl_to_global(const fan::opengl::context_t::image_load_properties_t& p) {
    return {
      .visual_output = opengl_to_global_address_mode(p.visual_output),
      .internal_format = opengl_to_global_format(p.internal_format),
      .format = opengl_to_global_format(p.format),
      .type = opengl_to_global_type(p.type),
      .min_filter = opengl_to_global_filter(p.min_filter),
      .mag_filter = opengl_to_global_filter(p.mag_filter),
    };
  }
}

namespace fan::opengl::core {
  int get_buffer_size(fan::opengl::context_t& context, GLenum target_buffer, GLuint buffer_object) {
    int size = 0;
    fan_opengl_call(glBindBuffer(target_buffer, buffer_object));
    fan_opengl_call(glGetBufferParameteriv(target_buffer, GL_BUFFER_SIZE, &size));
    return size;
  }

  void write_glbuffer(fan::opengl::context_t& context, GLuint buffer, const void* data, std::uintptr_t size, std::uint32_t usage, GLenum target) {
    //fan::print_impl("write_glbuffer", buffer);
    fan_opengl_call(glBindBuffer(target, buffer));
    fan_opengl_call(glBufferData(target, size, data, usage));
  }

  void get_glbuffer(fan::opengl::context_t& context, void* data, GLuint buffer_id, std::uintptr_t size, std::uintptr_t offset, GLenum target) {
  #if defined(__wasm__)
    fan::throw_error("unsupported func.");
  #else
    fan_opengl_call(glBindBuffer(target, buffer_id));
    fan_opengl_call(glGetBufferSubData(target, offset, size, data));
  #endif
  }

  void edit_glbuffer(fan::opengl::context_t& context, GLuint buffer, const void* data, std::uintptr_t offset, std::uintptr_t size, std::uintptr_t target) {
    fan_opengl_call(glBindBuffer(target, buffer));
    //fan::print_impl("edit_glbuffer", buffer);
#if FAN_DEBUG >= fan_debug_high
    int buffer_size = get_buffer_size(context, target, buffer);
    if ((buffer_size < (int)size) || (int)(offset + size) > buffer_size) {
      fan::throw_error("tried to write more than allocated");
    }
#endif
    fan_opengl_call(glBufferSubData(target, offset, size, data));
  }

  int get_bound_buffer(fan::opengl::context_t& context) {
  #if defined(__wasm__)
    fan::throw_error("unsupported func.");
    return -1;
  #else
    int buffer_id;
    fan_opengl_call(glGetIntegerv(GL_VERTEX_BINDING_BUFFER, &buffer_id));
    return buffer_id;
  #endif
  }

   void reserve_glbuffer(
    fan::opengl::context_t& ctx,
    GLuint buffer,
    std::uintptr_t& capacity,
    std::uintptr_t needed,
    std::uint32_t usage,
    GLenum target
  ){
    if (needed <= capacity) {
      return;
    }

    std::uintptr_t new_capacity = capacity ? capacity * 2 : 4096;
    if (new_capacity < needed) {
      new_capacity = needed;
    }

    fan::opengl::core::write_glbuffer(ctx, buffer, nullptr, new_capacity, usage, target);
    capacity = new_capacity;
  }

  void append_glbuffer(
    fan::opengl::context_t& ctx,
    GLuint buffer,
    std::uintptr_t& size_bytes,
    std::uintptr_t& capacity_bytes,
    const void* data,
    std::uintptr_t data_size,
    std::uint32_t usage,
    GLenum target
  ){
    reserve_glbuffer(ctx, buffer, capacity_bytes, size_bytes + data_size, usage, target);

    fan::opengl::core::edit_glbuffer(
      ctx,
      buffer,
      data,
      size_bytes,
      data_size,
      target
    );

    size_bytes += data_size;
  }


  void vao_t::open(fan::opengl::context_t& context) {
    fan_opengl_call(glGenVertexArrays(1, &m_buffer));
  }

  void vao_t::close(fan::opengl::context_t& context) {
    if (m_buffer == (decltype(m_buffer))-1) {
      return;
    }
    fan_opengl_call(glDeleteVertexArrays(1, &m_buffer));
    m_buffer = -1;
  }

  void vao_t::bind(fan::opengl::context_t& context) const {
    fan_opengl_call(glBindVertexArray(m_buffer));
  }

  void vao_t::unbind(fan::opengl::context_t& context) const {
    fan_opengl_call(glBindVertexArray(0));
  }

  bool vao_t::is_valid() const {
    return m_buffer != (GLuint)-1;
  }

  void vbo_t::open(fan::opengl::context_t& context, GLenum target_) {
    fan_opengl_call(glGenBuffers(1, &m_buffer));
    m_target = target_;
  }

  void vbo_t::close(fan::opengl::context_t& context) {
#if FAN_DEBUG >= fan_debug_medium
    if (m_buffer == (GLuint)-1) {
      return;
    }
#endif
    fan_opengl_call(glDeleteBuffers(1, &m_buffer));
    m_buffer = -1;
  }

  bool vbo_t::is_valid() const {
    return m_buffer != (GLuint)-1;
  }

  void vbo_t::bind(fan::opengl::context_t& context) const {
    fan_opengl_call(glBindBuffer(m_target, m_buffer));
  }

  void vbo_t::get_vram_instance(fan::opengl::context_t& context, void* data, std::uintptr_t size, std::uintptr_t offset) {
    fan::opengl::core::get_glbuffer(context, data, m_buffer, size, offset, m_target);
  }

  void vbo_t::bind_buffer_range(fan::opengl::context_t& context, std::uint32_t total_size) {
    fan_opengl_call(glBindBufferRange(GL_UNIFORM_BUFFER, 0, m_buffer, 0, total_size));
  }

  void vbo_t::edit_buffer(fan::opengl::context_t& context, const void* data, std::uintptr_t offset, std::uintptr_t size) {
    fan::opengl::core::edit_glbuffer(context, m_buffer, data, offset, size, m_target);
  }

  void vbo_t::write_buffer(fan::opengl::context_t& context, const void* data, std::uintptr_t size) {
    fan::opengl::core::write_glbuffer(context, m_buffer, data, size, m_usage, m_target);
  }

  void framebuffer_t::open(fan::opengl::context_t& context) {
    fan_opengl_call(glGenFramebuffers(1, &framebuffer));
  }

  void framebuffer_t::close(fan::opengl::context_t& context) {
    fan_opengl_call(glDeleteFramebuffers(1, &framebuffer));
  }

  void framebuffer_t::bind(fan::opengl::context_t& context) const {
    fan_opengl_call(glBindFramebuffer(GL_FRAMEBUFFER, framebuffer));
  }

  void framebuffer_t::unbind(fan::opengl::context_t& context) const {
    fan_opengl_call(glBindFramebuffer(GL_FRAMEBUFFER, 0));
  }

  bool framebuffer_t::ready(fan::opengl::context_t& context) const {
    return fan_opengl_call(glCheckFramebufferStatus(GL_FRAMEBUFFER)) == GL_FRAMEBUFFER_COMPLETE;
  }

  void framebuffer_t::bind_to_renderbuffer(fan::opengl::context_t& context, GLenum renderbuffer, const properties_t& p) {
    bind(context);
    fan_opengl_call(glFramebufferRenderbuffer(GL_FRAMEBUFFER, p.internalformat, GL_RENDERBUFFER, renderbuffer));
  }

  void framebuffer_t::bind_to_texture(fan::opengl::context_t& context, GLuint texture, GLenum attatchment) {
    fan_opengl_call(glFramebufferTexture2D(GL_FRAMEBUFFER, attatchment, GL_TEXTURE_2D, texture, 0));
  }

  void renderbuffer_t::open(fan::opengl::context_t& context) {
    fan_opengl_call(glGenRenderbuffers(1, &renderbuffer));
  }

  void renderbuffer_t::close(fan::opengl::context_t& context) {
    fan_opengl_call(glDeleteRenderbuffers(1, &renderbuffer));
  }

  void renderbuffer_t::bind(fan::opengl::context_t& context) const {
    fan_opengl_call(glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer));
  }

  void renderbuffer_t::set_storage(fan::opengl::context_t& context, const properties_t& p) const {
    bind(context);
    fan_opengl_call(glRenderbufferStorage(GL_RENDERBUFFER, p.internalformat, p.size.x, p.size.y));
  }

  void renderbuffer_t::bind_to_renderbuffer(fan::opengl::context_t& context, const properties_t& p) {
    fan_opengl_call(glFramebufferRenderbuffer(GL_FRAMEBUFFER, p.internalformat, GL_RENDERBUFFER, renderbuffer));
  }

  std::uint32_t get_draw_mode(std::uint8_t draw_mode) {
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
#if !defined(__wasm__)
    case fan::graphics::primitive_topology_t::lines_with_adjacency:
      return fan::opengl::context_t::primitive_topology_t::lines_with_adjacency;
    case fan::graphics::primitive_topology_t::line_strip_with_adjacency:
      return fan::opengl::context_t::primitive_topology_t::line_strip_with_adjacency;
    case fan::graphics::primitive_topology_t::triangles_with_adjacency:
      return fan::opengl::context_t::primitive_topology_t::triangles_with_adjacency;
    case fan::graphics::primitive_topology_t::triangle_strip_with_adjacency:
      return fan::opengl::context_t::primitive_topology_t::triangle_strip_with_adjacency;
#endif
    default:
      fan::throw_error("invalid draw mode");
      return -1;
    }
  }
}

namespace fan::graphics {
  fan::graphics::context_functions_t get_gl_context_functions() {
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
    cf.shader_set_vertex = [](void* context, fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& vertex_code) {
      ((fan::opengl::context_t*)context)->shader_set_vertex(nr, file_path, vertex_code);
    };
    cf.shader_set_fragment = [](void* context, fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& fragment_code) {
      ((fan::opengl::context_t*)context)->shader_set_fragment(nr, file_path, fragment_code);
    };
    cf.shader_set_compute = [](void* context, fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& compute_code) {
      ((fan::opengl::context_t*)context)->shader_set_compute(nr, file_path, compute_code);
    };
    cf.shader_dispatch_compute = [](void* context, fan::graphics::shader_nr_t nr, uint32_t x, uint32_t y, uint32_t z) {
      ((fan::opengl::context_t*)context)->shader_dispatch_compute(nr, x, y, z);
    };
    cf.shader_compile = [](void* context, fan::graphics::shader_nr_t nr) {
      return ((fan::opengl::context_t*)context)->shader_compile(nr);
    };
    /*image*/
    cf.image_create = [](void* context) {
      return ((fan::opengl::context_t*)context)->image_create();
    };
    cf.image_get_handle = [](void* context, fan::graphics::image_nr_t nr) {
      return (std::uint64_t)((fan::opengl::context_t*)context)->image_get_handle(nr);
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
    cf.image_bind_params = [](void* context, fan::graphics::image_nr_t nr, std::uint32_t unit, std::uint32_t access, std::uint32_t format) {
      ((fan::opengl::context_t*)context)->image_bind(nr, unit, access, format);
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
    cf.image_load_path = [](void* context, fan::str_view_t path, const std::source_location& callers_path = std::source_location::current()) {
      return ((fan::opengl::context_t*)context)->image_load(path, callers_path);
    };
    cf.image_load_path_props = [](void* context, fan::str_view_t path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current()) {
      return ((fan::opengl::context_t*)context)->image_load(path, ((fan::opengl::context_t*)context)->image_global_to_opengl(p), callers_path);
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
    cf.image_reload_path = [](void* context, fan::graphics::image_nr_t nr, fan::str_view_t path, const std::source_location& callers_path = std::source_location::current()) {
      return ((fan::opengl::context_t*)context)->image_reload(nr, path);
    };
    cf.image_reload_path_props = [](void* context, fan::graphics::image_nr_t nr, fan::str_view_t path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current()) {
      return ((fan::opengl::context_t*)context)->image_reload(nr, path, ((fan::opengl::context_t*)context)->image_global_to_opengl(p));
    };
    cf.image_create_color = [](void* context, const fan::color& color) {
      return ((fan::opengl::context_t*)context)->image_create(color);
    };
    cf.image_create_color_props = [](void* context, const fan::color& color, const fan::graphics::image_load_properties_t& p) {
      return ((fan::opengl::context_t*)context)->image_create(color, ((fan::opengl::context_t*)context)->image_global_to_opengl(p));
    };
    cf.image_create_data = [](void* context, void* data, const fan::vec2ui& size, const fan::graphics::image_load_properties_t& p) {
      return ((fan::opengl::context_t*)context)->image_create(data, size, ((fan::opengl::context_t*)context)->image_global_to_opengl(p));
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
    cf.camera_get_center = [](void* context, fan::graphics::camera_nr_t nr) {
      return ((fan::opengl::context_t*)context)->camera_get_center(nr);
    };
    cf.camera_set_center = [](void* context, fan::graphics::camera_nr_t nr, const fan::vec3& cp) {
      ((fan::opengl::context_t*)context)->camera_set_center(nr, cp);
    };
    cf.camera_get_size = [](void* context, fan::graphics::camera_nr_t nr) {
      return ((fan::opengl::context_t*)context)->camera_get_size(nr);
    };
    cf.camera_get_zoom = [](void* context, fan::graphics::camera_nr_t nr) {
      return ((fan::opengl::context_t*)context)->camera_get_zoom(nr);
    };
    cf.camera_set_zoom = [](void* context, fan::graphics::camera_nr_t nr, f32_t new_zoom) {
      ((fan::opengl::context_t*)context)->camera_set_zoom(nr, new_zoom);
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
    cf.image_read_pixels = [](void* context, fan::graphics::image_nr_t nr, fan::vec2 uv_pos, fan::vec2 uv_size) {
    #if defined(__wasm__)
      fan::throw_error("unsupported func.");
      return std::vector<std::uint8_t>{};
    #else
      auto& gl = *(fan::opengl::context_t*)context;
      auto size = fan::graphics::image_get_data(nr).size;
      auto img_settings = gl.image_get_settings(nr);
      std::uint32_t channels = fan::graphics::get_channel_amount(img_settings.format);
      int px = fan::math::clamp((int)fan::math::round(uv_pos.x * size.x), 0, (int)size.x);
      int py = fan::math::clamp((int)fan::math::round(uv_pos.y * size.y), 0, (int)size.y);
      int pw = fan::math::min((int)fan::math::round(uv_size.x * size.x), (int)size.x - px);
      int ph = fan::math::min((int)fan::math::round(uv_size.y * size.y), (int)size.y - py);
      std::vector<std::uint8_t> full(size.x * size.y * channels);
      glBindTexture(GL_TEXTURE_2D, gl.image_get_handle(nr));
      glGetTexImage(GL_TEXTURE_2D, 0,
        fan::opengl::context_t::global_to_opengl_format(img_settings.format),
        GL_UNSIGNED_BYTE, full.data());
      std::vector<std::uint8_t> out(pw * ph * channels);
      for (int row = 0; row < ph; ++row) {
        std::memcpy(&out[row * pw * channels],
          &full[((py + row) * (int)size.x + px) * channels],
          pw * channels);
      }
      return out;
    #endif
    };
    cf.image_get_pixel_data = [](void* context, fan::graphics::image_nr_t nr, std::uint32_t format, fan::vec2 uvp, fan::vec2 uvs) {
      return ((fan::opengl::context_t*)context)->image_get_pixel_data(nr,
        fan::opengl::context_t::global_to_opengl_format(format), uvp, uvs);
    };
    return cf;
  }
}

fan::opengl::context_t& fan::graphics::get_gl_context() {
  return (*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::ctx())));
}
#endif