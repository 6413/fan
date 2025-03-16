#pragma once

// build function with params, 2 no params
#define context_build_camera_functions(build_function) \
  CONCAT(build_function, 2)(camera_create, fan::graphics::camera_nr_t) \
  build_function(camera_get, fan::graphics::context_camera_t&, fan::graphics::camera_nr_t nr) \
  build_function(camera_erase, void, fan::graphics::camera_nr_t nr) \
  build_function(camera_open, fan::graphics::camera_nr_t, const fan::vec2& x, const fan::vec2& y) \
  build_function(camera_get_position, fan::vec3, fan::graphics::camera_nr_t nr) \
  build_function(camera_set_position, void, fan::graphics::camera_nr_t nr, const fan::vec3& cp) \
  build_function(camera_get_size, fan::vec2, fan::graphics::camera_nr_t nr) \
  build_function(camera_set_ortho, void, fan::graphics::camera_nr_t nr, fan::vec2 x, fan::vec2 y) \
  build_function(camera_set_perspective, void, fan::graphics::camera_nr_t nr, f32_t fov, const fan::vec2& window_size) \
  build_function(camera_rotate, void, fan::graphics::camera_nr_t nr, const fan::vec2& offset)

#define context_build_shader_functions(build_function) \
  CONCAT(build_function, 2)(shader_create, fan::graphics::shader_nr_t) \
  build_function(shader_get, void*, fan::graphics::shader_nr_t nr) \
  build_function(shader_erase, void, fan::graphics::shader_nr_t nr) \
  build_function(shader_use, void, fan::graphics::shader_nr_t nr) \
  build_function(shader_set_vertex, void, fan::graphics::shader_nr_t nr, const std::string& vertex_code) \
  build_function(shader_set_fragment, void, fan::graphics::shader_nr_t nr, const std::string& fragment_code) \
  build_function(shader_compile, bool, fan::graphics::shader_nr_t nr)

#define context_build_image_functions(build_function) \
  CONCAT(build_function, 2)(image_create, fan::graphics::image_nr_t) \
  build_function(image_get_handle, uint64_t, fan::graphics::image_nr_t nr) \
  build_function(image_get, void*, fan::graphics::image_nr_t nr) \
  build_function(image_erase, void, fan::graphics::image_nr_t nr) \
  build_function(image_bind, void, fan::graphics::image_nr_t nr) \
  build_function(image_unbind, void, fan::graphics::image_nr_t nr) \
  build_function(image_set_settings, void, const image_load_properties_t& p) \
  build_function(image_load_info, fan::graphics::image_nr_t, const fan::image::image_info_t& image_info) \
  build_function(image_load_info_props, fan::graphics::image_nr_t, const fan::image::image_info_t& image_info, const image_load_properties_t& p) \
  build_function(image_load_path, fan::graphics::image_nr_t, const std::string& path) \
  build_function(image_load_path_props, fan::graphics::image_nr_t, const std::string& path, const image_load_properties_t& p) \
  build_function(image_load_colors, fan::graphics::image_nr_t, fan::color* colors, const fan::vec2ui& size_) \
  build_function(image_load_colors_props, fan::graphics::image_nr_t, fan::color* colors, const fan::vec2ui& size_, const image_load_properties_t& p) \
  build_function(image_unload, void, fan::graphics::image_nr_t nr) \
  CONCAT(build_function, 2)(create_missing_texture, fan::graphics::image_nr_t) \
  CONCAT(build_function, 2)(create_transparent_texture, fan::graphics::image_nr_t) \
  build_function(image_reload_image_info, void, fan::graphics::image_nr_t nr, const fan::image::image_info_t& image_info) \
  build_function(image_reload_image_info_props, void, fan::graphics::image_nr_t nr, const fan::image::image_info_t& image_info, const image_load_properties_t& p) \
  build_function(image_reload_path, void, fan::graphics::image_nr_t nr, const std::string& path) \
  build_function(image_reload_path_props, void, fan::graphics::image_nr_t nr, const std::string& path, const image_load_properties_t& p) \
  build_function(image_get_pixel_data, std::unique_ptr<uint8_t[]>, fan::graphics::image_nr_t nr, uint32_t format, fan::vec2 uvp, fan::vec2 uvs) \
  build_function(image_create_color, fan::graphics::image_nr_t, const fan::color& color) \
  build_function(image_create_color_props, fan::graphics::image_nr_t, const fan::color& color, const fan::graphics::image_load_properties_t& p)

#define context_build_viewport_functions(build_function) \
  CONCAT(build_function, 2)(viewport_create, fan::graphics::viewport_nr_t) \
  build_function(viewport_get, fan::graphics::context_viewport_t&, fan::graphics::viewport_nr_t nr) \
  build_function(viewport_erase, void, fan::graphics::viewport_nr_t nr) \
  build_function(viewport_get_position, fan::vec2, fan::graphics::viewport_nr_t nr) \
  build_function(viewport_get_size, fan::vec2, fan::graphics::viewport_nr_t nr) \
  build_function(viewport_set, void, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) \
  build_function(viewport_set_nr, void, fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) \
  build_function(viewport_zero, void, fan::graphics::viewport_nr_t nr) \
  build_function(viewport_inside, bool, fan::graphics::viewport_nr_t nr, const fan::vec2& position) \
  build_function(viewport_inside_wir, bool, fan::graphics::viewport_nr_t nr, const fan::vec2& position)

#define context_typedef_func_ptr(name, ret_type, ...) \
  typedef ret_type (*name##_t)(void* ctx, __VA_ARGS__); \
  name##_t name;

#define context_typedef_func_ptr2(name, ret_type) \
  typedef ret_type (*name##_t)(void* ctx); \
  name##_t name;

#define context_declare_func(name, ret_type, ...) \
  ret_type name (__VA_ARGS__);
#define context_declare_func2 context_declare_func