module;

#include <fan/graphics/opengl/init.h>
#include <fan/event/types.h>

#include <source_location>
#include <cmath>
#include <coroutine>
#include <algorithm>
#include <filesystem>
#include <cstring>

module fan.graphics;

#if defined(FAN_JSON)
  import fan.types.json;
#endif
#if defined(FAN_GUI)
  import fan.graphics.gui.base;
#endif

namespace fan::window {
  void add_input_action(const int* keys, std::size_t count, const std::string& action_name) {
    fan::graphics::ctx().input_action->add(keys, count, action_name);
  }

  void add_input_action(std::initializer_list<int> keys, const std::string& action_name) {
    fan::graphics::ctx().input_action->add(keys, action_name);
  }

  void add_input_action(int key, const std::string& action_name) {
    fan::graphics::ctx().input_action->add(key, action_name);
  }

  bool is_input_action_active(const std::string& action_name, int pstate) {
    return fan::graphics::ctx().input_action->is_active(action_name);
  }

  bool is_action_clicked(const std::string& action_name) {
    return fan::graphics::ctx().input_action->is_active(action_name);
  }

  bool is_action_down(const std::string& action_name) {
    return fan::graphics::ctx().input_action->is_active(action_name, fan::window::input_action_t::press_or_repeat);
  }

  bool exists(const std::string& action_name) {
    return fan::graphics::ctx().input_action->input_actions.find(action_name) != fan::graphics::ctx().input_action->input_actions.end();
  }
}

namespace fan::graphics {
  fan::graphics::image_t invalid_image = [] {
    image_t image {false};
    image.sic();
    return image;
  }();

  fan::graphics::render_view_t add_render_view() {
    fan::graphics::render_view_t render_view;
    render_view.create();
    return render_view;
  }

  fan::graphics::render_view_t add_render_view(const fan::vec2& ortho_x, const fan::vec2& ortho_y, const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
    fan::graphics::render_view_t render_view;
    render_view.create();
    render_view.set(ortho_x, ortho_y, viewport_position, viewport_size, fan::graphics::get_window().get_size());
    return render_view;
  }

  std::vector<uint8_t> image_get_pixel_data(fan::graphics::image_nr_t nr, int image_format, fan::vec2 uvp, fan::vec2 uvs) {
    if (fan::graphics::get_window().renderer == fan::window_t::renderer_t::opengl) {
      return fan::graphics::ctx()->image_get_pixel_data(fan::graphics::ctx(), nr, fan::opengl::context_t::global_to_opengl_format(image_format), uvp, uvs);
    }
    else {
      fan::throw_error("");
      return {};
    }
  }

  fan::graphics::image_nr_t image_create() {
    return fan::graphics::ctx()->image_create(fan::graphics::ctx());
  }

  fan::graphics::context_image_t image_get(fan::graphics::image_nr_t nr) {
    fan::graphics::context_image_t img;
    if (fan::graphics::get_window().renderer == fan::window_t::renderer_t::opengl) {
      img.gl = *(fan::opengl::context_t::image_t*)fan::graphics::ctx()->image_get(fan::graphics::ctx(), nr);
    }
  #if defined(FAN_VULKAN)
    else if (fan::graphics::get_window().renderer == fan::window_t::renderer_t::vulkan) {
      img.vk = *(fan::vulkan::context_t::image_t*)fan::graphics::ctx()->image_get(fan::graphics::ctx(), nr);
    }
  #endif
    return img;
  }

  uint64_t image_get_handle(fan::graphics::image_nr_t nr) {
    return fan::graphics::ctx()->image_get_handle(fan::graphics::ctx(), nr);
  }

  void image_erase(fan::graphics::image_nr_t nr) {
    fan::graphics::ctx()->image_erase(fan::graphics::ctx(), nr);
  }

  void image_bind(fan::graphics::image_nr_t nr) {
    fan::graphics::ctx()->image_bind(fan::graphics::ctx(), nr);
  }

  void image_unbind(fan::graphics::image_nr_t nr) {
    fan::graphics::ctx()->image_unbind(fan::graphics::ctx(), nr);
  }

  fan::graphics::image_load_properties_t& image_get_settings(fan::graphics::image_nr_t nr) {
    return fan::graphics::ctx()->image_get_settings(fan::graphics::ctx(), nr);
  }

  void image_set_settings(fan::graphics::image_nr_t nr, const fan::graphics::image_load_properties_t& settings) {
    fan::graphics::ctx()->image_set_settings(fan::graphics::ctx(), nr, settings);
  }

  fan::graphics::image_nr_t image_load(const fan::image::info_t& image_info) {
    return fan::graphics::ctx()->image_load_info(fan::graphics::ctx(), image_info);
  }

  fan::graphics::image_nr_t image_load(const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p) {
    return fan::graphics::ctx()->image_load_info_props(fan::graphics::ctx(), image_info, p);
  }

  fan::graphics::image_nr_t image_load(const std::string& path, const std::source_location& callers_path) {
    return fan::graphics::ctx()->image_load_path(fan::graphics::ctx(), path, callers_path);
  }

  fan::graphics::image_nr_t image_load(const std::string& path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path) {
    return fan::graphics::ctx()->image_load_path_props(fan::graphics::ctx(), path, p, callers_path);
  }

  fan::graphics::image_nr_t image_load(fan::color* colors, const fan::vec2ui& size) {
    return fan::graphics::ctx()->image_load_colors(fan::graphics::ctx(), colors, size);
  }

  fan::graphics::image_nr_t image_load(fan::color* colors, const fan::vec2ui& size, const fan::graphics::image_load_properties_t& p) {
    return fan::graphics::ctx()->image_load_colors_props(fan::graphics::ctx(), colors, size, p);
  }

  void image_unload(fan::graphics::image_nr_t nr) {
    fan::graphics::ctx()->image_unload(fan::graphics::ctx(), nr);
  }

  bool is_image_valid(fan::graphics::image_nr_t nr) {
    return nr != fan::graphics::ctx().default_texture && nr.iic() == false;
  }

  fan::graphics::image_t image_load_pixel_art(const std::string& path) {
    return image_load(path, image_presets::pixel_art());
  }

  fan::graphics::image_t image_load_smooth(const std::string& path) {
    return image_load(path, image_presets::smooth());
  }

  fan::graphics::image_nr_t create_missing_texture() {
    return fan::graphics::ctx()->create_missing_texture(fan::graphics::ctx());
  }

  fan::graphics::image_nr_t create_transparent_texture() {
    return fan::graphics::ctx()->create_transparent_texture(fan::graphics::ctx());
  }

  void image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info) {
    fan::graphics::ctx()->image_reload_image_info(fan::graphics::ctx(), nr, image_info);
  }

  void image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p) {
    fan::graphics::ctx()->image_reload_image_info_props(fan::graphics::ctx(), nr, image_info, p);
  }

  void image_reload(fan::graphics::image_nr_t nr, const std::string& path, const std::source_location& callers_path) {
    fan::graphics::ctx()->image_reload_path(fan::graphics::ctx(), nr, path, callers_path);
  }

  void image_reload(fan::graphics::image_nr_t nr, const std::string& path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path) {
    fan::graphics::ctx()->image_reload_path_props(fan::graphics::ctx(), nr, path, p, callers_path);
  }

  fan::graphics::image_nr_t image_create(const fan::color& color) {
    return fan::graphics::ctx()->image_create_color(fan::graphics::ctx(), color);
  }

  fan::graphics::image_nr_t image_create(const fan::color& color, const fan::graphics::image_load_properties_t& p) {
    return fan::graphics::ctx()->image_create_color_props(fan::graphics::ctx(), color, p);
  }

  std::vector<uint8_t> read_pixels(const fan::vec2& position, const fan::vec2& size) {
    std::vector<uint8_t> pixels(size.multiply() * 4);
    glReadPixels(position.x, position.y, size.x, size.y, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    return pixels;
  }

  //std::vector<uint8_t> read_pixels_from_image(
  //  fan::graphics::image_nr_t nr, 
  //  const fan::vec2& uv_position, 
  //  const fan::vec2& uv_size) 
  //{
  //  auto img_settings = image_get_settings(nr);
  //  auto size = image_get_data(nr).size;
  //  uint32_t channels = fan::graphics::get_channel_amount(img_settings.format);

  //  int px = uv_position.x * size.x;
  //  int py = uv_position.y * size.y;
  //  int pw = uv_size.x * size.x;
  //  int ph = uv_size.y * size.y;

  //  px = std::max(0, std::min(px, (int)size.x - pw));
  //  py = std::max(0, std::min(py, (int)size.y - ph));
  //  pw = std::min(pw, (int)size.x - px);
  //  ph = std::min(ph, (int)size.y - py);

  //  std::vector<uint8_t> full(size.x * size.y * channels);
  //  glBindTexture(GL_TEXTURE_2D, image_get_handle(nr));

  //  glGetTexImage(GL_TEXTURE_2D, 0, 
  //    fan::opengl::context_t::global_to_opengl_format(img_settings.format),
  //    GL_UNSIGNED_BYTE, full.data());

  //  std::vector<uint8_t> out(pw * ph * channels);
  //  for (int row = 0; row < ph; ++row) {
  //    std::memcpy(&out[row * pw * channels], 
  //      &full[((py + row) * size.x + px) * channels], 
  //      pw * channels);
  //  }

  //  return out;
  //}

  std::vector<uint8_t> read_pixels_from_image(
    fan::graphics::image_nr_t nr, 
    const fan::vec2& uv_pos, 
    const fan::vec2& uv_size) 
  {
    auto size = image_get_data(nr).size;
    auto img_settings = image_get_settings(nr);
    uint32_t channels = fan::graphics::get_channel_amount(img_settings.format);

    int px = std::round(uv_pos.x * size.x);
    int py = std::round(uv_pos.y * size.y);
    int pw = std::round(uv_size.x * size.x);
    int ph = std::round(uv_size.y * size.y);

    px = std::clamp(px, 0, (int)size.x - pw);
    py = std::clamp(py, 0, (int)size.y - ph);
    pw = std::min(pw, (int)size.x - px);
    ph = std::min(ph, (int)size.y - py);

    std::vector<uint8_t> full(size.x * size.y * channels);
    glBindTexture(GL_TEXTURE_2D, image_get_handle(nr));

    glGetTexImage(GL_TEXTURE_2D, 0, 
      fan::opengl::context_t::global_to_opengl_format(img_settings.format),
      GL_UNSIGNED_BYTE, full.data());

    std::vector<uint8_t> out(pw * ph * channels);
    for (int row = 0; row < ph; ++row) {
      std::memcpy(&out[row * pw * channels], 
        &full[((py + row) * size.x + px) * channels], 
        pw * channels);
    }

    return out;
  }

  fan::graphics::shader_nr_t shader_create() {
    return fan::graphics::ctx()->shader_create(fan::graphics::ctx());
  }

  void shader_erase(fan::graphics::shader_nr_t nr) {
    fan::graphics::ctx()->shader_erase(fan::graphics::ctx(), nr);
  }

  void shader_use(fan::graphics::shader_nr_t nr) {
    fan::graphics::ctx()->shader_use(fan::graphics::ctx(), nr);
  }

  void shader_set_vertex(fan::graphics::shader_nr_t nr, const std::string& vertex_code) {
    fan::graphics::ctx()->shader_set_vertex(fan::graphics::ctx(), nr, vertex_code);
  }

  void shader_set_fragment(fan::graphics::shader_nr_t nr, const std::string& fragment_code) {
    fan::graphics::ctx()->shader_set_fragment(fan::graphics::ctx(), nr, fragment_code);
  }

  bool shader_compile(fan::graphics::shader_nr_t nr) {
    return fan::graphics::ctx()->shader_compile(fan::graphics::ctx(), nr);
  }

#if defined(FAN_2D)
  fan::graphics::shader_nr_t shader_get_nr(uint16_t shape_type) {
    return fan::graphics::get_shapes().shaper.GetShader(shape_type);
  }

  fan::graphics::shader_list_t::nd_t& shader_get_data(uint16_t shape_type) {
    return (*fan::graphics::ctx().shader_list)[shader_get_nr(shape_type)];
  }

  bool shader_update_fragment(uint16_t shape_type, const std::string& fragment) {
    auto shader_nr = shader_get_nr(shape_type);
    auto shader_data = shader_get_data(shape_type);
    shader_set_vertex(shader_nr, shader_data.svertex);
    shader_set_fragment(shader_nr, fragment);
    return shader_compile(shader_nr);
  }
#endif

  fan::graphics::camera_nr_t camera_create() {
    return fan::graphics::ctx()->camera_create(fan::graphics::ctx());
  }

  fan::graphics::context_camera_t& camera_get(fan::graphics::camera_nr_t nr) {
    return fan::graphics::ctx()->camera_get(fan::graphics::ctx(), nr);
  }

  void camera_erase(fan::graphics::camera_nr_t nr) {
    fan::graphics::ctx()->camera_erase(fan::graphics::ctx(), nr);
  }

  fan::graphics::camera_nr_t camera_create(const fan::vec2& x, const fan::vec2& y) {
    return fan::graphics::ctx()->camera_create_params(fan::graphics::ctx(), x, y);
  }

  fan::vec3 camera_get_position(fan::graphics::camera_nr_t nr) {
    return fan::graphics::ctx()->camera_get_position(fan::graphics::ctx(), nr);
  }

  void camera_set_position(fan::graphics::camera_nr_t nr, const fan::vec3& cp) {
    fan::graphics::ctx()->camera_set_position(fan::graphics::ctx(), nr, cp);
  }

  fan::vec2 camera_get_size(fan::graphics::camera_nr_t nr) {
    return fan::graphics::ctx()->camera_get_size(fan::graphics::ctx(), nr);
  }

  fan::vec2 viewport_get_size(fan::graphics::viewport_nr_t nr) {
    return fan::graphics::ctx()->viewport_get_size(fan::graphics::ctx(), nr);
  }

  f32_t camera_get_zoom(fan::graphics::camera_nr_t nr) {
    return fan::graphics::ctx()->camera_get_zoom(fan::graphics::ctx(), nr);
  }
  void camera_set_zoom(fan::graphics::camera_nr_t nr, f32_t new_zoom) {
    fan::graphics::ctx()->camera_set_zoom(fan::graphics::ctx(), nr, new_zoom);
  }

  void camera_set_ortho(fan::graphics::camera_nr_t nr, fan::vec2 x, fan::vec2 y) {
    fan::graphics::ctx()->camera_set_ortho(fan::graphics::ctx(), nr, x, y);
  }

  void camera_set_perspective(fan::graphics::camera_nr_t nr, f32_t fov, const fan::vec2& window_size) {
    fan::graphics::ctx()->camera_set_perspective(fan::graphics::ctx(), nr, fov, window_size);
  }

  void camera_rotate(fan::graphics::camera_nr_t nr, const fan::vec2& offset) {
    fan::graphics::ctx()->camera_rotate(fan::graphics::ctx(), nr, offset);
  }

  void camera_set_target(fan::graphics::camera_nr_t nr, const fan::vec2& target, f32_t move_speed) {
    f32_t screen_height = fan::graphics::get_window().get_size()[1];
    f32_t pixels_from_bottom = 400.0f;
    fan::vec2 src = camera_get_position(fan::graphics::get_orthographic_render_view().camera);
    camera_set_position(
      fan::graphics::get_orthographic_render_view().camera,
      move_speed == 0 ? target : src + (target - src) * fan::graphics::get_window().m_delta_time * move_speed
    );
  }
  void camera_look_at(fan::graphics::camera_nr_t nr, const fan::vec2& target, f32_t move_speed) {
    camera_set_target(nr, target, move_speed);
  }
  void camera_look_at(const fan::vec2& target, f32_t move_speed) {
    camera_set_target(fan::graphics::get_orthographic_render_view().camera, target, move_speed);
  }

  fan::graphics::viewport_nr_t viewport_create() {
    return fan::graphics::ctx()->viewport_create(fan::graphics::ctx());
  }

  fan::graphics::viewport_nr_t viewport_create(const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
    return fan::graphics::ctx()->viewport_create_params(fan::graphics::ctx(), viewport_position, viewport_size, fan::graphics::get_window().get_size());
  }

  fan::graphics::context_viewport_t& viewport_get(fan::graphics::viewport_nr_t nr) {
    return fan::graphics::ctx()->viewport_get(fan::graphics::ctx(), nr);
  }

  void viewport_erase(fan::graphics::viewport_nr_t nr) {
    fan::graphics::ctx()->viewport_erase(fan::graphics::ctx(), nr);
  }

  fan::vec2 viewport_get_position(fan::graphics::viewport_nr_t nr) {
    return fan::graphics::ctx()->viewport_get_position(fan::graphics::ctx(), nr);
  }

  void viewport_set(const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
    fan::graphics::ctx()->viewport_set(fan::graphics::ctx(), viewport_position, viewport_size, fan::graphics::get_window().get_size());
  }

  void viewport_set(fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
    fan::graphics::ctx()->viewport_set_nr(fan::graphics::ctx(), nr, viewport_position, viewport_size, fan::graphics::get_window().get_size());
  }

  void viewport_zero(fan::graphics::viewport_nr_t nr) {
    fan::graphics::ctx()->viewport_zero(fan::graphics::ctx(), nr);
  }

  bool inside(fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
    return fan::graphics::ctx()->viewport_inside(fan::graphics::ctx(), nr, position);
  }

  bool inside_wir(fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
    return fan::graphics::ctx()->viewport_inside_wir(fan::graphics::ctx(), nr, position);
  }

  bool inside(const fan::graphics::render_view_t& rv, const fan::vec2& p) {
    fan::vec2 tp = fan::graphics::screen_to_world(p, rv.viewport, rv.camera);
    f32_t z = fan::graphics::camera_get_zoom(rv.camera);
    auto c = fan::graphics::camera_get(rv.camera);
    fan::vec2 cp = fan::graphics::camera_get_position(rv.camera);

    f32_t l = cp.x + c.coordinates.left / z;
    f32_t r = cp.x + c.coordinates.right / z;
    f32_t t = cp.y + c.coordinates.top / z;
    f32_t b = cp.y + c.coordinates.bottom / z;

    return tp.x >= l && tp.x <= r && tp.y >= t && tp.y <= b;
  }

  bool is_mouse_inside(const fan::graphics::render_view_t& render_view) {
    return inside(render_view, get_mouse_position());
  }

#if defined(FAN_2D)

  light_t::light_t(light_properties_t p) {
    *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
      fan_init_struct(
        typename fan::graphics::shapes::light_t::properties_t,
        .camera = p.render_view->camera,
        .viewport = p.render_view->viewport,
        .position = p.position,
        .parallax_factor = p.parallax_factor,
        .size = p.size,
        .rotation_point = p.rotation_point,
        .color = p.color,
        .flags = p.flags,
        .angle = p.angle
      ), p.enable_culling
    );
  }

  light_t::light_t(const fan::vec3& position, const fan::vec2& size, const fan::color& color, render_view_t* render_view)
    : light_t(light_properties_t {
    .render_view = render_view,
    .position = position,
    .size = size,
    .color = color
      }) {}

line_t::line_t(line_properties_t p) {
  *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
    fan_init_struct(
      typename fan::graphics::shapes::line_t::properties_t,
      .camera = p.render_view->camera,
      .viewport = p.render_view->viewport,
      .src = p.src,
      .dst = p.dst,
      .color = p.color,
      .thickness = p.thickness,
      .blending = p.blending,
      .draw_mode = p.draw_mode
    ), p.enable_culling
  );
}

line_t::line_t(const fan::vec3& src, const fan::vec3& dst, const fan::color& color, f32_t thickness, render_view_t* render_view)
  : line_t(line_properties_t {
  .render_view = render_view,
  .src = src,
  .dst = dst,
  .color = color,
  .thickness = thickness
    }) {}

  rectangle_t::rectangle_t(rectangle_properties_t p) {
    *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
      fan_init_struct(
        typename fan::graphics::shapes::rectangle_t::properties_t,
        .camera = p.render_view->camera,
        .viewport = p.render_view->viewport,
        .position = p.position,
        .size = p.size,
        .color = p.color,
        .outline_color = p.outline_color,
        .angle = p.angle,
        .rotation_point = p.rotation_point,
        .blending = p.blending
      ), p.enable_culling
    );
  }

  rectangle_t::rectangle_t(const fan::vec3& position, const fan::vec2& size, const fan::color& color, render_view_t* render_view)
    : rectangle_t(rectangle_properties_t {
    .render_view = render_view,
    .position = position,
    .size = size,
    .color = color
      }) {}

  sprite_t::sprite_t(sprite_properties_t p) {
    *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
      fan_init_struct(
        typename fan::graphics::shapes::sprite_t::properties_t,
        .camera = p.render_view->camera,
        .viewport = p.render_view->viewport,
        .parallax_factor = p.parallax_factor,
        .position = p.position,
        .size = p.size,
        .angle = p.angle,
        .image = p.image,
        .tc_position = p.tc_position,
        .tc_size = p.tc_size,
        .images = p.images,
        .color = p.color,
        .rotation_point = p.rotation_point,
        .blending = p.blending,
        .flags = p.flags,
        .texture_pack_unique_id = p.texture_pack_unique_id
      ), p.enable_culling
    );
  }

  sprite_t::sprite_t(const fan::vec3& position, const fan::vec2& size, const fan::graphics::image_t& image, render_view_t* render_view)
    : sprite_t(sprite_properties_t {
    .render_view = render_view,
    .position = position,
    .size = size,
    .image = image
      }) {}

  unlit_sprite_t::unlit_sprite_t(unlit_sprite_properties_t p) {
    *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
      fan_init_struct(
        typename fan::graphics::shapes::unlit_sprite_t::properties_t,
        .camera = p.render_view->camera,
        .viewport = p.render_view->viewport,
        .position = p.position,
        .size = p.size,
        .angle = p.angle,
        .image = p.image,
        .images = p.images,
        .color = p.color,
        .tc_position = p.tc_position,
        .tc_size = p.tc_size,
        .rotation_point = p.rotation_point,
        .blending = p.blending
      ), p.enable_culling
    );
  }

  unlit_sprite_t::unlit_sprite_t(const fan::vec3& position, const fan::vec2& size, const fan::graphics::image_t& image, render_view_t* render_view)
    : unlit_sprite_t(unlit_sprite_properties_t {
    .render_view = render_view,
    .position = position,
    .size = size,
    .image = image
      }) {}

  circle_t::circle_t(circle_properties_t p) {
    *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
      fan_init_struct(
        typename fan::graphics::shapes::circle_t::properties_t,
        .camera = p.render_view->camera,
        .viewport = p.render_view->viewport,
        .position = p.position,
        .radius = p.radius,
        .angle = p.angle,
        .color = p.color,
        .blending = p.blending,
        .flags = p.flags
      ),
      p.enable_culling
    );
  }
  circle_t::circle_t(const fan::vec3& position, f32_t radius, const fan::color& color, render_view_t* render_view)
    : circle_t(circle_properties_t {
    .render_view = render_view,
    .position = position,
    .radius = radius,
    .color = color
      }) {}

  capsule_t::capsule_t(capsule_properties_t p) {
    *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
      fan_init_struct(
        typename fan::graphics::shapes::capsule_t::properties_t,
        .camera = p.render_view->camera,
        .viewport = p.render_view->viewport,
        .position = p.position,
        .center0 = p.center0,
        .center1 = p.center1,
        .radius = p.radius,
        .angle = p.angle,
        .color = p.color,
        .outline_color = p.outline_color,
        .blending = p.blending,
        .flags = p.flags
      ), p.enable_culling
    );
  }

  polygon_t::polygon_t(polygon_properties_t p) {
    *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
      fan_init_struct(
        typename fan::graphics::shapes::polygon_t::properties_t,
        .camera = p.render_view->camera,
        .viewport = p.render_view->viewport,
        .position = p.position,
        .vertices = p.vertices,
        .angle = p.angle,
        .rotation_point = p.rotation_point,
        .blending = p.blending,
        .draw_mode = p.draw_mode,
        .vertex_count = p.vertex_count
      ), p.enable_culling);
  }

  grid_t::grid_t(grid_properties_t p) {
    *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
      fan_init_struct(
        typename fan::graphics::shapes::grid_t::properties_t,
        .camera = p.render_view->camera,
        .viewport = p.render_view->viewport,
        .position = p.position,
        .size = p.size,
        .grid_size = p.grid_size,
        .rotation_point = p.rotation_point,
        .color = p.color,
        .angle = p.angle
      ), p.enable_culling);
  }

  universal_image_renderer_t::universal_image_renderer_t(const universal_image_renderer_properties_t& p) {
    *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
      fan_init_struct(
        typename fan::graphics::shapes::universal_image_renderer_t::properties_t,
        .camera = p.render_view->camera,
        .viewport = p.render_view->viewport,
        .position = p.position,
        .size = p.size,
        .tc_position = p.tc_position,
        .tc_size = p.tc_size,
        .blending = p.blending,
        .images = p.images,
        .draw_mode = p.draw_mode,
        ), p.enable_culling
    );
  }

  gradient_t::gradient_t(const gradient_properties_t& p) {
    *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
      fan_init_struct(
        typename fan::graphics::shapes::gradient_t::properties_t,
        .camera = p.render_view->camera,
        .viewport = p.render_view->viewport,
        .position = p.position,
        .size = p.size,
        .color = p.color,
        .blending = p.blending,
        .angle = p.angle,
        .rotation_point = p.rotation_point
      ), p.enable_culling
    );
  }

  shader_shape_t::shader_shape_t(const shader_shape_properties_t& p) {
    *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
      fan_init_struct(
        typename fan::graphics::shapes::shader_shape_t::properties_t,
        .camera = p.render_view->camera,
        .viewport = p.render_view->viewport,
        .position = p.position,
        .size = p.size,
        .rotation_point = p.rotation_point,
        .color = p.color,
        .angle = p.angle,
        .flags = p.flags,
        .tc_position = p.tc_position,
        .tc_size = p.tc_size,
        .shader = p.shader,
        .blending = p.blending,
        .image = p.image,
        .images = p.images,
        .draw_mode = p.draw_mode
      ), p.enable_culling
    );
  }

  shadow_t::shadow_t(shadow_properties_t p) {
    *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
      fan_init_struct(
        typename fan::graphics::shapes::shadow_t::properties_t,
        .camera = p.render_view->camera,
        .viewport = p.render_view->viewport,
        .position = p.position,
        .shape = p.shape,
        .size = p.size,
        .rotation_point = p.rotation_point,
        .color = p.color,
        .flags = p.flags,
        .angle = p.angle,
        .light_position = p.light_position,
        .light_radius = p.light_radius
      ), p.enable_culling);
  }

  fan::graphics::shapes::shape_t& add_shape_to_immediate_draw(fan::graphics::shapes::shape_t&& s) {
    fan::graphics::get_shapes().immediate_render_list->emplace_back(std::move(s));
    return fan::graphics::get_shapes().immediate_render_list->back();
  }

  uint32_t add_shape_to_static_draw(fan::graphics::shapes::shape_t&& s) {
    auto ret = s.NRI;
    (*fan::graphics::get_shapes().static_render_list)[ret] = std::move(s);
    return ret;
  }

  void remove_static_shape_draw(const fan::graphics::shapes::shape_t& s) {
    fan::graphics::get_shapes().static_render_list->erase(s.NRI);
  }

  fan::graphics::shapes::shape_t& rectangle(const rectangle_properties_t& props) {
    return add_shape_to_immediate_draw(rectangle_t(props));
  }

  fan::graphics::shapes::shape_t& rectangle(const fan::vec3& position, const fan::vec2& size, const fan::color& color, render_view_t* render_view) {
    return add_shape_to_immediate_draw(rectangle_t(rectangle_properties_t {
      .render_view = render_view,
      .position = position,
      .size = size,
      .color = color
      }));
  }

  fan::graphics::shapes::shape_t& sprite(const sprite_properties_t& props) {
    return add_shape_to_immediate_draw(sprite_t(props));
  }

  fan::graphics::shapes::shape_t& unlit_sprite(const unlit_sprite_properties_t& props) {
    return add_shape_to_immediate_draw(unlit_sprite_t(props));
  }

  fan::graphics::shapes::shape_t& line(const line_properties_t& props) {
    return add_shape_to_immediate_draw(line_t(props));
  }

  fan::graphics::shapes::shape_t& line(const fan::vec3& src, const fan::vec3& dst, const fan::color& color, f32_t thickness, render_view_t* render_view) {
    return add_shape_to_immediate_draw(line_t(line_properties_t {
      .render_view = render_view,
      .src = src,
      .dst = dst,
      .color = color,
      .thickness = thickness
      }));
  }

  fan::graphics::shapes::shape_t& light(const light_properties_t& props) {
    return add_shape_to_immediate_draw(light_t(props));
  }

  fan::graphics::shapes::shape_t& circle(const circle_properties_t& props) {
    return add_shape_to_immediate_draw(circle_t(props));
  }

  fan::graphics::shapes::shape_t& circle(const fan::vec3& position, f32_t radius, const fan::color& color, render_view_t* render_view) {
    return add_shape_to_immediate_draw(circle_t(circle_properties_t {
      .render_view = render_view,
      .position = position,
      .radius = radius,
      .color = color
      }));
  }

  fan::graphics::shapes::shape_t& capsule(const capsule_properties_t& props) {
    return add_shape_to_immediate_draw(capsule_t(props));
  }

  fan::graphics::shapes::shape_t& polygon(const polygon_properties_t& props) {
    return add_shape_to_immediate_draw(polygon_t(props));
  }

  fan::graphics::shapes::shape_t& grid(const grid_properties_t& props) {
    return add_shape_to_immediate_draw(grid_t(props));
  }

#if defined(FAN_PHYSICS_2D)
  void aabb(const fan::physics::aabb_t& b, f32_t depth, const fan::color& c, f32_t thickness, render_view_t* render_view) {
    fan::graphics::line({.render_view = render_view, .src = {b.min, depth}, .dst = {b.max.x, b.min.y}, .color = c, .thickness=thickness});
    fan::graphics::line({.render_view = render_view, .src = {b.max.x, b.min.y, depth}, .dst = {b.max}, .color = c, .thickness=thickness});
    fan::graphics::line({.render_view = render_view, .src = {b.max, depth}, .dst = {b.min.x, b.max.y}, .color = c, .thickness=thickness});
    fan::graphics::line({.render_view = render_view, .src = {b.min.x, b.max.y, depth}, .dst = {b.min}, .color = c, .thickness=thickness});
  }


  void aabb(const fan::graphics::shapes::shape_t& s, f32_t depth, const fan::color& c, f32_t thickness, render_view_t* render_view) {
    fan::graphics::aabb(s.get_aabb(), depth, c, thickness, render_view);
  }

  void aabb(const fan::vec2& min, const fan::vec2& max, f32_t depth, const fan::color& c, f32_t thickness, render_view_t* render_view ) {
    fan::physics::aabb_t aabb_;
    aabb_.min = min;
    aabb_.max = max;
    aabb(aabb_, depth, c, thickness, render_view);
  }
  void aabb(const fan::vec2& min, const fan::vec2& max, f32_t thickness, render_view_t* render_view) {
    aabb(min, max, 0xFFF0, fan::colors::white, thickness, render_view);
  }
#endif

#if defined(FAN_JSON)

  struct json_cache_t {
    fan::graphics::shape_t shape;
    fan::vec2 original_pos;
  };
  std::unordered_map<std::string, json_cache_t> json_cache;

  fan::graphics::shape_t shape_from_json(const std::string& json_path, const std::source_location& callers_path) {
    if (json_cache.find(json_path) == json_cache.end()) {
      fan::json json_data = fan::graphics::read_json(json_path, callers_path);
      resolve_json_image_paths(json_data, json_path, callers_path);
      fan::graphics::parse_animations(json_path, json_data, callers_path);
      fan::graphics::shape_t shape = fan::graphics::extract_single_shape(json_data, callers_path);
      auto& cache = json_cache[json_path];
      cache.shape = shape;
      cache.original_pos = shape.get_position();
      cache.shape.set_position(fan::vec2(-0xfffff));
      return shape;
    }
    auto& cache = json_cache[json_path];
    fan::graphics::shape_t shape = cache.shape;
    shape.set_position(cache.original_pos);
    return shape;
  }

  void resolve_json_image_paths(fan::json& out, const std::string& json_path, const std::source_location& callers_path) {
    out.find_and_iterate("image_path", [&json_path, &callers_path](fan::json& value) {
      std::filesystem::path base = fan::io::file::find_relative_path(json_path, callers_path);
      base = std::filesystem::is_directory(base) ? base : base.parent_path();
      value = (base / std::filesystem::path(value.get<std::string>())).generic_string();
    });
  }
  
  fan::graphics::sprite_t sprite_sheet_from_json(
    const sprite_sheet_config_t config,
    const std::source_location& callers_path) 
  {
    auto shape = shape_from_json(config.path, callers_path);
    shape.set_animation_loop(shape.get_current_animation_id(), config.loop);
    if (config.start) {
      shape.play_sprite_sheet();
    }
    return shape;
  }
#endif

  fan::graphics::shapes::polygon_t::properties_t create_hexagon(f32_t radius, const fan::color& color) {
    fan::graphics::shapes::polygon_t::properties_t pp;
    for (int i = 0; i < 6; ++i) {
      pp.vertices.push_back(fan::graphics::vertex_t {fan::vec3(0, 0, 0), color});
      f32_t angle1 = 2 * fan::math::pi * i / 6;
      f32_t x1 = radius * std::cos(angle1);
      f32_t y1 = radius * std::sin(angle1);
      pp.vertices.push_back(fan::graphics::vertex_t {fan::vec3(fan::vec2(x1, y1), 0), color});
      f32_t angle2 = 2 * fan::math::pi * ((i + 1) % 6) / 6;
      f32_t x2 = radius * std::cos(angle2);
      f32_t y2 = radius * std::sin(angle2);
      pp.vertices.push_back(fan::graphics::vertex_t {fan::vec3(fan::vec2(x2, y2), 0), color});
    }
    return pp;
  }

  fan::line3 get_highlight_positions(const fan::vec3& op_, const fan::vec2& os, int index) {
    fan::line3 positions;
    fan::vec2 op = op_;
    switch (index) {
    case 0:
      positions[0] = fan::vec3(op - os, op_.z + 1);
      positions[1] = fan::vec3(op + fan::vec2(os.x, -os.y), op_.z + 1);
      break;
    case 1:
      positions[0] = fan::vec3(op + fan::vec2(os.x, -os.y), op_.z + 1);
      positions[1] = fan::vec3(op + os, op_.z + 1);
      break;
    case 2:
      positions[0] = fan::vec3(op + os, op_.z + 1);
      positions[1] = fan::vec3(op + fan::vec2(-os.x, os.y), op_.z + 1);
      break;
    case 3:
      positions[0] = fan::vec3(op + fan::vec2(-os.x, os.y), op_.z + 1);
      positions[1] = fan::vec3(op - os, op_.z + 1);
      break;
    default:
      // Handle invalid index if necessary
      break;
    }

    return positions;
  }

#endif

  interactive_camera_t::operator render_view_t* () {
    return &render_view;
  }

  void interactive_camera_t::reset() {
    ignore_input = false;
    zoom_on_window_resize = true;
    pan_with_middle_mouse = false;
    reset_view();
  }

  void interactive_camera_t::reset_view() {
    set_position(get_initial_position());
    update();
  }

  void interactive_camera_t::update() {
    fan::vec2 s = fan::graphics::ctx()->viewport_get_size(
      fan::graphics::ctx(),
      render_view.viewport
    );;
    fan::graphics::ctx()->camera_set_ortho(
      fan::graphics::ctx(),
      render_view.camera,
      fan::vec2(-s.x / 2.f, s.x / 2.f),
      fan::vec2(-s.y / 2.f, s.y / 2.f)
    );
  }

  void interactive_camera_t::create(
    fan::graphics::camera_t camera_nr,
    fan::graphics::viewport_t viewport_nr,
    f32_t new_zoom,
    const fan::vec2& initial_pos
  ) {
    render_view.camera = camera_nr;
    render_view.viewport = viewport_nr;
    set_zoom(new_zoom);
    auto& window = fan::graphics::get_window();
    old_window_size = window.get_size();
    if (initial_pos == -0xFAFA) {
      set_initial_position(viewport_get_size() / 2.f);
    }
    else {
      set_initial_position(initial_pos);
    }
    static auto update_ortho = [this](void* ptr) {
      update();
    };

    update();

    auto it = fan::graphics::ctx().update_callback->NewNodeLast();
    (*fan::graphics::ctx().update_callback)[it] = update_ortho;

    resize_callback_nr = window.add_resize_callback([&](const auto& d) {
      if (old_window_size.x > 0 && old_window_size.y > 0) {
        fan::graphics::viewport_set(render_view.viewport, fan::vec2(0, 0), d.size);
        fan::vec2 ratio = fan::vec2(d.size) / old_window_size;
        f32_t size_ratio = (ratio.y + ratio.x) / 2.0f;
        f32_t zoom_change = get_zoom() * (size_ratio - 1.0f);
        set_zoom(get_zoom() + zoom_change);
      }
      old_window_size = d.size;
    });

    button_cb_nr = window.add_buttons_callback([&](const auto& d) {
      if (ignore_input) {
        return;
      }

    #if defined(FAN_GUI)
      if (fan::graphics::gui::want_io()) {
        return;
      }
    #endif

      bool mouse_inside_viewport = fan::graphics::inside(
        render_view.viewport,
        fan::window::get_mouse_position()
      );
      if (mouse_inside_viewport) {
      #if defined(FAN_GUI)
        auto* context = fan::graphics::gui::get_context();
        auto* hovered_window = context->HoveredWindow;
        if (hovered_window) {
          fan::graphics::gui::set_window_focus(hovered_window->Name);
        }
      #endif
        if (d.button == fan::mouse_scroll_up) {
          set_zoom(get_zoom() * 1.2f);
        }
        else if (d.button == fan::mouse_scroll_down) {
          set_zoom(get_zoom() / 1.2f);
        }
      }
      auto state = d.window->key_state(fan::mouse_middle);
      if (state == (int)fan::mouse_state::press) {
        clicked_inside_viewport = mouse_inside_viewport;
      }
      else if (state == (int)fan::mouse_state::release) {
        clicked_inside_viewport = false;
      }
    });

    mouse_motion_nr = window.add_mouse_motion_callback([&](const auto& d) {
      auto state = d.window->key_state(fan::mouse_middle);
      if (state == (int)fan::mouse_state::press ||
        state == (int)fan::mouse_state::repeat
        ) {
        if (pan_with_middle_mouse && clicked_inside_viewport) {
          fan::vec2 viewport_size = fan::graphics::viewport_get_size(render_view.viewport);
          camera_offset -= (d.motion * viewport_size / (viewport_size * get_zoom()));
          fan::graphics::camera_set_position(render_view.camera, camera_offset);
        }
      }
    });
  }

  void interactive_camera_t::create(const fan::graphics::render_view_t& render_view, f32_t new_zoom) {
    create(render_view.camera, render_view.viewport, new_zoom);
  }

  void interactive_camera_t::create_default(f32_t zoom) {
    render_view.create_default(zoom); // create own render view, leaks...
    create(render_view.camera, render_view.viewport, zoom);
  }

  interactive_camera_t::interactive_camera_t(f32_t zoom) {
    create_default(zoom);
  }

  interactive_camera_t::interactive_camera_t(
    fan::graphics::camera_t camera_nr,
    fan::graphics::viewport_t viewport_nr,
    f32_t new_zoom,
    const fan::vec2& initial_pos
  ) {
    create(camera_nr, viewport_nr, new_zoom, initial_pos);
    initial_position = fan::graphics::viewport_get_size(viewport_nr) / 2.f;
    set_position(initial_position);
  }

  interactive_camera_t::interactive_camera_t(const fan::graphics::render_view_t& render_view, f32_t new_zoom) :
    interactive_camera_t(render_view.camera, render_view.viewport, new_zoom) {
    initial_position = fan::graphics::viewport_get_size(render_view.viewport) / 2.f;
    set_position(initial_position);
  }

  interactive_camera_t::~interactive_camera_t() {
    if (uc_nr.iic() == false) {
      fan::graphics::ctx().update_callback->unlrec(uc_nr);
      uc_nr.sic();
    }
  }

  fan::vec2 interactive_camera_t::get_initial_position() const {
    return initial_position;
  }

  void interactive_camera_t::set_initial_position(const fan::vec2& position) {
    initial_position = position;
  }

  fan::vec2 interactive_camera_t::get_position() const {
    return camera_offset;
  }

  void interactive_camera_t::set_position(const fan::vec2& position) {
    camera_offset = position;
    fan::graphics::camera_set_position(render_view.camera, camera_offset);
    update();
  }

  f32_t interactive_camera_t::get_zoom() const {
    return fan::graphics::camera_get_zoom(render_view.camera);
  }

  void interactive_camera_t::set_zoom(f32_t new_zoom) {
    fan::graphics::camera_set_zoom(render_view.camera, new_zoom);
    update();
  }

  fan::vec2 interactive_camera_t::get_size() const {
    return fan::graphics::ctx()->camera_get_size(
      fan::graphics::ctx(),
      render_view.camera);
  }

  fan::vec4 interactive_camera_t::get_ortho() const {
    return fan::graphics::camera_get(render_view.camera).coordinates.v;
  }

  fan::vec2 interactive_camera_t::get_viewport_size() const {
    return fan::graphics::viewport_get_size(render_view.viewport);
  }

  world_window_t::world_window_t() : render_view(true), cam(render_view) {}
  void world_window_t::update(const fan::vec2& viewport_pos, const fan::vec2& viewport_size) {
    fan::graphics::viewport_set(
      render_view.viewport,
      viewport_pos,
      viewport_size
    );

    fan::graphics::camera_set_ortho(
      render_view.camera,
      fan::vec2(-viewport_size.x / 2, viewport_size.x / 2),
      fan::vec2(-viewport_size.y / 2, viewport_size.y / 2)
    );
  }
  world_window_t::operator render_view_t* () {
    return &render_view;
  }

  image_divider_t::image_divider_t() {
    e.open(open_properties);
    texture_properties.visual_output = fan::graphics::image_sampler_address_mode::clamp_to_edge;
    texture_properties.min_filter = fan::graphics::image_filter::nearest;
    texture_properties.mag_filter = fan::graphics::image_filter::nearest;
  }

#if defined(FAN_2D)

  fan::graphics::shapes::polygon_t::properties_t create_sine_ground(const fan::vec2& position, f32_t amplitude, f32_t frequency, f32_t width, f32_t groundWidth) {
    fan::graphics::shapes::polygon_t::properties_t pp;
    // for triangle strip
    for (f32_t x = 0; x < groundWidth - width; x += width) {
      f32_t y1 = position.y / 2 + amplitude * std::sin(frequency * x);
      f32_t y2 = position.y / 2 + amplitude * std::sin(frequency * (x + width));

      // top
      pp.vertices.push_back({fan::vec2(position.x + x, y1), fan::colors::red});
      // bottom
      pp.vertices.push_back({fan::vec2(position.x + x, position.y), fan::colors::white});
      // next top
      pp.vertices.push_back({fan::vec2(position.x + x + width, y2), fan::colors::red});
      // next bottom
      pp.vertices.push_back({fan::vec2(position.x + x + width, position.y), fan::colors::white});
    }

    return pp;
  }

  std::vector<fan::vec2> ground_points(const fan::vec2& position, f32_t amplitude, f32_t frequency, f32_t width, f32_t groundWidth) {
    std::vector<fan::vec2> outline_points;
    for (f32_t x = 0; x <= groundWidth; x += width) {
      f32_t y = position.y / 2 + amplitude * std::sin(frequency * x);
      outline_points.push_back(fan::vec2(position.x + x, y));
    }
    outline_points.push_back(fan::vec2(position.x + groundWidth, position.y));
    outline_points.push_back(fan::vec2(position.x, position.y));
    return outline_points;
  }

  trail_t::trail_t() {
    update_callback_nr = fan::graphics::ctx().update_callback->NewNodeLast();
    (*fan::graphics::ctx().update_callback)[update_callback_nr] = [this](void* ptr) {
      update();
    };
  }

  trail_t::~trail_t() {
    if (!update_callback_nr) {
      return;
    }
    fan::graphics::ctx().update_callback->unlrec(update_callback_nr);
    update_callback_nr.sic();
  }

  void trail_t::set_point(const fan::vec3& point, f32_t drift_intensity) {
    static fan::time::timer timer {(double)300000000ULL, true};
    bool should_reset = trails.empty() || timer;

    if (should_reset) {
      trails.resize(trails.size() + 1);
      trails.back().vertices.clear();
      trails.back().creation_time = fan::time::now();
      trails.back().base_alpha = 0.2f + (drift_intensity * 0.6f);
    }

    bool start_new_trail = false;
    if (!trails.empty() && !trails.back().vertices.empty()) {
      fan::vec3 last_point = trails.back().vertices.back().position;
      f32_t distance = std::sqrt(std::pow(point.x - last_point.x, 2) + std::pow(point.y - last_point.y, 2));
      if (distance > 50.0f) {
        start_new_trail = true;
      }
    }

    if (start_new_trail) {
      trails.resize(trails.size() + 1);
      trails.back().vertices.clear();
      trails.back().creation_time = fan::time::now();
      trails.back().base_alpha = 0.2f + (drift_intensity * 0.6f);
    }

    fan::vec2 direction = fan::vec2(1, 0);
    if (trails.back().vertices.size() >= 2) {
      fan::vec3 last_point = trails.back().vertices[trails.back().vertices.size() - 2].position;
      fan::vec3 diff = point - last_point;
      direction = fan::vec2(diff.x, diff.y);
      f32_t len = std::sqrt(direction.x * direction.x + direction.y * direction.y);
      if (len > 0) {
        direction.x /= len;
        direction.y /= len;
      }
    }

    fan::vec2 perp = fan::vec2(-direction.y, direction.x);
    fan::graphics::vertex_t vertex;
    vertex.color = fan::color(color.r, color.g, color.b, trails.back().base_alpha);

    vertex.position = point + fan::vec3(perp.x * thickness * 0.5f, perp.y * thickness * 0.5f, 0);
    trails.back().vertices.emplace_back(vertex);

    vertex.position = point - fan::vec3(perp.x * thickness * 0.5f, perp.y * thickness * 0.5f, 0);
    trails.back().vertices.emplace_back(vertex);

    trails.back().polygon = fan::graphics::polygon_t {{
        .position = fan::vec3(0, 0, point.z),
        .vertices = trails.back().vertices,
        .draw_mode = fan::graphics::primitive_topology_t::triangle_strip,
      }};

    timer.restart();
  }

  void trail_t::update() {
    uint64_t current_time = fan::time::now();

    for (auto& trail : trails) {
      uint64_t age = current_time - trail.creation_time;
      f32_t fade_factor = 1.0f;

      if (age > fade_duration) {
        fade_factor = std::max(0.0f, 1.0f - static_cast<f32_t>(age - fade_duration) / static_cast<f32_t>(max_trail_lifetime - fade_duration));
      }
      f32_t current_alpha = trail.base_alpha * fade_factor;

      for (size_t i = 0; i < trail.vertices.size(); i += 2) {
        f32_t position_factor = static_cast<f32_t>(i) / static_cast<f32_t>(trail.vertices.size() - 2);
        f32_t vertex_alpha = current_alpha * (0.2f + 0.8f * position_factor);

        trail.vertices[i].color.a = vertex_alpha;     // left vertex
        trail.vertices[i + 1].color.a = vertex_alpha;   // right vertex
      }
      fan::vec3 pos = trail.polygon.get_position();
      trail.polygon = {{
          .position = pos,
          .vertices = trail.vertices,
          .draw_mode = fan::graphics::primitive_topology_t::triangle_strip,
        }};
    }

    trails.erase(
      std::remove_if(trails.begin(), trails.end(), [&](const trail_segment_t& trail) {
      uint64_t age = current_time - trail.creation_time;
      return age > max_trail_lifetime;
    }),
      trails.end()
    );
  }

  f32_t get_depth_from_y(const fan::vec2& position, f32_t tile_size_y) {
    return std::floor((position.y) / tile_size_y) + (0xFAAA - 2) / 2 + 18.f;
  }

  tilemap_t::tilemap_t(const fan::vec2& tile_size,
    const fan::color& color,
    const fan::vec2& area,
    const fan::vec2& offset,
    render_view_t* render_view) {
    create(tile_size, color, area, offset, render_view);
  }

  fan::vec2i tilemap_t::get_cell_count() {
    return fan::vec2i(
      shapes.empty() ? 0 : (int)shapes[0].size(),
      (int)shapes.size()
    );
  }

  template <typename T>
  fan::graphics::shapes::shape_t& tilemap_t::operator[](const fan::vec2_wrap_t<T>& v) {
    return shapes[v.y][v.x];
  }

  void tilemap_t::add_wall(const fan::vec2i& cell, fan::graphics::algorithm::pathfind::generator& gen) {
    if (wall_cells.find(cell) == wall_cells.end()) {
      gen.add_collision(cell);
      wall_cells.insert(cell);
      shapes[cell.y][cell.x].set_color(fan::colors::gray / 3);
    }
  }

  void tilemap_t::remove_wall(const fan::vec2i& cell, fan::graphics::algorithm::pathfind::generator& gen) {
    if (wall_cells.find(cell) != wall_cells.end()) {
      gen.remove_collision(cell);
      wall_cells.erase(cell);
      shapes[cell.y][cell.x].set_color(fan::colors::gray);
    }
  }

  void tilemap_t::reset_colors(const fan::color& color) {
    for (int i = 0; i < size.y; i++) {
      for (int j = 0; j < size.x; j++) {
        fan::vec2i cell(j, i);
        if (wall_cells.contains(cell)) {
          continue;
        }
        shapes[i][j].set_color(color);
      }
    }
  }

  void tilemap_t::set_source(const fan::vec2i& cell, const fan::color& color) {
    if (!wall_cells.contains(cell) &&
      cell.x >= 0 && cell.x < shapes[0].size() &&
      cell.y >= 0 && cell.y < shapes.size()) {
      shapes[cell.y][cell.x].set_color(color);
    }
  }

  void tilemap_t::set_destination(const fan::vec2i& cell, const fan::color& color) {
    if (!wall_cells.contains(cell) &&
      cell.x >= 0 && cell.x < shapes[0].size() &&
      cell.y >= 0 && cell.y < shapes.size()) {
      shapes[cell.y][cell.x].set_color(color);
    }
  }

  void tilemap_t::highlight_path(
    const fan::graphics::algorithm::pathfind::coordinate_list& path,
    const fan::color& color
  ) {
    for (const auto& p : path) {
      if (!wall_cells.contains(p) &&
        p.x >= 0 && p.x < shapes[0].size() &&
        p.y >= 0 && p.y < shapes.size()) {
        shapes[p.y][p.x].set_color(color);
      }
    }
  }

  fan::graphics::algorithm::pathfind::coordinate_list tilemap_t::find_path(
    const fan::vec2i& src,
    const fan::vec2i& dst,
    fan::graphics::algorithm::pathfind::generator& gen,
    fan::graphics::algorithm::pathfind::heuristic_function heuristic,
    bool diagonal
  ) {
    gen.set_heuristic(heuristic);
    gen.set_diagonal_movement(diagonal);
    return gen.find_path(src, dst);
  }

  void tilemap_t::create(
    const fan::vec2& tile_size,
    const fan::color& color,
    const fan::vec2& area,
    const fan::vec2& offset,
    render_view_t* render_view
  ) {
    fan::vec2 map_size(
      std::floor(area.x / tile_size.x),
      std::floor(area.y / tile_size.y)
    );

    size = map_size;
    this->tile_size = tile_size;
    positions.resize(map_size.y, std::vector<fan::vec2>(map_size.x));
    shapes.resize(map_size.y, std::vector<fan::graphics::shape_t>(map_size.x));
    for (int i = 0; i < map_size.y; i++) {
      for (int j = 0; j < map_size.x; j++) {
        positions[i][j] = offset + tile_size / 2 + fan::vec2(j * tile_size.x, i * tile_size.y);

        shapes[i][j] = fan::graphics::rectangle_t {{
          .render_view = render_view,
          .position = fan::vec3(positions[i][j], 0),
          .size = tile_size / 2,
          .color = color
        }};
      }
    }
  }

  void tilemap_t::set_tile_color(const fan::vec2i& pos, const fan::color& c) {
    if (pos.x < 0 || pos.x >= size.x) return;
    if (pos.y < 0 || pos.y >= size.y) return;
    shapes[pos.y][pos.x].set_color(c);
  }

  constexpr f32_t tilemap_t::circle_overlap(f32_t r, f32_t i0, f32_t i1) {
    if (i0 <= 0 && i1 >= 0) return r;
    f32_t y = fan::math::min(fan::math::min(std::fabs(i0), std::fabs(i1)) / r, 1.f);
    return std::sqrt(1.f - y * y) * r;
  }

  void tilemap_t::highlight_circle(const fan::graphics::shapes::shape_t& circle,
    const fan::color& highlight_color
  ) {
    fan::vec2 wp = circle.get_position();
    f32_t r = circle.get_radius();
    auto gi = fan::cast<sint32_t>(decltype(wp){});

    constexpr auto recurse = []<uint32_t d>(const auto& self,
      tilemap_t & tilemap,
      auto& gi,
      fan::vec2 wp,
      f32_t r,
      f32_t er,
      const fan::color & hc
      ) {
      f32_t cell_size = tilemap.tile_size[d];

      if constexpr (d + 1 < wp.size()) {
        gi[d] = (wp[d] - er) / cell_size;
        f32_t sp = (f32_t)gi[d] * cell_size;
        while (true) {
          f32_t rp = sp - wp[d];
          f32_t roff = circle_overlap(r, rp, rp + cell_size);
          self.template operator() < d + 1 > (self, tilemap, gi, wp, r, roff, hc);
          gi[d]++;
          sp += cell_size;
          if (sp > wp[d] + er) break;
        }
      }
      else if constexpr (d < wp.size()) {
        gi[d] = (wp[d] - er) / cell_size;
        sint32_t to = (wp[d] + er) / cell_size;
        for (; gi[d] <= to; gi[d]++) {
          fan::vec2i pos {gi[0], gi[1]};
          tilemap.set_tile_color(pos, hc);
        }
      }
    };

    recurse.template operator() < 0 > (recurse, *this, gi, wp, r, r, highlight_color);
  }

  void tilemap_t::highlight_line(const fan::graphics::shapes::shape_t& line,
    const fan::color& color,
    render_view_t* render_view
  ) {
    fan::vec2 src = line.get_src();
    fan::vec2 dst = line.get_dst();

    auto raycast_positions = fan::graphics::algorithm::grid_raycast(
      {src, dst},
      tile_size
    );

    for (auto& pos : raycast_positions) {
      set_tile_color(pos, color);
    }
  }

  void tilemap_t::highlight(const fan::graphics::shapes::shape_t& shape,
    const fan::color& color
  ) {
    using namespace fan::graphics;

    switch (shape.get_shape_type()) {
    case shapes::shape_type_t::circle:
      highlight_circle(shape, color);
      break;
    case shapes::shape_type_t::line:
      highlight_line(shape, color);
      break;
    default:
      fan::throw_error("method not implemented");
      break;
    }
  }

  fan::color terrain_palette_t::get(int value) const {
    if (value <= stops.front().first) return stops.front().second;
    if (value >= stops.back().first) return stops.back().second;

    for (size_t i = 0; i < stops.size() - 1; ++i) {
      if (value >= stops[i].first && value <= stops[i + 1].first) {
        int v1 = stops[i].first;
        int v2 = stops[i + 1].first;
        auto c1 = stops[i].second;
        auto c2 = stops[i + 1].second;
        f32_t factor = (value - v1) / f32_t(v2 - v1);
        return c1.lerp(c2, factor);
      }
    }
    return fan::color::rgb(0, 0, 0);
  }

  static void generate_mesh_cell(
    int i, int j,
    const vec2& noise_size,
    const std::vector<uint8_t>& noise_data,
    const fan::graphics::image_t& texture,
    const terrain_palette_t& palette,
    sprite_properties_t& sp,
    std::vector<fan::graphics::shape_t>& out_mesh
  ) {
    int index = (i * noise_size.x + j) * 3;
    int grayscale = noise_data[index];

    sp.position = fan::vec2(i, j) * sp.size * 2;
    sp.image = texture;
    sp.color = palette.get(grayscale);
    sp.color.a = 1;

    out_mesh[i * noise_size.x + j] = fan::graphics::sprite_t(sp);
  }

  void generate_mesh(
    const vec2& noise_size,
    const std::vector<uint8_t>& noise_data,
    const fan::graphics::image_t& texture,
    std::vector<fan::graphics::shape_t>& out_mesh,
    const terrain_palette_t& palette,
    const sprite_properties_t& cp
  ) {
    sprite_properties_t sp = cp;
    sp.size = fan::graphics::viewport_get_size(sp.render_view->viewport) / noise_size / 2;
    out_mesh.resize(noise_size.multiply());
    for (int i = 0; i < noise_size.y; ++i) {
      for (int j = 0; j < noise_size.x; ++j) {
        generate_mesh_cell(i, j, noise_size, noise_data, texture, palette, sp, out_mesh);
      }
    }
  }
  fan::event::task_t async_generate_mesh(
    const vec2& noise_size, 
    const std::vector<uint8_t>& noise_data, 
    const fan::graphics::image_t& texture, 
    std::vector<fan::graphics::shape_t>& out_mesh, 
    const terrain_palette_t& palette,
    const sprite_properties_t& cp
    ) {
    sprite_properties_t sp = cp;
    sp.size = fan::graphics::viewport_get_size(sp.render_view->viewport) / noise_size / 2;
    out_mesh.resize(noise_size.multiply());
    for (int i = 0; i < noise_size.y; ++i) {
      for (int j = 0; j < noise_size.x; ++j) {
        generate_mesh_cell(i, j, noise_size, noise_data, texture, palette, sp, out_mesh);
      }
      co_await fan::co_sleep(1);
    }
  }
  #endif
}
#if defined(FAN_2D)
namespace fan::image {
  plane_split_t plane_split(void* pixel_data, const fan::vec2ui& size, const fan::graphics::image_format& format) {
    plane_split_t result;
    uint64_t offset = 0;
    if (format == fan::graphics::image_format::yuv420p) {
      result.planes[0] = pixel_data;
      result.planes[1] = (uint8_t*)pixel_data + (offset += size.multiply());
      result.planes[2] = (uint8_t*)pixel_data + (offset += size.multiply() / 4);
    }
    else {
      fan::throw_error_impl("undefined");
    }
    return result;
  }
}

namespace fan::graphics {
  bool tile_world_generator_t::is_solid(int x, int y) {
    if (x < 0 || x >= map_size.x || y < 0 || y >= map_size.y) {
      return true;
    }
    return tiles[x + y * map_size.x];
  }

  int tile_world_generator_t::count_neighbors(int x, int y) {
    int c = 0;
    for (int j = -1; j <= 1; j++) {
      for (int i = -1; i <= 1; i++) {
        if (is_solid(x + i, y + j)) {
          c++;
        }
      }
    }
    return c;
  }

  void tile_world_generator_t::iterate() {
    std::vector<bool> nt(map_size.x * map_size.y);
    for (int y = 0; y < map_size.y; y++) {
      for (int x = 0; x < map_size.x; x++) {
        nt[x + y * map_size.x] = count_neighbors(x, y) >= 5;
      }
    }
    tiles = std::move(nt);
  }

  void tile_world_generator_t::init_tile_world() {
    tiles.resize(map_size.x * map_size.y);
    for (int i = 0; i < map_size.x * map_size.y; i++) {
      tiles[i] = fan::random::value(0.f, 1.0f) < 0.45f;
    }
  }

  void tile_world_generator_t::init() {
    init_tile_world();
  }

  animation_frame_awaiter::animation_frame_awaiter(
    fan::graphics::shapes::shape_t* sprite_,
    const std::string& anim_,
    int frame_
  ) : sprite(sprite_), animation_name(anim_), target_frame(frame_) {}

  bool animation_frame_awaiter::check_condition() const {
    return sprite && sprite->get_current_animation_frame() >= (target_frame - 1) &&
      sprite->get_current_animation().name == animation_name;
  }
}

namespace fan {

  f32_t apply_ease(ease_e easing, f32_t t) {
    switch (easing) {
    case ease_e::linear:
      return t;
    case ease_e::sine:
      return (std::sin((t - 0.5f) * fan::math::pi) + 1.f) * 0.5f;
    case ease_e::pulse:
      return std::sin(t * fan::math::pi);
    case ease_e::ease_in:
      return t * t;
    case ease_e::ease_out:
      return 1.f - (1.f - t) * (1.f - t);
    }
    return t;
  }

  template <typename T>
  auto make_lerp() {
    if constexpr (requires (T a, T b, f32_t u) { a.lerp(b, u); }) {
      return [](const T& a, const T& b, f32_t u) {
        return a.lerp(b, u);
      };
    }
    else {
      return [](const T& a, const T& b, f32_t u) {
        return a + (b - a) * u;
      };
    }
  }

  auto_color_transition_t pulse_red(f32_t duration) {
    auto_color_transition_t t;
    t.from = fan::colors::white;
    t.to = fan::color(1.f, 0.2f, 0.2f);
    t.duration = duration;
    t.loop = true;
    t.easing = ease_e::pulse;
    t.lerp = make_lerp<fan::color>();
    return t;
  }
  color_transition_t fade_out(f32_t duration) {
    color_transition_t t;
    t.from = fan::colors::white;
    t.to = fan::colors::transparent;
    t.duration = duration;
    t.loop = false;
    t.easing = ease_e::ease_out;
    t.lerp = make_lerp<fan::color>();
    return t;
  }
  vec2_transition_t move_linear(const fan::vec2& from, const fan::vec2& to, f32_t duration) {
    vec2_transition_t t;
    t.from = from;
    t.to = to;
    t.duration = duration;
    t.loop = false;
    t.easing = ease_e::linear;
    t.lerp = make_lerp<fan::vec2>();
    return t;
  }
  vec2_transition_t move_pingpong(const fan::vec2& from, const fan::vec2& to, f32_t duration) {
    vec2_transition_t t;
    t.from = from;
    t.to = to;
    t.duration = duration;
    t.loop = true;
    t.easing = ease_e::pulse;
    t.lerp = make_lerp<fan::vec2>();
    return t;
  }

}

#endif