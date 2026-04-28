module;

#include <fan/utility.h>

#include <vector>
#include <string>
#include <source_location>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <filesystem>

#if defined (FAN_OPENGL)
  #include <fan/graphics/opengl/init.h>
#endif

module fan.graphics.common_context;

import fan.print;
import fan.math;

namespace fan::graphics {
  void lighting_t::set_target(const fan::vec3& t, f32_t d) {
    start = ambient;
    target = t;
    duration = d;
    elapsed = 0.0f;
  }
  void lighting_t::update(f32_t delta_time) {
    if (duration == 0) {
      ambient = target;
      return;
    }
    if (elapsed < duration) {
      elapsed += delta_time;
      f32_t t = std::min(elapsed / duration, 1.0f);
      ambient = fan::math::lerp(start, target, t);
    }
  }
  bool lighting_t::is_near(const fan::vec3& t, f32_t eps) const {
    return ambient.distance(t) < eps;
  }
  bool lighting_t::is_near_target(f32_t eps) const {
    return is_near(target, eps);
  }

  void render_context_handle_t::set_context(context_functions_t& ctx, void* context) {
    context_functions = &ctx;
    render_context = context;
  }
  context_functions_t* render_context_handle_t::operator->() {
    return context_functions;
  }
  render_context_handle_t::operator void* () {
    return render_context;
  }
  std::uint8_t render_context_handle_t::get_renderer() {
    return window->renderer;
  }
  fan::window_t& get_window() {
    return *fan::graphics::ctx().window;
  }
  render_context_handle_t& ctx() {
    static render_context_handle_t handle;
    return handle;
  }
  fan::graphics::render_view_t& get_orthographic_render_view() {
    return *ctx().orthographic_render_view;
  }
  fan::graphics::render_view_t& get_perspective_render_view() {
    return *ctx().perspective_render_view;
  }
  fan::graphics::image_data_t& image_get_data(fan::graphics::image_nr_t nr) {
    return (*ctx().image_list)[nr];
  }
  lighting_t& get_lighting() {
    return *ctx().lighting;
  }
#if defined(FAN_GUI)
  gui_draw_cb_t& get_gui_draw_cbs() {
    return *ctx().gui_draw_cbs;
  }
#endif

  fan::graphics::image_t get_default_texture() {
    return ctx().default_texture;
  }

  image_t::image_t(__empty_struct st) : fan::graphics::image_nr_t() {}
  image_t::image_t() : fan::graphics::image_nr_t(ctx().default_texture) {}
  image_t::image_t(fan::graphics::image_nr_t image) : fan::graphics::image_nr_t(image) {}
  image_t::image_t(const fan::color& color)
    : fan::graphics::image_nr_t(ctx()->image_create_color(ctx(), color)) {}
  image_t::image_t(fan::str_view_t path, const std::source_location& callers_path)
    : fan::graphics::image_nr_t(ctx()->image_load_path(ctx(), path, callers_path)) {}
  image_t::image_t(fan::str_view_t path, const fan::graphics::image_load_properties_t lp, const std::source_location& callers_path)
    : fan::graphics::image_nr_t(ctx()->image_load_path_props(ctx(), path, lp, callers_path)) {}
  image_t::image_t(const char* path, const std::source_location& callers_path) : 
      image_t(fan::str_view_t(path), callers_path) { }
  image_t::image_t(const fan::image::info_t& info)
    : fan::graphics::image_nr_t(ctx()->image_load_info(ctx(), info)) {}
  image_t::image_t(const fan::image::info_t& info, const fan::graphics::image_load_properties_t& lp)
    : fan::graphics::image_nr_t(ctx()->image_load_info_props(ctx(), info, lp)) {}
  image_t::image_t(fan::color* colors, const fan::vec2ui& size)
    : fan::graphics::image_nr_t(ctx()->image_load_colors(ctx(), colors, size)) {}
  image_t::image_t(fan::color* colors, const fan::vec2ui& size, const fan::graphics::image_load_properties_t& lp)
    : fan::graphics::image_nr_t(ctx()->image_load_colors_props(ctx(), colors, size, lp)) {}
  image_t::image_t(std::span<const fan::color> colors, const fan::vec2ui& size)
    : fan::graphics::image_nr_t(ctx()->image_load_colors_props(ctx(), const_cast<fan::color*>(colors.data()), size, image_presets::pixel_art())) {}
  image_t::image_t(const fan::vec2& size, uint32_t channels, const image_load_properties_t& lp) {
    std::vector<uint8_t> blank(size.multiply() * channels, 0);
    fan::image::info_t info;
    info.size = size;
    info.channels = channels;
    info.data = blank.data();
    *(fan::graphics::image_nr_t*)this = ctx()->image_load_info_props(ctx(), info, lp);
  }

  // for no gloco access
  image_t image_t::invalid() {
    image_t mg{__empty_struct()};
    return mg;
  }

  fan::vec2 image_t::get_size() const {
    return fan::graphics::image_get_data(*this).size;
  }
  image_load_properties_t image_t::get_load_properties() const {
    return fan::graphics::image_get_data(*this).image_settings;
  }
  std::string image_t::get_path() const {
    return fan::graphics::image_get_data(*this).image_path;
  }
  bool image_t::valid() const {
    return *this != fan::graphics::ctx().default_texture && iic() == false;
  }

  void image_t::reload(const fan::image::info_t& info) {
    ctx()->image_reload_image_info(ctx(), *this, info);
  }
  void image_t::reload(const fan::image::info_t& info, const fan::graphics::image_load_properties_t& lp) {
    ctx()->image_reload_image_info_props(ctx(), *this, info, lp);
  }
  void image_t::reload(const std::string& path, const std::source_location& callers_path) {
    ctx()->image_reload_path(ctx(), *this, path, callers_path);
  }
  void image_t::reload(const std::string& path, const fan::graphics::image_load_properties_t& lp, const std::source_location& callers_path) {
    ctx()->image_reload_path_props(ctx(), *this, path, lp, callers_path);
  }
  void image_t::unload() {
    ctx()->image_unload(ctx(), *this);
    sic();
  }
  void image_t::update(const void* data, uint32_t channels) {
    fan::image::info_t info;
    info.size = get_size();
    info.channels = channels;
    info.data = const_cast<void*>(data);
    reload(info);
  }
  void image_t::update(const std::vector<uint8_t>& data, uint32_t channels) {
    update(data.data(), channels);
  }
  std::vector<uint8_t> image_t::get_pixel_data(int image_format, fan::vec2 uvp, fan::vec2 uvs) const {
    return ctx()->image_get_pixel_data(ctx(), *this, image_format, uvp, uvs);
  }
  std::vector<uint8_t> image_t::read_pixels(const fan::vec2& uv_pos, const fan::vec2& uv_size) const {
    return ctx()->image_read_pixels(ctx(), *this, uv_pos, uv_size);
  }
  void image_t::bind() const {
    ctx()->image_bind(ctx(), *this);
  }
  void image_t::unbind() const {
    ctx()->image_unbind(ctx(), *this);
  }
  uint64_t image_t::get_handle() const {
    return ctx()->image_get_handle(ctx(), *this);
  }
  image_load_properties_t& image_t::get_settings() {
    return ctx()->image_get_settings(ctx(), *this);
  }
  void image_t::set_settings(const fan::graphics::image_load_properties_t& settings) {
    ctx()->image_set_settings(ctx(), *this, settings);
  }

  render_view_t::render_view_t(bool) {
    create_default(fan::window::get_size());
  }

  void render_view_t::create() {
    camera = ctx()->camera_create(ctx());
    viewport = ctx()->viewport_create(ctx());
  }
  void render_view_t::create_default(const fan::vec2& window_size, f32_t zoom) {
    create();
    ctx()->camera_set_ortho(
      ctx(),
      camera,
      fan::vec2(0, window_size.x),
      fan::vec2(0, window_size.y)
    );
    ctx()->camera_set_zoom(ctx(), camera, zoom);
    ctx()->viewport_set_nr(
      ctx(),
      viewport,
      fan::vec2(0, 0),
      window_size,
      window_size
    );
  }
  void render_view_t::remove() {
    ctx()->camera_erase(ctx(), camera);
    ctx()->viewport_erase(ctx(), viewport);
  }
  void render_view_t::set(
    const fan::vec2& ortho_x, const fan::vec2& ortho_y,
    const fan::vec2& viewport_position,
    const fan::vec2& viewport_size,
    const fan::vec2& window_size
  ) {
    ctx()->camera_set_ortho(ctx(), camera, ortho_x, ortho_y);
    ctx()->viewport_set(
      ctx(), viewport_position, viewport_size, window_size
    );
  }

  std::string render_view_t::debug_string() {
    std::string s;

    fan::vec3 cam_pos = ctx()->camera_get_position(ctx(), camera);
    fan::vec2 cam_size = ctx()->camera_get_size(ctx(), camera);
    f32_t cam_zoom = ctx()->camera_get_zoom(ctx(), camera);

    s += "Camera Info:\n";
    s += "  Position: " + cam_pos.to_string() + "\n";
    s += "  Size: " + cam_size.to_string() + "\n";
    s += "  Zoom: " + std::to_string(cam_zoom) + "\n";

    fan::vec2 vp_pos = ctx()->viewport_get_position(ctx(), viewport);
    fan::vec2 vp_size = ctx()->viewport_get_size(ctx(), viewport);

    s += "Viewport Info:\n";
    s += "  Position: " + vp_pos.to_string() + "\n";
    s += "  Size: " + vp_size.to_string() + "\n";

    return s;
  }

  render_view_t::operator fan::graphics::camera_t&() {
    return camera;
  }
  render_view_t::operator fan::graphics::viewport_t&() {
    return viewport;
  }

  fan::vec3 render_view_t::get_camera_position() const {
    return camera_get_position(camera);
  }
  void render_view_t::set_camera_position(fan::vec3 pos) {
    camera_set_position(camera, pos);
  }

  fan::vec2 translate_position(const fan::vec2& p, viewport_t viewport, camera_t camera) {
    auto v = ctx()->viewport_get(ctx(), viewport);
    auto c = ctx()->camera_get(ctx(), camera);
    fan::vec2 viewport_position = v.position;
    fan::vec2 viewport_size = v.size;
    f32_t l = c.coordinates.left / c.zoom;
    f32_t r = c.coordinates.right / c.zoom;
    f32_t t = c.coordinates.top / c.zoom;
    f32_t b = c.coordinates.bottom / c.zoom;
    fan::vec2 tp = p - viewport_position;
    fan::vec2 d = viewport_size;
    tp /= d;
    tp = fan::vec2(r * tp.x - l * tp.x + l, b * tp.y - t * tp.y + t);
    return tp;
  }
  fan::vec2 screen_to_world(const fan::vec2& p, fan::graphics::viewport_t viewport, fan::graphics::camera_t camera) {
    auto v = ctx()->viewport_get(ctx(), viewport);
    auto c = ctx()->camera_get(ctx(), camera);
    fan::vec2 viewport_position = v.position;
    fan::vec2 viewport_size = v.size;
    f32_t l = c.coordinates.left / c.zoom;
    f32_t r = c.coordinates.right / c.zoom;
    f32_t t = c.coordinates.top / c.zoom;
    f32_t b = c.coordinates.bottom / c.zoom;
    fan::vec2 tp = p - viewport_position;
    fan::vec2 d = viewport_size;
    tp /= d;
    tp = fan::vec2(r * tp.x - l * tp.x + l, b * tp.y - t * tp.y + t);
    tp += c.position;
    return tp;
  }
  fan::vec2 screen_to_world(const fan::vec2& p, const render_view_t& render_view) {
    return screen_to_world(p, render_view.viewport, render_view.camera);
  }
  fan::vec2 world_to_screen(const fan::vec2& p, fan::graphics::viewport_t viewport, fan::graphics::camera_t camera) {
    auto v = ctx()->viewport_get(ctx(), viewport);
    auto c = ctx()->camera_get(ctx(), camera);
    fan::vec2 viewport_position = v.position;
    fan::vec2 viewport_size = v.size;
    f32_t l = c.coordinates.left / c.zoom;
    f32_t r = c.coordinates.right / c.zoom;
    f32_t t = c.coordinates.top / c.zoom;
    f32_t b = c.coordinates.bottom / c.zoom;
    fan::vec2 tp = p - c.position;
    f32_t u = (tp.x - l) / (r - l);
    f32_t vcoord = (tp.y - t) / (b - t);
    tp = fan::vec2(u, vcoord) * viewport_size;
    tp += viewport_position;
    return tp;
  }
  fan::vec2 world_to_screen(const fan::vec2& p, const render_view_t& render_view) {
    return world_to_screen(p, render_view.viewport, render_view.camera);
  }
  fan::vec2 get_mouse_position() {
    return fan::graphics::ctx().window->get_mouse_position();
  }
  fan::vec2 get_mouse_position(const camera_t& camera, const viewport_t& viewport) {
    return fan::graphics::screen_to_world(get_mouse_position(), viewport, camera);
  }
  fan::vec2 get_mouse_position(const fan::graphics::render_view_t& render_view) {
    return get_mouse_position(render_view.camera, render_view.viewport);
  }

  fan::vec2 get_mouse_world_pos() {
    return screen_to_world(get_mouse_position());
  }
}
namespace fan::window {
  void add_input_action(const int* keys, std::size_t count, const std::string_view& action_name) {
    for (std::size_t i = 0; i < count; ++i) {
      fan::graphics::ctx().input_action->add(keys[i], action_name);
    }
  }
  void add_input_action(std::initializer_list<int> keys, const std::string_view& action_name) {
    fan::graphics::ctx().input_action->add(keys, action_name);
  }
  void add_input_action(int key, const std::string_view& action_name) {
    fan::graphics::ctx().input_action->add(key, action_name);
  }
  bool is_input_action_active(const std::string_view& action_name, int pstate) {
    return fan::graphics::ctx().input_action->is_active(action_name);
  }
  bool is_action_clicked(const std::string_view& action_name) {
    return fan::graphics::ctx().input_action->is_active(action_name, fan::window::input_action_t::press);
  }
  bool is_action_down(const std::string_view& action_name) {
    return fan::graphics::ctx().input_action->is_active(action_name, fan::window::input_action_t::press_or_repeat);
  }
  bool exists(const std::string_view& action_name) {
    return fan::graphics::ctx().input_action->input_actions.find(action_name) != fan::graphics::ctx().input_action->input_actions.end();
  }

  fan::vec2 get_input_vector(
    const std::string& forward,
    const std::string& back,
    const std::string& left,
    const std::string& right
  ) {
    auto& ia = *fan::graphics::ctx().input_action;
    fan::vec2 v(
      ia.is_down(right) - ia.is_down(left),
      ia.is_down(back) - ia.is_down(forward)
    );
    fan::vec2 v2 = fan::graphics::ctx().window->get_gamepad_axis(fan::graphics::ctx().input_action->get_first_gamepad_key(left));
    if (v2.length() > fan::graphics::ctx().window->gamepad_axis_deadzone) {
      return v2;
    }
    return v;
  }
  fan::vec2 get_input_vector(fan::vec2 scalar) {
    return get_input_vector() * scalar;
  }
  fan::vec2 get_size() {
    return fan::graphics::ctx().window->get_size();
  }
  void set_size(const fan::vec2& size) {
    fan::graphics::ctx().window->set_size(size);
    fan::graphics::ctx()->viewport_set_nr(
      fan::graphics::ctx(),
      fan::graphics::ctx().orthographic_render_view->viewport,
      fan::vec2(0, 0),
      size,
      fan::window::get_size()
    );
    fan::graphics::ctx()->camera_set_ortho(
      fan::graphics::ctx(),
      fan::graphics::ctx().orthographic_render_view->camera,
      fan::vec2(0, size.x),
      fan::vec2(0, size.y)
    );
    fan::graphics::ctx()->viewport_set_nr(
      fan::graphics::ctx(),
      fan::graphics::ctx().perspective_render_view->viewport,
      fan::vec2(0, 0),
      size,
      fan::window::get_size()
    );
    fan::graphics::ctx()->camera_set_ortho(
      fan::graphics::ctx(),
      fan::graphics::ctx().perspective_render_view->camera,
      fan::vec2(0, size.x),
      fan::vec2(0, size.y)
    );
  }
  fan::vec2 get_mouse_position() {
    return fan::graphics::get_mouse_position();
  }
  bool is_mouse_clicked(int button) {
    return fan::graphics::ctx().window->key_state(button) == (int)fan::mouse_state::press;
  }
  bool is_mouse_down(int button) {
    int state = fan::graphics::ctx().window->key_state(button);
    return
      state == (int)fan::mouse_state::press ||
      state == (int)fan::mouse_state::repeat;
  }
  bool is_mouse_released(int button) {
    return fan::graphics::ctx().window->key_state(button) == (int)fan::mouse_state::release;
  }
  fan::vec2 get_mouse_drag(int button) {
    auto* win = fan::graphics::ctx().window;
    if (is_mouse_down(button)) {
      if (win->drag_delta_start != fan::vec2(-1)) {
        return win->get_mouse_position() - win->drag_delta_start;
      }
    }
    return fan::vec2();
  }
  bool is_key_clicked(int key) {
    return fan::graphics::ctx().window->is_key_clicked(key);
  }
  bool is_key_down(int key) {
    return fan::graphics::ctx().window->is_key_down(key);
  }
  bool is_key_released(int key) {
    return fan::graphics::ctx().window->key_state(key) == (int)fan::mouse_state::release;
  }
  bool is_gamepad_button_down(int key) {
    return fan::graphics::ctx().window->is_gamepad_button_down(key);
  }
  bool is_gamepad_axis_active(int key) {
    return fan::graphics::ctx().window->is_gamepad_axis_active(key);
  }
  fan::vec2 get_current_gamepad_axis(int key) {
    return fan::graphics::ctx().window->get_current_gamepad_axis(key);
  }
  bool is_input_clicked(const std::string& name) {
    return fan::graphics::ctx().input_action->is_clicked(name);
  }
  bool is_input_down(const std::string& name) {
    return fan::graphics::ctx().input_action->is_down(name);
  }
  bool is_input_released(const std::string& name) {
    return fan::graphics::ctx().input_action->is_released(name);
  }
  char get_char_pressed() {
    return fan::graphics::ctx().window->get_char_pressed();
  }
}


namespace fan::graphics {
  fan::graphics::image_t image_create() {
    return fan::graphics::ctx()->image_create(fan::graphics::ctx());
  }

  uint64_t image_get_handle(fan::graphics::image_t nr) {
    return fan::graphics::ctx()->image_get_handle(fan::graphics::ctx(), nr);
  }

  void image_erase(fan::graphics::image_t nr) {
    fan::graphics::ctx()->image_erase(fan::graphics::ctx(), nr);
  }

  void image_bind(fan::graphics::image_t nr) {
    fan::graphics::ctx()->image_bind(fan::graphics::ctx(), nr);
  }

  void image_unbind(fan::graphics::image_t nr) {
    fan::graphics::ctx()->image_unbind(fan::graphics::ctx(), nr);
  }

  fan::graphics::image_load_properties_t& image_get_settings(fan::graphics::image_t nr) {
    return fan::graphics::ctx()->image_get_settings(fan::graphics::ctx(), nr);
  }

  void image_set_settings(fan::graphics::image_t nr, const fan::graphics::image_load_properties_t& settings) {
    fan::graphics::ctx()->image_set_settings(fan::graphics::ctx(), nr, settings);
  }

  fan::graphics::image_t image_load(const fan::image::info_t& image_info) {
    return fan::graphics::ctx()->image_load_info(fan::graphics::ctx(), image_info);
  }

  fan::graphics::image_t image_load(const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p) {
    return fan::graphics::ctx()->image_load_info_props(fan::graphics::ctx(), image_info, p);
  }

  fan::graphics::image_t image_load(const std::string& path, const std::source_location& callers_path) {
    return fan::graphics::ctx()->image_load_path(fan::graphics::ctx(), path, callers_path);
  }

  fan::graphics::image_t image_load(const std::string& path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path) {
    return fan::graphics::ctx()->image_load_path_props(fan::graphics::ctx(), path, p, callers_path);
  }

  fan::graphics::image_t image_load(fan::color* colors, const fan::vec2ui& size) {
    return fan::graphics::ctx()->image_load_colors(fan::graphics::ctx(), colors, size);
  }

  fan::graphics::image_t image_load(fan::color* colors, const fan::vec2ui& size, const fan::graphics::image_load_properties_t& p) {
    return fan::graphics::ctx()->image_load_colors_props(fan::graphics::ctx(), colors, size, p);
  }

  fan::graphics::image_t image_load(std::span<const fan::color> colors, const fan::vec2ui& size) {
    return fan::graphics::ctx()->image_load_colors_props(fan::graphics::ctx(), const_cast<fan::color*>(colors.data()), size, image_presets::pixel_art());
  }

  void image_unload(fan::graphics::image_t nr) {
    fan::graphics::ctx()->image_unload(fan::graphics::ctx(), nr);
  }

  bool is_image_valid(fan::graphics::image_t nr) {
    return nr != fan::graphics::ctx().default_texture && nr.iic() == false;
  }

  fan::graphics::image_t image_load_pixel_art(const std::string& path) {
    return image_load(path, image_presets::pixel_art());
  }

  fan::graphics::image_t image_load_smooth(const std::string& path) {
    return image_load(path, image_presets::smooth());
  }

  fan::graphics::image_t create_missing_texture() {
    return fan::graphics::ctx()->create_missing_texture(fan::graphics::ctx());
  }

  fan::graphics::image_t create_transparent_texture() {
    return fan::graphics::ctx()->create_transparent_texture(fan::graphics::ctx());
  }

  void image_reload(fan::graphics::image_t nr, const fan::image::info_t& image_info) {
    fan::graphics::ctx()->image_reload_image_info(fan::graphics::ctx(), nr, image_info);
  }

  void image_reload(fan::graphics::image_t nr, const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p) {
    fan::graphics::ctx()->image_reload_image_info_props(fan::graphics::ctx(), nr, image_info, p);
  }

  void image_reload(fan::graphics::image_t nr, const std::string& path, const std::source_location& callers_path) {
    fan::graphics::ctx()->image_reload_path(fan::graphics::ctx(), nr, path, callers_path);
  }

  void image_reload(fan::graphics::image_t nr, const std::string& path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path) {
    fan::graphics::ctx()->image_reload_path_props(fan::graphics::ctx(), nr, path, p, callers_path);
  }

  fan::graphics::image_t image_create(const fan::color& color) {
    return fan::graphics::ctx()->image_create_color(fan::graphics::ctx(), color);
  }

  fan::graphics::image_t image_create(const fan::color& color, const fan::graphics::image_load_properties_t& p) {
    return fan::graphics::ctx()->image_create_color_props(fan::graphics::ctx(), color, p);
  }

#if defined(FAN_OPENGL)
  std::vector<uint8_t> read_pixels(const fan::vec2& position, const fan::vec2& size) {
    std::vector<uint8_t> pixels(size.multiply() * 4);
    glReadPixels(position.x, position.y, size.x, size.y, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    return pixels;
  }
#endif

  fan::graphics::shader_t shader_create() {
    return fan::graphics::ctx()->shader_create(fan::graphics::ctx());
  }
  fan::graphics::shader_t shader_create(
    const std::string_view vertex_file_path,
    const fan::str_view_t vertex, 
    const std::string_view fragment_file_path,
    const fan::str_view_t fragment) 
  {
    if (ctx().get_renderer() == fan::window_t::renderer_t::opengl) {
      fan::graphics::shader_t shader = ctx()->shader_create(ctx());
      ctx()->shader_set_vertex(ctx(), shader, vertex_file_path, std::string(vertex));
      ctx()->shader_set_fragment(ctx(), shader, fragment_file_path, std::string(fragment));
      if (!ctx()->shader_compile(ctx(), shader)) {
        ctx()->shader_erase(ctx(), shader);
        shader.sic();
      }
      return shader;
    }
    else {
      fan::print("todo");
    }
    return {};
  }

  void shader_erase(fan::graphics::shader_nr_t nr) {
    fan::graphics::ctx()->shader_erase(fan::graphics::ctx(), nr);
  }

  void shader_use(fan::graphics::shader_nr_t nr) {
    fan::graphics::ctx()->shader_use(fan::graphics::ctx(), nr);
  }

  void shader_set_vertex(fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& vertex_code) {
    fan::graphics::ctx()->shader_set_vertex(fan::graphics::ctx(), nr, file_path, vertex_code);
  }

  void shader_set_fragment(fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& fragment_code) {
    fan::graphics::ctx()->shader_set_fragment(fan::graphics::ctx(), nr, file_path, fragment_code);
  }

  bool shader_compile(fan::graphics::shader_nr_t nr) {
    return fan::graphics::ctx()->shader_compile(fan::graphics::ctx(), nr);
  }

  fan::graphics::shader_t get_sprite_shader(const std::string_view fragment_file_path, const fan::str_view_t fragment) {
    if (ctx().get_renderer() == fan::window_t::renderer_t::opengl) {
      auto str = fan::graphics::read_shader("shaders/opengl/2D/objects/sprite.vs");
      return fan::graphics::shader_create(
        "shaders/opengl/2D/objects/sprite.vs",
        str, 
        fragment_file_path,
        fragment
      );
    }
    else {
      fan::print("todo");
    }
    return {};
  }

  std::string read_shader(
    std::string_view path,
    const std::source_location& callers_path
  ) {
    std::string code;
    auto found = fan::io::file::find_relative_path(path, callers_path);
    if (found.empty()) {
      return code;
    }
    fan::io::file::read(found, &code);
    return code;
  }


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

  void camera_set_position(const fan::vec3& cp) {
    camera_set_position(*fan::graphics::ctx().orthographic_render_view, cp);
  }

  fan::vec3 camera_get_center(fan::graphics::camera_nr_t nr) {
    return fan::graphics::ctx()->camera_get_center(fan::graphics::ctx(), nr);
  }

  void camera_set_center(fan::graphics::camera_nr_t nr, const fan::vec3& cp) {
    fan::graphics::ctx()->camera_set_center(fan::graphics::ctx(), nr, cp);
  }

  void camera_set_center(const fan::vec3& cp) {
    camera_set_center(*fan::graphics::ctx().orthographic_render_view, cp);
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
    auto& c = camera_get(nr);
    fan::vec2 offset = fan::vec2(c.coordinates.left + c.coordinates.right, c.coordinates.top + c.coordinates.bottom) / (2.f * c.zoom);
    
    fan::vec2 src_center = camera_get_position(nr) + offset;
    fan::vec2 new_center = move_speed == 0 
      ? target 
      : src_center + (target - src_center) * fan::graphics::get_window().m_delta_time * move_speed;

    camera_set_center(nr, fan::vec3(new_center, 0.f));
  }

  void camera_set_target(const fan::vec2& target, f32_t move_speed) {
    camera_set_target(fan::graphics::get_orthographic_render_view(), target, move_speed);
  }

  void camera_look_at(fan::graphics::camera_nr_t nr, const fan::vec2& target, f32_t move_speed) {
    camera_set_target(nr, target, move_speed);
  }

  void camera_look_at(const fan::vec2& target, f32_t move_speed) {
    camera_set_target(fan::graphics::get_orthographic_render_view(), target, move_speed);
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
}