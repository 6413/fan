module;

#include <vector>
#include <string>
#include <source_location>
#include <cstdint>

module fan.graphics.common_context;

namespace fan::graphics {
  void lighting_t::set_target(const fan::vec3& t, f32_t d) {
    start = ambient;
    target = t;
    duration = d;
    elapsed = 0.0f;
  }
  void lighting_t::update(f32_t delta_time) {
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
#if defined(fan_gui)
  gui_draw_cb_t& get_gui_draw_cbs() {
    return *ctx().gui_draw_cbs;
  }
#endif

  fan::graphics::image_t get_default_texture() {
    return ctx().default_texture;
  }

  image_t::image_t(bool) : fan::graphics::image_nr_t() {}
  image_t::image_t() : fan::graphics::image_nr_t(ctx().default_texture) {}
  image_t::image_t(fan::graphics::image_nr_t image) : fan::graphics::image_nr_t(image) {}
  image_t::image_t(const fan::color& color)
    : fan::graphics::image_nr_t(ctx()->image_create_color(ctx(), color)) {}
  image_t::image_t(const char* path, const std::source_location& callers_path)
    : image_t(std::string(path), callers_path) {}
  image_t::image_t(const std::string& path, const std::source_location& callers_path)
    : fan::graphics::image_nr_t(ctx()->image_load_path(ctx(), path, callers_path)) {}

  image_t::image_t(const char* path, const fan::graphics::image_load_properties_t lp, const std::source_location& callers_path)
    : image_t(std::string(path), lp, callers_path) {}

  image_t::image_t(const std::string& path, const fan::graphics::image_load_properties_t lp, const std::source_location& callers_path) 
    : fan::graphics::image_nr_t(ctx()->image_load_path_props(ctx(), path, lp, callers_path)) {}

  fan::vec2 image_t::get_size() const {
    return fan::graphics::image_get_data(*this).size;
  }
  image_load_properties_t image_t::get_load_properties() const {
    return fan::graphics::image_get_data(*this).image_settings;
  }
  std::string image_t::get_path() const {
    return fan::graphics::image_get_data(*this).image_path;
  }
  image_t::operator fan::graphics::image_nr_t& () {
    return static_cast<fan::graphics::image_nr_t&>(*this);
  }
  image_t::operator const fan::graphics::image_nr_t& () const {
    return static_cast<const fan::graphics::image_nr_t&>(*this);
  }
  bool image_t::valid() const {
    return *this != fan::graphics::ctx().default_texture && iic() == false;
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
}
namespace fan::window {
  fan::vec2 get_input_vector(
    const std::string& forward,
    const std::string& back,
    const std::string& left,
    const std::string& right
  ) {
    auto& ia = *fan::graphics::ctx().input_action;
    fan::vec2 v(
      ia.is_action_down(right) - ia.is_action_down(left),
      ia.is_action_down(back) - ia.is_action_down(forward)
    );
    return v.length() > 0 ? v.normalized() : v;
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
  bool is_key_pressed(int key) {
    return fan::graphics::ctx().window->key_state(key) == (int)fan::mouse_state::press;
  }
  bool is_key_down(int key) {
    int state = fan::graphics::ctx().window->key_state(key);
    return
      state == (int)fan::mouse_state::press ||
      state == (int)fan::mouse_state::repeat;
  }
  bool is_key_released(int key) {
    return fan::graphics::ctx().window->key_state(key) == (int)fan::mouse_state::release;
  }
}