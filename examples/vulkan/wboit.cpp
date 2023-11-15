// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 3
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_vulkan

#define loco_window
#define loco_context

//#define loco_post_process

//#define loco_wboit
#define loco_rectangle
#include _FAN_PATH(graphics/loco.h)

constexpr uint32_t count = 10000;

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  pile_t() {
    loco.open_camera(
      &camera,
      ortho_x,
      ortho_y
    );
    loco.window.add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      viewport.set(loco.get_context(), 0, d.size, d.size);
    });
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, loco.window.get_size(), loco.window.get_size());
  }

  loco_t loco;
  loco_t::camera_t camera;
  fan::graphics::viewport_t viewport;
  fan::graphics::cid_t cids[count];
};

int main() {

  pile_t* pile = new pile_t;

  loco_t::shapes_t::rectangle_t::properties_t p;
  p.get_camera() = &pile->camera;
  p.get_viewport() = &pile->viewport;

  p.size = fan::vec2(0.5);
  p.color = fan::color(1, 0, 0, 0.5);
  p.position = fan::vec2(-0.25, 0);
  p.position.z = 1;
  pile->loco.rectangle.push_back(&pile->cids[0], p);
  p.color = fan::color(0, 0, 1, 0.5);
  p.position = fan::vec2(0.25, 0);
  f32_t dd = 0;
  p.position.z = dd;
  pile->loco.rectangle.push_back(&pile->cids[1], p);

  auto& window = *pile->loco.get_window();
  window.add_buttons_callback([&](const fan::window_t::mouse_buttons_cb_data_t& d) {
    if (d.state != fan::mouse_state::press) {
      return;
    }
    if (d.button == fan::mouse_left)
    pile->loco.rectangle.set(
      &pile->cids[0],
      &loco_t::shapes_t::rectangle_t::instance_t::position,
      fan::vec3(pile->loco.get_mouse_position(pile->viewport), 1)
    );
    if (d.button == fan::mouse_right)
    pile->loco.rectangle.set(
      &pile->cids[1],
      &loco_t::shapes_t::rectangle_t::instance_t::position,
      fan::vec3(pile->loco.get_mouse_position(pile->viewport), dd)
    );
  });

  pile->loco.set_vsync(false);
  
  pile->loco.loop([&] {
    pile->loco.get_fps();
  });

  return 0;
}