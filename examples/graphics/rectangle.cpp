// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)
#define loco_window
#define loco_context

#define loco_rectangle
#include _FAN_PATH(graphics/loco.h)

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(0, 800);
  static constexpr fan::vec2 ortho_y = fan::vec2(0, 800);

  pile_t() {
    loco.open_camera(
      &camera,
      ortho_x,
      ortho_y
    );
    loco.get_window()->add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      viewport.set(0, d.size, d.size);
    });
    viewport.open();
    viewport.set(0, loco.get_window()->get_size(), loco.get_window()->get_size());
  }

  loco_t loco;
  loco_t::camera_t camera;
  loco_t::viewport_t viewport;
};

int main() {
  pile_t* pile = new pile_t;

  loco_t::rectangle_t::properties_t rectangle_properties;
  rectangle_properties.camera = &pile->camera;
  rectangle_properties.viewport = &pile->viewport;

  rectangle_properties.position = fan::vec3(400, 400, 0);
  rectangle_properties.size = fan::vec2(100, 100);
  rectangle_properties.color = fan::colors::red;

  loco_t::shape_t rectangle = rectangle_properties;

  pile->loco.set_vsync(false);
  
  f32_t angle = 0;

  pile->loco.loop([&] {
    pile->loco.get_fps();
    //fan::vec3 p = pile->loco.get_window()->get_mouse_position();
    //rectangle.set_position(p);
    //rectangle.set_angle(angle);
    //angle += pile->loco.get_delta_time();
  });

  return 0;
}