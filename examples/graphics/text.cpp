#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

//#define loco_vulkan

#define loco_window
#define loco_context

#define loco_letter
#define loco_text
#include _FAN_PATH(graphics/loco.h)

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  pile_t() {
    loco.open_camera(
      &camera,
      ortho_x,
      ortho_y
    );
    loco.get_window()->add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      fan::vec2 window_size = d.size;
      fan::vec2 ratio = window_size / window_size.max();
      camera.set_ortho(
        ortho_x * ratio.x, 
        ortho_y * ratio.y
      );
      viewport.set(0, window_size, window_size);
    });
    viewport.open();
    viewport.set(0, loco.get_window()->get_size(), loco.get_window()->get_size());
  }

  loco_t loco;
  loco_t::camera_t camera;
  fan::graphics::viewport_t viewport;
};


int main() {
  pile_t* pile = new pile_t;

  loco_t::text_t::properties_t p;

  p.position = 0;

  p.camera = &pile->camera;
  p.viewport = &pile->viewport;

  p.font_size = 0.1;
  p.text = "hello";
  p.color = fan::colors::white;

  loco_t::shape_t text0 = p;

  pile->loco.set_vsync(false);

  f32_t x = 0;

  pile->loco.loop([&] {
    //text0.set_text(fan::random::string(10));
    text0.set_font_size(sin(x) * 5);
    x += pile->loco.get_delta_time();
    pile->loco.get_fps();
  });

  return 0;
}