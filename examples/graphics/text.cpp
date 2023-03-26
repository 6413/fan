#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

//#define loco_vulkan

#define loco_window
#define loco_context

#define loco_no_inline
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
        &loco,
        ortho_x * ratio.x, 
        ortho_y * ratio.y
      );
      viewport.set(loco.get_context(), 0, window_size, window_size);
    });
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, loco.get_window()->get_size(), loco.get_window()->get_size());
  }

  loco_t loco;
  loco_t::camera_t camera;
  fan::graphics::viewport_t viewport;
};

pile_t* pile = new pile_t;

#define loco_access &pile->loco
#include _FAN_PATH(graphics/loco_define.h)

int main() {

  loco_t::text_t::properties_t p;

  p.position = 0;

  p.camera = &pile->camera;
  p.viewport = &pile->viewport;

  p.font_size = 0.05;
  p.text = "01234";
  p.color = fan::colors::white;

  loco_t::id_t text0 = p;
  text0.erase();
  p.text = "56789";
  loco_t::id_t text1 = p;
  text1.set_text("56789");
  text1.set_color(fan::color(1, 0, 0, 0.7));

  pile->loco.set_vsync(false);

  pile->loco.loop([&] {
    pile->loco.get_fps();
  });

  return 0;
}