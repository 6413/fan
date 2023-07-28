#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_vulkan

#define loco_window
#define loco_context

#define loco_letter
#include _FAN_PATH(graphics/loco.h)

constexpr uint32_t count = 10000;

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  void open() {
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
      viewport.set(loco.get_context(), 0, d.size, d.size );
    });
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, loco.get_window()->get_size(), loco.get_window()->get_size());
  }

  loco_t loco;
  loco_t::camera_t camera;
  fan::graphics::viewport_t viewport;
  fan::graphics::cid_t cids[count];
};

int main() {

  pile_t* pile = new pile_t;
  pile->open();

  loco_t::letter_t::properties_t p;

  p.camera = &pile->camera;
  p.viewport = &pile->viewport;

  for (uint32_t i = 0; i < count; i++) {
    p.position = fan::vec2(fan::random::value_f32(-1, 1), fan::random::value_f32(-1, 1));
    p.color = fan::color(1, 0, f32_t(i) / count, 1);
    p.font_size = 0.1;
    fan::string str = fan::random::string(1);
    std::wstring w(str.begin(), str.end());
    p.letter_id = pile->loco.font.decode_letter(w[0]);

    pile->loco.letter.push_back(&pile->cids[i], p);
  }

  pile->loco.set_vsync(false);

  pile->loco.loop([&] {
    pile->loco.get_fps();
  });

  return 0;
}