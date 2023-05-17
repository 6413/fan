// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

//#define loco_vulkan

#define loco_window
#define loco_context
#define loco_rectangle
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
    loco.open_viewport(
      &viewport,
      0,
      loco.get_window()->get_size()
    );
  }

  loco_t loco;
  loco_t::camera_t camera;
  loco_t::viewport_t viewport;
};

bool isBlackKey(int key) {
  // black keys are the keys that are not part of the C major scale
  int keyInOctave = (key + 8) % 12;
  return keyInOctave == 1 || keyInOctave == 3 || keyInOctave == 6 || keyInOctave == 8 || keyInOctave == 10;
}

int main() {

  pile_t pile;

  loco_t::rectangle_t::properties_t rp;
  rp.camera = &pile.camera;
  rp.viewport = &pile.viewport;
  rp.position = fan::vec2(-1, 0);
  rp.color = fan::colors::white;
  

  constexpr int keys = 89;
  double pos = -1.0;
  loco_t::shape_t piano_keys[keys];
  for (int i = 0; i < keys; i++) {
    rp.size = fan::vec2(0.01, 0.2);
    if (isBlackKey(i)) {
      rp.color = fan::colors::green;
      rp.size = fan::vec2(0.01, 0.1);
      rp.position = fan::vec2(pos, -rp.size.y);
    }
    else {
      rp.color = fan::colors::white;
      rp.position = fan::vec2(pos, 0);
    }
    piano_keys[i] = rp;
    pos += rp.size.x * 2.1;

  }

  // key hertz
  auto f = [](float n) {
    return pow(2, (n - 49) / 12) * 440;
  };

  pile.loco.get_window()->add_buttons_callback([&](const auto& data) {
    if (data.button != fan::mouse_left) {
      return;
    }
    if (data.state != fan::mouse_state::press) {
      return;
    }
    for (int i = 0; i < keys; i++) {
      bool mouse_collides = fan_2d::collision::rectangle::point_inside_no_rotation(
        pile.loco.get_mouse_position(pile.camera, pile.viewport),
        piano_keys[i].get_position(),
        piano_keys[i].get_size()
      );
      if (mouse_collides) {
        Beep(f(i), 250);
        fan::print(f(i));
      }
    }
    });

  //shape.set_position(fan::vec2(0, 0));
  //shape.set_size(fan::vec2(1, 1));




  pile.loco.loop([] {

    });

  return 0;
}