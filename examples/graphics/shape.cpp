#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 3
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_window
#define loco_context

void* global_loco;

#define loco_rectangle
#include _FAN_PATH(graphics/loco0.h)

#define WITCH_INCLUDE_PATH C:\libs\WITCH
#include _INCLUDE_TOKEN(WITCH_INCLUDE_PATH, WITCH.h)

#include <fan/audio/audio.h>

using engine_t = engine_wrap<loco_t::rectangle_t>;

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
      //viewport.set(loco.get_context(), 0, d.size, d.size);
    });
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, loco.get_window()->get_size(), loco.get_window()->get_size());
  }

  engine_t loco;
  engine_t::camera_t camera;
  fan::graphics::viewport_t viewport;
  fan::graphics::cid_t cid;
};

pile_t* pile = new pile_t;

#define loco_access &pile->loco
#include _FAN_PATH(graphics/loco0_define.h)

int main() {
  using instance_t = loco_t::instance_t;
  loco_t::cid_t cid_list[2];

  uint8_t type = 0;

  switch (type) {
    case 0: {
      pile->loco.push_shape(&cid_list[0], rectangle_t{
        //...
      });
      break;
    }
  }

 /* instance_t rect(
    rectangle_t{
      .camera = &pile->camera,
      .viewport = &pile->viewport,
      .position = fan::vec2(0, 0),
      .size = 0.1,
      .color = fan::colors::blue
    }
  );

  rect = sprite_t{
      .camera = &pile->camera,
      .viewport = &pile->viewport,
      .position = fan::vec2(0, 0),
      .size = 0.1,
      .color = fan::colors::blue
  }*/

  pile->loco.loop([&] {
    //rect.set_position(pile->loco.get_mouse_position(pile->viewport));
  });

  return 0;
}