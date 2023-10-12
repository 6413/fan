// rectangle text button using loco

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 1
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

//#define loco_vulkan

#define loco_window
#define loco_context

//#define loco_post_process
#define loco_button
#include _FAN_PATH(graphics/loco.h)

loco_t::shape_t b0, b1;

struct pile_t {

  pile_t() {
    static constexpr fan::vec2 original_window_size{ 800, 800 };

    auto viewport_size = fan::vec2(original_window_size.x, original_window_size.y);

    fan::vec2 ratio = viewport_size / viewport_size.max();
    loco.open_camera(
      &camera,
      fan::vec2(-1, 1) * ratio.x,
      fan::vec2(-1, 1) * ratio.y
    );

    loco.get_window()->add_resize_callback([this](const fan::window_t::resize_cb_data_t& d) {
      fan::vec2 window_size = d.size;
      viewport.set(fan::vec2(0, 0), window_size, window_size);

      camera.set_ortho(
        fan::vec2(-1, 1),
        fan::vec2(-1, 1),
        &viewport
      );
     });
    loco.open_viewport(&viewport, fan::vec2(0, 0), viewport_size);
    viewport.original_viewport_size = viewport_size;
  }

  loco_t loco;
  loco_t::camera_t camera;
  fan::graphics::viewport_t viewport;
};

pile_t* pile = new pile_t;

struct some_data_t {
  loco_t::camera_t camera;
};

int main() {

  loco_t::button_t::properties_t tp;
  tp.camera = &pile->camera;

  tp.viewport = &pile->viewport;
 // tp.position = 400;
  tp.position = fan::vec3(0, 0, 0);
  //tp.position.y = 0;
 // tp.position.z = 50;
  // font size is 0-1
  tp.size = fan::vec2(.100, .30);

  tp.mouse_move_cb = [] (const loco_t::mouse_move_data_t& mm_d) -> int {
    //fan::print(mm_d.position, (int)mm_d.mouse_stage);
    return 0;
  };
  tp.mouse_button_cb = [](const loco_t::mouse_button_data_t& ii_d) -> int {
    if (ii_d.button_state != fan::mouse_state::press) {
      return 0;
    }
   /* if (ii_d.button_state == fan::key_state::press) {
      ii_d.flag->ignore_move_focus_check = true;
    }
    else {
      ii_d.flag->ignore_move_focus_check = false;
    }*/
    system("start chrome.exe");
//    fan::print("moi");
    return 0;
  };

  loco_t::theme_t theme = loco_t::themes::deep_red();
  tp.theme = &theme;
  tp.text = "open chrome";
  //tp.position = 400;
  b0 = tp;

  tp.position.x = .500;
  b1 = tp;
  /*original_size = tp.size;
  original_position = tp.position;*/

  fan::vec2 ws = pile->loco.get_window()->get_size();

  auto vs = fan::vec2(ws.x, ws.y / 2);

  fan::vec2 ratio = vs / ws.max();

  //b0.set_position(original_position * ratio);
  //b0.set_size(original_size * ratio.min());

  pile->loco.loop([&] {

  });

  return 0;
}
