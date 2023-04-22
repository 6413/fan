// rectangle text button using loco

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif

#define fan_debug 0

#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

//#define loco_vulkan
#define loco_window
#define loco_context

#define loco_text_box
#include _FAN_PATH(graphics/loco.h)

struct pile_t {

  pile_t() {
    fan::vec2 window_size = loco.get_window()->get_size();
    fan::vec2 ratio = window_size / window_size.max();
    loco.open_camera(
      &camera,
      fan::vec2(0, 800) * ratio.x,
      fan::vec2(0, 800) * ratio.y
    );
    loco.get_window()->add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      fan::vec2 window_size = d.window->get_size();
      fan::vec2 ratio = window_size / window_size.max();
      //std::swap(ratio.x, ratio.y);
      //pile_t* pile = (pile_t*)userptr;
      camera.set_ortho(
        &loco,
        fan::vec2(0, 800) * ratio.x,
        fan::vec2(0, 800) * ratio.y
      );
      viewport.set(loco.get_context(), 0, loco.get_window()->get_size(), loco.get_window()->get_size());
    });

    fan::vec2 position = 0;
    fan::vec2 size = loco.get_window()->get_size();
    //position.y -= 200;
    //position.y += size.y / 2;
    //size.y /= 2;
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), position, size, loco.get_window()->get_size());
  }

  loco_t loco;
  loco_t::camera_t camera;
  fan::graphics::viewport_t viewport;
};


pile_t* pile = new pile_t;

int main() {

  loco_t::text_box_t::properties_t tp;
  tp.camera = &pile->camera;
  tp.viewport = &pile->viewport;
  // tp.position = 400;
  tp.position = fan::vec2(200, 200);
  //tp.position.y = 0;
 // tp.position.z = 50;
  tp.size = fan::vec2(300, 100)
    //* 300
    ;
  tp.text = "W||||W";
  tp.font_size = 32;
  //tp.font_size = 32;
  tp.mouse_move_cb = [](const loco_t::mouse_move_data_t& mm_d) -> int {
    return 0;
  };
  tp.mouse_button_cb = [](const loco_t::mouse_button_data_t& ii_d) -> int {
    /* if (ii_d.button_state == fan::key_state::press) {
       ii_d.flag->ignore_move_focus_check = true;
     }
     else {
       ii_d.flag->ignore_move_focus_check = false;
     }*/
    return 0;
  };
  loco_t::theme_t gray_theme = loco_t::themes::gray();
  gray_theme.open(pile->loco.get_context());
  gray_theme.button.text_color.r /= 1.1;
  gray_theme.button.text_color.g /= 1.1;
  gray_theme.button.text_color.b /= 1.1;
  //((loco_t::theme_t*)pile->loco.get_context()->theme_list[gray_theme.theme_reference].theme_id)->
  
  tp.theme = &gray_theme;
  constexpr auto count = 10;
  
  loco_t::shape_t tb0 = tp;
  tp.position = fan::vec2(600, 600);
  tp.text = "test";
  loco_t::shape_t tb1 = tp;

  //pile->loco.button.set_theme(&cids[0], &gray_theme, 0.1);

  pile->loco.set_vsync(false);

  pile->loco.loop([&] {
    pile->loco.get_fps();
    //pile->loco.button.set(&cids[0], &loco_t::button_t::instance_t::position, fan::vec2(-0.5, .5));
  });

  return 0;
}