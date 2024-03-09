#include fan_pch

struct pile_t {

  pile_t() {
    fan::vec2 window_size = loco.window.get_size();
    fan::vec2 ratio = window_size / window_size.max();
    // wed will be inaccurate with low float multiplier with ndc
    loco.open_camera(
      &camera,
      fan::vec2(0, 800) * ratio.x,
      fan::vec2(0, 800) * ratio.y
    );
    loco.open_viewport(&viewport, 0, loco.window.get_size());
    loco.window.add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      fan::vec2 window_size = d.window->get_size();
      fan::vec2 ratio = window_size / window_size.max();
      //std::swap(ratio.x, ratio.y);
      //pile_t* pile = (pile_t*)userptr;
      camera.set_ortho(
        fan::vec2(0, window_size.x) * ratio.x,
        fan::vec2(0, window_size.y) * ratio.y
      );
      viewport.set(0, loco.window.get_size(), loco.window.get_size());
    });
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
  tp.position = fan::vec2(200, 200);
  tp.size = fan::vec2(300, 100);

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
  loco_t::theme_t gray_theme = loco_t::themes::deep_red();
  
  tp.theme = &gray_theme;
  constexpr auto count = 10;
  
  loco_t::shape_t tb0 = tp;
  tp.position = fan::vec2(600, 600);
  tp.text = "abcdefghijlmnopqrstuvxyz";
  loco_t::shape_t tb1 = tp;

  pile->loco.set_vsync(false);

  //tb1.set_text("abc");
  //tb1.set_font_size(64);

  pile->loco.loop([&] {
    pile->loco.get_fps();
    //pile->loco.button.set(&cids[0], &loco_t::button_t::instance_t::position, fan::vec2(-0.5, .5));
  });

  return 0;
}