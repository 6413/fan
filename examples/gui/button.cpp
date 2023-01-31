// rectangle text button using loco

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 1
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

//#define loco_vulkan

#define loco_framebuffer

#define loco_window
#define loco_context

//#define loco_post_process
#define loco_button
#include _FAN_PATH(graphics/loco.h)

struct pile_t {

  void open() {
    fan::vec2 window_size = loco.get_window()->get_size();
    fan::vec2 ratio = window_size / window_size.max();
    loco.open_matrices(
      &matrices,
      fan::vec2(-1, 1),
      fan::vec2(-1, 1)
    );
    loco.get_window()->add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      fan::vec2 window_size = d.window->get_size();
      fan::vec2 ratio = window_size / window_size.max();
      //std::swap(ratio.x, ratio.y);
      //pile_t* pile = (pile_t*)userptr;
      matrices.set_ortho(
        &loco,
        fan::vec2(-1, 1) * ratio.x,
        fan::vec2(-1, 1) * ratio.y
      );

     });
    loco.open_viewport(&viewport, 0, loco.get_window()->get_size());
  }

  loco_t loco;
  loco_t::matrices_t matrices;
  fan::graphics::viewport_t viewport;
};

int main() {

  fan::sys::set_utf8_cout();

  pile_t pile;
  pile.open();

  loco_t::button_t::properties_t tp;
  tp.matrices = &pile.matrices;

  tp.viewport = &pile.viewport;
 // tp.position = 400;
  tp.position = fan::vec3(0, 0, 0);
  //tp.position.y = 0;
 // tp.position.z = 50;
  tp.font_size = 0.1;
  tp.size = fan::vec2(0.3, 0.1);
  tp.text = "$€ fan";
  //tp.font_size = 32;
  tp.mouse_move_cb = [] (const loco_t::mouse_move_data_t& mm_d) -> int {
    //fan::print(mm_d.position, (int)mm_d.mouse_stage);
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

  loco_t::theme_t theme(pile.loco.get_context(), loco_t::themes::gray(0.5));
  theme.mouse_move_cb = [&](const auto& d) -> int {
    fan::print(pile.loco.button.get_text(d.cid));
    return 0;
  };
  tp.theme = &theme;
  constexpr auto count = 10;
  fan::graphics::cid_t cids[count];
  pile.loco.button.push_back(&cids[0], tp);
  tp.position.x += 0.3;
  //tp.position.z += 2;
  tp.text = "$€ fan $€";
  pile.loco.button.push_back(&cids[1], tp);
  //pile.loco.get_context()->opengl.glPolygonMode(fan::opengl::GL_FRONT_AND_BACK, fan::opengl::GL_LINE);
  pile.loco.loop([&] {

  });

  return 0;
}