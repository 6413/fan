// rectangle text button using loco

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_window
#define loco_context

#define loco_menu_maker
#include _FAN_PATH(graphics/loco.h)

struct pile_t {

  void open() {
    loco.open(loco_t::properties_t());
    fan::vec2 window_size = loco.get_window()->get_size();
    fan::vec2 ratio = window_size / window_size.max();
    fan::graphics::open_matrices(
      loco.get_context(),
      &matrices,
      fan::vec2(-1, 1) * ratio.x,
      fan::vec2(-1, 1) * ratio.y
    );
    loco.get_window()->add_resize_callback([&](fan::window_t* window, const fan::vec2i& size) {
      fan::vec2 window_size = size;
      fan::vec2 ratio = window_size / window_size.max();
      //std::swap(ratio.x, ratio.y);
      matrices.set_ortho(
        fan::vec2(-1, 1) * ratio.x,
        fan::vec2(-1, 1) * ratio.y
      );
      viewport.set(loco.get_context(), 0, size, size);
      });

    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, window_size, window_size);
  }

  loco_t loco;
  fan::opengl::matrices_t matrices;
  fan::opengl::viewport_t viewport;
};

int main() {

  pile_t* pile = new pile_t;
  pile->open();

  loco_t::dropdown_t::nr_t ids[2];

  loco_t::dropdown_t::open_properties_t op;
  op.matrices = &pile->matrices;
  op.viewport = &pile->viewport;
  fan_2d::graphics::gui::theme_t theme = fan_2d::graphics::gui::themes::gray();
  theme.open(pile->loco.get_context());
  op.theme = &theme;
  op.gui_size = 0.05;
  //op.position = fan::vec2(-1.0 + op.gui_size * 5, -1.0 + op.gui_size * 1);
  op.position = 0;
  op.position.z = 0;
  ids[0] = pile->loco.dropdown.push_menu(op);
  op.gui_size *= 3;
  op.position = fan::vec2(op.gui_size * (5.0 / 3), -1.0 + op.gui_size);
  ids[1] = pile->loco.dropdown.push_menu(op);
  loco_t::dropdown_t::properties_t p;
  p.text = "dropdown";
  p.mouse_button_cb = [&](const loco_t::mouse_button_data_t& mb) -> int {
    if (mb.button != fan::mouse_left) {
      return 0;
    }
    if (mb.button_state != fan::key_state::release) {
      return 0;
    }
    return 0;
  };

  p.items = {
    "apples",
    "grapes"
  };
  pile->loco.dropdown.push_back(ids[0], p);

  p.items = {
  "test",
  "button"
  };

  pile->loco.dropdown.push_back(ids[1], p);

  pile->loco.get_context()->set_vsync(pile->loco.get_window(), 0);

  pile->loco.loop([&] {
    pile->loco.get_fps();
  });

  return 0;
}