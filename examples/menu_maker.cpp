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
    loco.get_window()->add_resize_callback(this, [](fan::window_t* window, const fan::vec2i& size, void* userptr) {
      fan::vec2 window_size = window->get_size();
      fan::vec2 ratio = window_size / window_size.max();
      //std::swap(ratio.x, ratio.y);
      pile_t* pile = (pile_t*)userptr;
      pile->matrices.set_ortho(
        fan::vec2(-1, 1) * ratio.x,
        fan::vec2(-1, 1) * ratio.y
      );
      pile->viewport.set(pile->loco.get_context(), 0, pile->loco.get_window()->get_size(), pile->loco.get_window()->get_size());
     });

    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, window_size, window_size);
  }

  loco_t loco;
  fan::opengl::matrices_t matrices;
  fan::opengl::viewport_t viewport;
};

int main() {

  pile_t pile;
  pile.open();

  loco_t::menu_maker_t::id_t ids[2];

  loco_t::menu_maker_t::open_properties_t op;
  op.matrices = &pile.matrices;
  op.viewport = &pile.viewport;
  fan_2d::graphics::gui::theme_t theme = fan_2d::graphics::gui::themes::deep_red();
  theme.open(pile.loco.get_context());
  op.theme = &theme;
  op.gui_size = 0.05;
  op.position = fan::vec2(-1.0 + op.gui_size * 5, -1.0 + op.gui_size * 1);
  ids[0] = pile.loco.menu_maker.push_menu(op);
  op.gui_size *= 3;
  op.position = fan::vec2(op.gui_size * (5.0 / 3), -1.0 + op.gui_size);
  ids[1] = pile.loco.menu_maker.push_menu(op);
  loco_t::menu_maker_t::properties_t p;
  p.text = "Create New Stage";
  p.mouse_button_cb = [&](const loco_t::mouse_button_data_t& mb) {
    if (mb.button != fan::mouse_left) {
      return;
    }
    if (mb.button_state != fan::key_state::release) {
      return;
    }
    pile_t* pile = OFFSETLESS(OFFSETLESS(mb.vfi, loco_t, vfi), pile_t, loco);
    loco_t::menu_maker_t::properties_t p;
    static int x = 0;
    p.text = std::string("Stage") + fan::to_string(x++);
    p.mouse_button_cb = [](const loco_t::mouse_button_data_t& mb) {
      if (mb.button != fan::mouse_left) {
        return;
      }
      if (mb.button_state != fan::key_state::release) {
        return;
      }
      pile_t* pile = OFFSETLESS(OFFSETLESS(mb.vfi, loco_t, vfi), pile_t, loco);
      fan::opengl::cid_t* cid = mb.cid;
      if (mb.mouse_stage == loco_t::vfi_t::mouse_stage_e::inside) {
        pile->loco.button.set_theme(cid, pile->loco.button.get_theme(cid), loco_t::button_t::press);
      }
      else {
        pile->loco.button.set_theme(cid, pile->loco.button.get_theme(cid), loco_t::button_t::inactive);
      }
    };
    pile->loco.menu_maker.push_back(ids[1], p);
  };

  pile.loco.menu_maker.push_back(ids[0], p);
  p.text = "Gui stage";
  pile.loco.menu_maker.push_back(ids[0], p);
  p.text = "Function stage";
  pile.loco.menu_maker.push_back(ids[0], p);

  pile.loco.loop([&] {

  });

  return 0;
}