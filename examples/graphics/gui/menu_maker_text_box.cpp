#include <fan/pch.h>

struct pile_t {

  void open() {
    fan::vec2 window_size = loco.window.get_size();
    fan::vec2 ratio = window_size / window_size.max();
    loco.open_camera(
      &camera,
      fan::vec2(-1, 1) * ratio.x,
      fan::vec2(-1, 1) * ratio.y
    );
    loco.window.add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      fan::vec2 window_size = d.size;
      fan::vec2 ratio = window_size / window_size.max();
      //std::swap(ratio.x, ratio.y);
      camera.set_ortho(
        &loco,
        fan::vec2(-1, 1) * ratio.x,
        fan::vec2(-1, 1) * ratio.y
      );
      viewport.set(loco.get_context(), 0, d.size, d.size);
     });

    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, window_size, window_size);
  }

  loco_t loco;
  loco_t::camera_t camera;
  fan::graphics::viewport_t viewport;
};

int main() {

  pile_t* pile = new pile_t;
  pile->open();

  loco_t::menu_maker_text_box_t::nr_t ids[2];

  loco_t::menu_maker_text_box_t::open_properties_t op;
  op.camera = &pile->camera;
  op.viewport = &pile->viewport;
  loco_t::theme_t theme = loco_t::themes::deep_red();
  theme.open(pile->loco.get_context());
  op.theme = &theme;
  op.gui_size = 0.05;
  op.position = fan::vec2(0, 0);
  ids[0] = pile->loco.menu_maker_text_box.push_menu(op);
  op.gui_size *= 3;
  op.position = fan::vec2(op.gui_size * (5.0 / 3), -1.0 + op.gui_size);
  ids[1] = pile->loco.menu_maker_text_box.push_menu(op);
  loco_t::menu_maker_text_box_t::properties_t p;
  p.text = L"Create New Stage";
  p.mouse_button_cb = [&](const loco_t::mouse_button_data_t& mb) -> int {
    if (mb.button != fan::mouse_left) {
      return 0;
    }
    if (mb.button_state != fan::mouse_state::release) {
      return 0;
    }
    pile_t* pile = OFFSETLESS(OFFSETLESS(mb.vfi, loco_t, vfi), pile_t, loco);
    loco_t::menu_maker_text_box_t::properties_t p;
    static int x = 0;
   // p.text = fan::wstring(L"Stage") + fan::to_wstring(x++);
   /* p.mouse_button_cb = [](const loco_t::mouse_button_data_t& mb) -> int {
      if (mb.button != fan::mouse_left) {
        return 0;
      }
      if (mb.button_state != fan::mouse_state::release) {
        return 0;
      }
      pile_t* pile = OFFSETLESS(OFFSETLESS(mb.vfi, loco_t, vfi), pile_t, loco);
      fan::graphics::cid_t* cid = mb.cid;
      if (mb.mouse_stage == loco_t::vfi_t::mouse_stage_e::inside) {
        pile->loco.button.set_theme(cid, pile->loco.button.get_theme(cid), loco_t::button_t::press);
      }
      else {
        pile->loco.button.set_theme(cid, pile->loco.button.get_theme(cid), loco_t::button_t::inactive);
      }
      return 0;
    };*/
    //pile->loco.menu_maker_text_box.push_back(ids[1], p);

    return 0;
  };
  
  pile->loco.menu_maker_text_box.push_back(ids[0], p);
  /*p.text = L"Gui stage";
  p.mouse_button_cb = [](const loco_t::mouse_button_data_t& mb) -> int {
    return 0;
  };
  pile->loco.menu_maker.push_back(ids[0], p);
  p.text = L"Function stage";
  pile->loco.menu_maker.push_back(ids[0], p);*/

  pile->loco.get_context()->set_vsync(pile->loco.get_window(), 0);

  pile->loco.loop([&] {
    pile->loco.get_fps();
  });

  return 0;
}