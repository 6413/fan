// rectangle text button using loco

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_vulkan

#define loco_window
#define loco_context

#define loco_menu_maker 
#include _FAN_PATH(graphics/loco.h)

struct pile_t {

  void open() {
    fan::vec2 window_size = loco.get_window()->get_size();
    fan::vec2 ratio = window_size / window_size.max();
    loco.open_matrices(
      &matrices,
      fan::vec2(-1, 1) * ratio.x,
      fan::vec2(-1, 1) * ratio.y
    );
    loco.get_window()->add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      fan::vec2 window_size = d.size;
      fan::vec2 ratio = window_size / window_size.max();
      //std::swap(ratio.x, ratio.y);
      matrices.set_ortho(
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
  loco_t::matrices_t matrices;
  fan::graphics::viewport_t viewport;
};

int main() {

  pile_t* pile = new pile_t;
  pile->open();

  loco_t::menu_maker_t::nr_t ids[1];
  ids[0] = pile->loco.menu_maker.instances.gnric();
  auto invalidate_nr = [&] {
    if (pile->loco.menu_maker.instances.inric(ids[0])) {
      return;
    }
    pile->loco.menu_maker.erase_menu(ids[0]);
    ids[0] = pile->loco.menu_maker.instances.gnric();
  };
  auto push_menu = [&] (auto mb, const fan::string& element_name) {
    pile_t* pile = OFFSETLESS(OFFSETLESS(mb.vfi, loco_t, vfi), pile_t, loco);
    loco_t::menu_maker_t::properties_t p;
    p.text = element_name;
    p.mouse_button_cb = [&](const loco_t::mouse_button_data_t& mb) -> int {
      if (mb.button != fan::mouse_left) {
        invalidate_nr();
        return 0;
      }
      if (mb.button_state != fan::mouse_state::release) {
        invalidate_nr();
        return 0;
      }
      invalidate_nr();
      pile_t* pile = OFFSETLESS(OFFSETLESS(mb.vfi, loco_t, vfi), pile_t, loco);
      fan::graphics::cid_t* cid = mb.cid;
      if (mb.mouse_stage == loco_t::vfi_t::mouse_stage_e::inside) {
        pile->loco.button.set_theme(cid, pile->loco.button.get_theme(cid), loco_t::button_t::press);
      }
      else {
        pile->loco.button.set_theme(cid, pile->loco.button.get_theme(cid), loco_t::button_t::inactive);
      }
      return 0;
    };
    pile->loco.menu_maker.push_back(ids[0], p);
  };

  loco_t::menu_maker_t::open_properties_t op;
  op.matrices = &pile->matrices;
  op.viewport = &pile->viewport;
  loco_t::theme_t theme = loco_t::themes::deep_red();
  theme.open(pile->loco.get_context());
  op.theme = &theme;
  op.gui_size = 0.03;

  loco_t::vfi_t::properties_t vfip;
  vfip.shape_type = loco_t::vfi_t::shape_t::rectangle;
  vfip.shape.rectangle.position = fan::vec3(0, 0, 1);
  vfip.shape.rectangle.matrices = op.matrices;
  vfip.shape.rectangle.viewport = op.viewport;
  vfip.shape.rectangle.size = fan::vec2(1, 1);

  vfip.mouse_button_cb = [&](const loco_t::vfi_t::mouse_button_data_t& mb) -> int {
    if (mb.button != fan::mouse_right) {
      invalidate_nr();
      return 0;
    }
    if (mb.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
      invalidate_nr();
      return 0;
    }
    if (mb.button_state != fan::mouse_state::release) {
      invalidate_nr();
      return 0;
    }
    if (pile->loco.menu_maker.instances.inric(ids[0])) {
      op.position = mb.position + pile->loco.menu_maker.get_button_measurements(op.gui_size);
      op.position.z = 2;
      ids[0] = pile->loco.menu_maker.push_menu(op);
      push_menu(mb, L"button");
      push_menu(mb, L"text");
      push_menu(mb, L"sprite");
    }

    return 0;
  };
  auto shape_id = pile->loco.push_back_input_hitbox(vfip);

  vfip = {};
  vfip.shape_type = loco_t::vfi_t::shape_t::always;
  vfip.shape.always.z = 0;

  shape_id = pile->loco.push_back_input_hitbox(vfip);

  pile->loco.get_context()->set_vsync(pile->loco.get_window(), 0);

  pile->loco.loop([&] {
    pile->loco.get_fps();
  });

  return 0;
}
