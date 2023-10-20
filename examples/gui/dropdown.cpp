#include fan_pch

struct pile_t {

  void open() {
    int x = offsetof(pile_t, loco);
    fan::vec2 window_size = loco.get_window()->get_size();
    fan::vec2 ratio = window_size / window_size.max();
    loco.open_camera(
      &camera,
      fan::vec2(-1, 1) * ratio.x,
      fan::vec2(-1, 1) * ratio.y
    );
    loco.get_window()->add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      fan::vec2 window_size = d.size;
      fan::vec2 ratio = window_size / window_size.max();
      //std::swap(ratio.x, ratio.y);
      camera.set_ortho(
        fan::vec2(-1, 1) * ratio.x,
        fan::vec2(-1, 1) * ratio.y
      );
      viewport.set(0, d.size, d.size);
      });

    viewport.open();
    viewport.set(0, window_size, window_size);
  }

  loco_t loco;
  loco_t::camera_t camera;
  fan::graphics::viewport_t viewport;
};

int main() {

  pile_t pile;
  pile.open();

  loco_t::theme_t theme = loco_t::themes::gray();

  loco_t::dropdown_t::open_properties_t op;
  op.camera = &pile.camera;
  op.viewport = &pile.viewport;
  op.theme = &theme;
  op.gui_size = fan::vec2(0.05 * 5, 0.05);
  op.direction.x = 0;
  op.direction.y = 1;
  op.position = 0;
  op.title = "test";
  op.titleable = true;
  
  loco_t::dropdown_t::menu_id_t menu0;
  menu0.open(op);

  {
    loco_t::dropdown_t::element_properties_t p;

    //p.mouse_button_cb = ...;
    p.text = "0";
    menu0.add(p);
    //p.mouse_button_cb = ...;
    p.text = "1";
    menu0.add(p);
        p.text = "t";
    menu0.add(p);
        p.text = "1";
    menu0.add(p);
        p.text = "1";
    menu0.add(p);
        p.text = "1";
    menu0.add(p);
        p.text = "1";
    menu0.add(p);
  }

  op.position -= 0.4;
  op.titleable = false;
  op.direction = fan::vec2(1, 0);
  op.gui_size = fan::vec2(0.05, 0.05);
  op.text_box = true;
  loco_t::dropdown_t::menu_id_t menu1;
  menu1.open(op);

  {
    loco_t::dropdown_t::element_properties_t p;

    //p.mouse_button_cb = ...;
    p.text = "1";
    menu1.add(p);
    //p.mouse_button_cb = ...;
    p.text = "3";
    menu1.add(p);
        p.text = "5";
    menu1.add(p);
        p.text = "2";
    menu1.add(p);
        p.text = "6";
    menu1.add(p);
        p.text = "7";
    menu1.add(p);
        p.text = "9";
    menu1.add(p);
  }

  //ids[0] = pile->loco.dropdown.push_menu(op);

  //loco_t::dropdown_t::properties_t p;
  //p.text = "dropdown";
  //p.mouse_button_cb = [&](const loco_t::mouse_button_data_t& mb) -> int {

  //  if (mb.button != fan::mouse_left) {
  //    return 0;
  //  }
  //  if (mb.button_state != fan::mouse_state::release) {
  //    return 0;
  //  }
  //  return 0;
  //};
  //p.items.push_back("apples");
  //p.items.push_back("grapes");

  //pile->loco.dropdown.push_back(ids[0], p);

  //pile->loco.get_context()->set_vsync(pile->loco.get_window(), 0);

  pile.loco.loop([&] {
   // pile->loco.get_fps();
  });

  return 0;
}