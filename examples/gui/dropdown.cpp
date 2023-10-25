#include fan_pch

int main() {
  loco_t loco;

  loco_t::theme_t theme = loco_t::themes::gray();

  loco_t::dropdown_t::open_properties_t op;
  op.camera = &fan::graphics::default_camera->camera;
  op.viewport = &fan::graphics::default_camera->viewport;
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

    p.text = "0";
    menu0.add(p);

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
  op.direction = fan::vec2(0, 1);
  op.gui_size = fan::vec2(0.1, 0.05);
  op.text_box = true;
  loco_t::dropdown_t::menu_id_t menu1;
  menu1.open(op);

  {
    loco_t::dropdown_t::element_properties_t p;

    p.text = "1";
    menu1.add(p);

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

  loco.loop([&] {
   // pile->loco.get_fps();
  });

  return 0;
}