#define gui fan::graphics::gui
#define gameplay fan::graphics::gameplay

void init_theme(){
  inventory.style.theme.panel_bg = fan::color(41 / 255.f, 50 / 255.f, 69 / 255.f, 1.f);
  inventory.style.theme.panel_border = fan::color(25 / 255.f, 25 / 255.f, 31 / 255.f, 1.f);
  inventory.style.theme.panel_corner_accent = fan::color(1.f, 1.f, 1.f, 1.f);
  inventory.style.theme.slot_bg = fan::color(0.f, 0.f, 0.f, 205 / 255.f);
  inventory.style.theme.slot_bg_hover = fan::color(78 / 255.f, 96 / 255.f, 131 / 255.f, 1.f);
  inventory.style.theme.slot_border = fan::color(0.f, 0.f, 0.f, 240 / 255.f);
  inventory.style.theme.selected_border_color = fan::color::from_rgba(0xBEC0EAFF);
}
void init_inventory_hotbar(){
  f32_t x_count = 9;
  inventory.create(x_count * 3, x_count);
  inventory.visible = false;
  inventory.style.slot_size = fan::vec2(81, 81);
  inventory.style.slot_padding = fan::vec2(12, 12);
  inventory.style.border_thickness = 3;
  inventory.style.corner_rounding = 8;
  inventory.style.panel_border_thickness = 4;
  inventory.style.panel_corner_rounding = 12;
  inventory.style.padding_left = 32;
  inventory.style.padding_right = 32;
  inventory.style.padding_top = 399;
  inventory.style.padding_bottom = 20;
  gui::bind_inventory_hotbar(inventory, hotbar);

  hotbar.create(9);

  inventory.on_item_use = [](uint32_t slot_index, const gameplay::item_t& item) {
    pile->player.use_item(item);
  };

  hotbar.on_item_use = [](uint32_t slot_index, const gameplay::item_t& item) {
    pile->player.use_item(item);
  };

  //gui::bind_inventory_equipment(inventory, equipment);
  //gui::bind_hotbar_equipment(hotbar, equipment);
}

void open(void* sod) {
  health_empty = fan::graphics::image_t("gui/hp_empty.webp");
  health_full = fan::graphics::image_t("gui/hp_full.webp");
  health_potion = fan::graphics::image_t("gui/health_potion.webp");
  gui::load_fonts(font_pixel, "fonts/PixelatedEleganceRegular-ovyAA.ttf");

  init_theme();
  init_inventory_hotbar();
}

void close() {

}

void handle_inventory_hotbar() {
  //gui::render_theme_editor(inventory.style.theme, inventory, hotbar);
  hotbar.handle_input();
  if (pile->engine.is_input_clicked(actions::toggle_inventory)) {
    inventory.visible = !inventory.visible;
  }
  if (inventory.visible) {
    inventory.render([&]{
      fan::vec2 grid_origin = inventory.get_slot_grid_origin();

      // 20px above grid
      fan::vec2 equip_pos = grid_origin - fan::vec2(0, equipment.get_slot_size().y + 20);

      gui::set_cursor_pos(equip_pos);
      equipment.render_inside_inventory(inventory.style.theme, inventory.drag_state);

      if (inventory.drag_state.active && gui::input::left_released()) {
        auto& drag = inventory.drag_state;

        uint32_t i = inventory.get_hovered_slot();
        uint32_t h = hotbar.get_hovered_slot();
        uint32_t e = equipment.get_hovered_slot();

        if (i != UINT32_MAX && inventory.try_drop_here(i, drag)) return;
        if (h != UINT32_MAX && hotbar.try_drop_here(h, drag)) return;
        if (e != UINT32_MAX && equipment.try_drop_here(e, drag)) return;

        if (inventory.destroy_on_drop_outside) {
          gui::drag_drop::cancel(drag);
        }
        else {
          inventory.return_drag_to_source();
        }
      }

    });
  }
  hotbar.render(
    inventory.style.theme,
    inventory.drag_state,
    inventory.hovered_secondary_slot
  );
}

uint32_t count_all_items(items::id_e id) {
  uint32_t count = 0;

  // inventory
  {
    auto& inv = pile->get_gui().inventory;
    auto slot_index = inv.find_item(id);
    if (slot_index) {
      auto& slot = inv.slots[*slot_index];
      if (!slot.is_empty()) {
        count += *slot.stack_size;
      }
    }
  }
  // hotbar
  {
    auto& hotbar = pile->get_gui().hotbar;
    for (auto& slot : hotbar.slots) {
      if (!slot.is_empty() && *slot.id == id) {
        count += *slot.stack_size;
      }
    }
  }
  // drag
  {
    auto& drag = pile->get_gui().inventory.drag_state;
    if (drag.active && drag.id == id) {
      count += drag.stack_size;
    }
  }

  return count;
}

void render_potions(f32_t prev_image_size) {
  fan::vec2 wnd_size = fan::window::get_size();
  f32_t potion_size = (wnd_size / 48.f).max();

  f32_t cp = 0;
  auto& v = gui::get_style();
  gui::set_cursor_pos_x(cp + v.ItemSpacing.x + prev_image_size / 2.f - potion_size / 2.f);
  gui::image(health_potion, potion_size);
  gui::same_line();

  gui::push_font(gui::get_font(font_pixel, gui::get_font_size()));

  uint32_t potion_count = count_all_items(items::id_e::health_potion);

  std::string potion_text = "x " + std::to_string(potion_count);
  f32_t text_height = gui::get_text_size(potion_text).y;
  gui::set_cursor_pos_y(gui::get_cursor_pos_y() + potion_size - text_height);
  gui::text(potion_text);

  gui::pop_font();
}

void handle_hud() {
  fan::vec2 wnd_size = fan::window::get_size();

  f32_t heart_size = (wnd_size / 32.f).max();
  if (auto hud = gui::hud("##platformer_gui")) {
    int heart_count = pile->player.body.get_max_health() / 10.f;
    for (int i = 0; i < heart_count; ++i) {
      gui::same_line();
      fan::graphics::image_t hp_image = health_empty;
      //0-1
      f32_t progress = pile->player.body.get_health() / pile->player.body.get_max_health();
      if (progress * heart_count > i) {
        hp_image = health_full;
      }
      gui::image(hp_image, heart_size);
    }
    render_potions(heart_size);
  }
}

void update() {
  handle_inventory_hotbar();
  handle_hud();
}

fan::graphics::image_t health_empty;
fan::graphics::image_t health_full;
fan::graphics::image_t health_potion;

gui::font_t* font_pixel[std::size(gui::font_sizes)]{};

gui::inventory_t inventory;
gui::hotbar_t hotbar;

#undef gameplay

gui::gameplay::equipment_t equipment;

#undef gui