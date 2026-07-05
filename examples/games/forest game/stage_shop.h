void open(void* sod) {
  fan::time::timer t{ true };
  pile.player.body.set_physics_position(fan::vec2(320.384949, 382.723236));
  pile.is_map_changing = false;
  
  pile.active_map_id = main_map_id = pile.renderer.add(pile.renderer.get_compiled(stage_name), {
    .position = pile.player.body.get_position(),
    .size = fan::vec2(16, 9) / 1.5f,
    .depth_fn = tilemap_loader_t::default_depth_fn
  });
  
  pile.engine.camera_set_position(pile.engine.orthographic_render_view.camera, pile.player.body.get_position());
  player_sensor_door = pile.renderer.add_sensor_rectangle(main_map_id, "player_sensor_door");
  if (!player_sensor_door) fan::throw_error("sensor not found");
  pile.player.light.set_size(0);
  
  auto& map = *pile.renderer.get_map_node(main_map_id).compiled_map;
  f32_t tile_y = map.tile_size.y * 2.f;
  fan::vec2 v_pos = player_sensor_door.get_position() + fan::vec2(0, tile_y * (3.7f - map.map_size.y));
  
  vendor = fan::graphics::sprite_t(
    fan::vec3(v_pos, fan::graphics::get_player_depth_from_y(v_pos, 16.f, tile_y)),
    fan::vec2(8, 16.f),
    vendor_image
  );
  
  f32_t s_size = vendor.get_size().max() * 1.2f;
  vendor_buy_sensor = fan::physics::create_sensor_rectangle(vendor.get_position() + fan::vec2(0, s_size * 2.f), s_size);
  gui::print("The map was in: ", t.seconds(), " seconds.");
}
void close() {
  pile.player.light.set_size(pile.player.light_size);
  vendor_buy_sensor.destroy();
  pile.renderer.erase(main_map_id);
}

fan::event::task_t dialogue() {
  size_t last = std::size(vendor_dialogue) - 1;
  for (size_t i = 0; i < last; ++i) {
    co_await dialogue_box.text_delayed("Vendor", vendor_dialogue[i]);
    co_await dialogue_box.wait_user_input();
  }
  gui::print("Choice selected: ", co_await dialogue_box.choice("Vendor", vendor_dialogue[last], std::span(vendor_answers), fan::vec2(0.8f, 0.1f), 0.3f));
}
void render_dialogue() {
  gui::style_scope_t style;
    style.color(gui::col_window_bg, (0x3B2A1A_rgb).set_alpha(0.90f))
    .color(gui::col_border, (0xA68B5B_rgb).set_alpha(0.95f))
    .var(gui::style_var_window_border_size, 5.0f)
    .var(gui::style_var_window_rounding, 8.f)
    .var(gui::style_var_window_padding, fan::vec2(20, 20.f));

  auto ws = gui::get_window_size();
  dialogue_box.font_size = 43.2f;
  dialogue_box.render("Dialogue box", gui::get_font(43.2f), fan::vec2(ws.x / 1.2f, ws.y / 5.0f), ws.x / 2, 32, [&] {
    ws = gui::get_window_size();
    gui::image(vendor_image, ws.y / 1.2f, {0,0}, {1, 0.46875f});
    gui::same_line();
    dialogue_box.cursor_position.y = 0;
    dialogue_box.set_indent(ws.y - 20.f);
  });
}

void update() {
  if (!dialogue_task.valid() && fan::physics::is_on_sensor(pile.player.body, vendor_buy_sensor) && pile.engine.is_key_clicked(fan::key_e)) {
    dialogue_task = dialogue();
  }
  if (dialogue_task.valid()) render_dialogue();

  if (!pile.is_map_changing && fan::physics::is_on_sensor(pile.player.body, player_sensor_door)) {
    pile.is_map_changing = true;
    pile.engine.stage_change<stage_shop_t, stage_forest_t>();
  }
  pile.renderer.update(main_map_id, pile.player.body.get_position());
  pile.step();
}

tilemap_loader_t::id_t main_map_id;
fan::physics::body_id_t player_sensor_door, vendor_buy_sensor;
fan::graphics::sprite_t vendor;
fan::graphics::image_t vendor_image{"npc/vendor/vendor.png", image_presets::pixel_art()};
static inline std::string_view vendor_dialogue[] = { "Hello there!", "Feel free to browse the shop.", "I also buy junk." };
static inline std::string_view vendor_answers[] = { "Yes, gladly!", "No." };
fan::graphics::gui::dialogue_box_t dialogue_box;
fan::event::task_t dialogue_task;