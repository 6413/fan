void open(void* sod) {
  fan::time::timer t{ true };
  pile.player.body.set_position(fan::vec2(320.384949, 382.723236));
  pile.player.body.set_physics_position(pile.player.body.get_position());
  
  main_map_id = pile.renderer.add(pile.renderer.get_compiled(stage_name), {
    .position = pile.player.body.get_position(),
    .size = fan::vec2(16, 9) / 1.5f,
    .depth_fn = tilemap_loader_t::default_depth_fn
  });
  
  pile.active_map_id = main_map_id;

  pile.loco.camera_set_position(pile.loco.orthographic_render_view.camera, pile.player.body.get_position());

  player_sensor_door = pile.renderer.add_sensor_rectangle(main_map_id, "player_sensor_door");
  if (!player_sensor_door) {
    fan::throw_error("sensor not found");
  }

  old_light_size = pile.player.light.get_size();
  pile.player.light.set_size(0);
  
  vendor_image = pile.loco.image_load("npc/vendor/vendor.png", {
    .min_filter = image_filter_e::nearest,
    .mag_filter = image_filter_e::nearest
  });
  
  auto& map = *pile.renderer.get_map_node(main_map_id).compiled_map;
  fan::vec2 vendor_pos = player_sensor_door.get_position();
  vendor_pos.y += -(map.tile_size.y * 2.f * map.map_size.y) + map.tile_size.y * 2.f * 3.7f;
  
  f32_t tile_size_y = map.tile_size.y * 2.f;
  f32_t vendor_size_y = 16.f;
  
  vendor = fan::graphics::sprite_t{{
    .position = fan::vec3(vendor_pos, fan::graphics::get_player_depth_from_y(vendor_pos, vendor_size_y, tile_size_y)),
    .size = fan::vec2(8, vendor_size_y),
    .image = vendor_image
  }};
  
  fan::vec2 sensor_size = vendor.get_size().max() * 1.2f;
  vendor_buy_sensor = fan::physics::create_sensor_rectangle(vendor.get_position() + fan::vec2(0, sensor_size.y * 2.f), sensor_size);
  
  gui::print("The map was in: ", t.seconds(), " seconds.");
}

void close() {
  pile.player.light.set_size(old_light_size);
  vendor_buy_sensor.destroy();
  pile.renderer.erase(main_map_id);
}

fan::event::task_t dialogue() {
  for (const auto& line : vendor_dialogue) {
    co_await dialogue_box.text_delayed("Vendor", line);
    co_await dialogue_box.wait_user_input();
  }
  is_in_dialogue = false;
}

void render_dialogue() {
  #define gui fan::graphics::gui
  fan::vec2 window_size = gui::get_window_size();
  window_size.x /= 1.2;
  window_size.y /= 5;

  gui::push_style_color(gui::col_window_bg, fan::color::from_rgb(0x3B2A1A).set_alpha(0.90f));
  gui::push_style_color(gui::col_border, fan::color::from_rgb(0xA68B5B).set_alpha(0.95f));

  gui::push_style_var(gui::style_var_window_border_size, 5.0f);
  gui::push_style_var(gui::style_var_window_rounding, 8.f);
  gui::push_style_var(gui::style_var_window_padding, fan::vec2(20, 20.f));

  dialogue_box.font_size = 24.f * 1.8f;
  dialogue_box.render(
    "Dialogue box",
    gui::get_font(dialogue_box.font_size),
    window_size,
    gui::get_window_size().x / 2,
    32,
    [&] {
      gui::image(vendor_image, gui::get_window_size().y / 1.2f, fan::vec2(0), fan::vec2(1, 0.46875f));
      gui::same_line();
      dialogue_box.cursor_position.y = 0;
      dialogue_box.indent = gui::get_window_size().y - 20.f;
    }
  );

  gui::pop_style_var(3);
  gui::pop_style_color(2);
  #undef gui
}

void update() {
  if (!is_in_dialogue && fan::physics::is_on_sensor(pile.player.body, vendor_buy_sensor) && pile.loco.is_key_clicked(fan::key_e)) {
    dialogue_task = dialogue();
    is_in_dialogue = true;
  }
  
  if (is_in_dialogue) {
    render_dialogue();
  }

  if (!pile.is_map_changing && fan::physics::is_on_sensor(pile.player.body, player_sensor_door)) {
    pile.is_map_changing = true;

    fan::vec3 fadein_color = pile.renderer.get_compiled("stage_forest")->lighting.ambient;
    
    pile.map_transition_task = pile.stage_loader.change_stage<stage_forest_t>(
      pile.loco.get_lighting(),
      pile.fadeout_target_color,
      fadein_color,
      stage_common.stage_id,
      pile.current_stage
    );
  }

  if (pile.is_map_changing && !pile.map_transition_task.valid()) {
    pile.is_map_changing = false;
  }

  pile.renderer.update(main_map_id, pile.player.body.get_position());
  pile.step();
}

tilemap_loader_t::id_t main_map_id;
fan::vec2 old_light_size;
fan::physics::body_id_t player_sensor_door;
fan::graphics::sprite_t vendor;
fan::graphics::image_t vendor_image;
fan::physics::body_id_t vendor_buy_sensor;

static inline std::string vendor_dialogue[] = {
  "Hello there!",
  "Feel free to browse the shop.",
  "I also buy junk.",
};

int current_answer = 0;

fan::graphics::gui::dialogue_box_t dialogue_box;
fan::event::task_t dialogue_task;
bool is_in_dialogue = false;

static inline std::string lore_chapter1_answers[] = {
  "Yes, gladly!",
  "No."
};