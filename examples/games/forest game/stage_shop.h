void open(void* sod) {
  fan::time::timer t{ true };
  fan::vec2i render_size(16, 9);
  render_size /= 1.5;
  fte_loader_t::properties_t p;
  p.size = render_size;
  pile.player.body.set_position(fan::vec2{ 320.384949, 382.723236 });
  pile.player.body.set_physics_position(pile.player.body.get_position());
  p.position = pile.player.body.get_position();
  main_map_id = pile.renderer.add(&pile.maps_compiled[stage_name], p);
  fan::vec2 dst = pile.player.body.get_position();
  pile.loco.camera_set_position(
    pile.loco.orthographic_render_view.camera,
    dst
  );

  player_sensor_door = pile.renderer.add_sensor_rectangle(main_map_id, "player_sensor_door");

  if (!player_sensor_door) {
    fan::throw_error("sensor not found");
  }
  pile.is_map_changing = false;
  old_light_size = pile.player.light.get_size();
  pile.player.light.set_size(0);
  fan::graphics::gui::printf("The map was in: {:.4} seconds.", t.seconds());

  fan::graphics::image_load_properties_t lp;
  lp.min_filter = lp.mag_filter = fan::graphics::image_filter::nearest;
  vendor_image = pile.loco.image_load("npc/vendor/vendor.png", lp);
  fan::vec2 vendor_pos = player_sensor_door.get_position();

  auto& map = *pile.renderer.map_list[main_map_id].compiled_map;
  vendor_pos.y += -(map.tile_size.y * 2.f * map.map_size.y) + map.tile_size.y * 2.f * 3.7f;
  vendor = fan::graphics::sprite_t{ {
    .position = fan::vec3(vendor_pos, fan::graphics::get_depth_from_y(vendor_pos, 64.f)-1.f), /*hardcoded tile_size*/
    .size = fan::vec2(8, 16),
    .image = vendor_image
  } };
  fan::vec2 sensor_size = vendor.get_size().max() * 1.2f;
  vendor_buy_sensor = fan::physics::create_sensor_rectangle(vendor.get_position() + fan::vec2(0, sensor_size.y * 2.f), sensor_size);
}

void close() {
  pile.player.light.set_size(old_light_size);
}

fan::event::task_t dialogue() {
  for (int i = 0; i < std::size(vendor_dialogue); ++i) {
    co_await dialogue_box.text_delayed(vendor_dialogue[i]);

    co_await dialogue_box.wait_user_input();
  }
  //while (dialogue_box.get_button_choice() == -1) {
  //  dialogue_box.button(lore_chapter1_answers[0], fan::vec2(0.8, 0.1), fan::vec2(128, 32));
  //  for (int i = 1; i < std::size(lore_chapter1_answers); ++i) {
  //    dialogue_box.button(lore_chapter1_answers[i], fan::vec2(0.8, 0.4), fan::vec2(128, 32));
  //  }
  //  co_await dialogue_box.wait_user_input();
  //}
  //if (dialogue_box.get_button_choice() == 0) {
  //  guide_reputation += 0.1;
  //}
  //else {
  //  guide_reputation = 0.2;
  //}
  is_in_dialogue = false;
}

void render_dialogue() {
  using namespace fan::graphics;
  f32_t font_size = 24.f;
  fan::vec2 window_size = gui::get_window_size();
  window_size.x /= 1.2;
  window_size.y /= 5;

  gui::push_style_color(gui::col_window_bg, fan::colors::black.set_alpha(0.60f));

  gui::push_style_var(gui::style_var_window_border_size, 1.f);
  gui::push_style_var(gui::style_var_window_rounding, 8.f);
  gui::push_style_var(gui::style_var_window_padding, fan::vec2(20, 20.f));

  dialogue_box.font_size = font_size * 2.5;
  dialogue_box.render(
    "Dialogue box",
    gui::get_font(dialogue_box.font_size),
    window_size,
    gui::get_window_size().x / 2,
    32
  );

  gui::pop_style_var(3);
  gui::pop_style_color();
}


void update() {

  if (fan::physics::is_on_sensor(pile.player.body, vendor_buy_sensor) && fan::window::is_key_pressed(fan::key_e)) {
    dialogue_task = dialogue();
    is_in_dialogue = true;
  }
  if (is_in_dialogue) {
    render_dialogue();
  }

  if (!pile.is_map_changing && pile.loco.lighting.is_near(fan::vec3(pile.fadeout_target_color))) {
    gloco->lighting.set_target(pile.maps_compiled[stage_name].lighting.ambient);
  }

  if (!pile.is_map_changing && fan::physics::is_on_sensor(pile.player.body, player_sensor_door)) {
    pile.loco.lighting.set_target(fan::vec3(pile.fadeout_target_color));
    pile.is_map_changing = true;
  }
  else if (pile.is_map_changing && pile.loco.lighting.is_near_target()) {
    pile.loco.lighting.set_target(fan::vec3(pile.fadeout_target_color));
    pile.renderer.erase(main_map_id);
    pile.stage_loader.erase_stage(this->stage_common.stage_id);
    auto node_id = pile.stage_loader.open_stage<stage_forest_t>();
    pile.current_stage = node_id.NRI;
    return;
  }


  pile.renderer.update(main_map_id, pile.player.body.get_position());
  pile.step();
}



fte_loader_t::id_t main_map_id;

fan::vec2 old_light_size;
fan::physics::body_id_t player_sensor_door;

fan::graphics::sprite_t vendor;
fan::graphics::image_t vendor_image;
fan::physics::body_id_t vendor_buy_sensor;

static inline std::string vendor_dialogue[] = {
  "Hello there!|",
  "Feel free to browse the shop.|",
  "I also buy junk.|",
};
int current_answer = 0;

fan::graphics::gui::dialogue_box_t dialogue_box;
fan::event::task_t dialogue_task;
bool is_in_dialogue = false;

static inline std::string lore_chapter1_answers[] = {
  "Yes, gladly!",
  "No."
};