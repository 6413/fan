void create_manual_collisions(std::vector<fan::physics::entity_t>& collisions) {
  pile.renderer.iterate_tiles(main_map_id, [&](const auto& t) {
    const auto& name = gloco()->texture_pack[t.texture_pack_unique_id].name;
    if (!(name == "tile0" || name == "tile1" || name == "tile2")) {
      return;
    }
    collisions.push_back(pile.engine.get_physics_context().create_circle(
      fan::vec2(t.position).offset_y(t.size.y/2.f),
      t.size.y / 3.f,
      0,
      fan::physics::body_type_e::static_body,
      fan::physics::shape_properties_t{ .friction = 0 }
    ));
    pile.pathfinder.add_collision(collisions.back().get_position());
  });
}

void open(void* sod) {
  fan::time::timer t{ true };
  
  main_compiled_map = pile.renderer.get_compiled(stage_name);
  if (main_compiled_map == nullptr) {
    fan::throw_error("Failed to fetch compiled map! Did you spell the name wrong in compile()?");
  }

  main_map_id = pile.renderer.add(main_compiled_map, {
    .position = pile.player.body.get_position(),
    .size = fan::vec2(16, 9) / 1.5f,
    .depth_fn = tilemap_loader_t::default_depth_fn
  });
  
  pile.active_map_id = main_map_id;

  create_manual_collisions(collisions);

  player_sensor_door = pile.renderer.get_physics_body(main_map_id, "player_sensor_door");
  if (!player_sensor_door.is_valid()) {
    fan::throw_error("sensor not found");
  }

  rect_dst = {{
    .position = 0,
    .size = main_compiled_map->tile_size / 4,
    .color = fan::colors::red.set_alpha(0.3),
    .blending = true
  }};

  if (pile.stage_loader.previous_stage_name == stage_shop_t::stage_name) {
    pile.player.body.set_physics_position(player_sensor_door.get_physics_position() + fan::vec2(0, player_sensor_door.get_size().y * 2.f));
  }
  else {
    pile.player.body.set_physics_position(fan::vec2(1019.59076, 400.117065));
  }
  pile.engine.camera_set_position(pile.engine.orthographic_render_view.camera, pile.player.body.get_position());
  
  gui::print("The map was in: ", t.seconds(), " seconds.");

  auto& map = *pile.renderer.get_map_node(main_map_id).compiled_map;
  pile.path_grid_size = map.tile_size.x * 2.f;
  pile.pathfinder.init(map.map_size, false);

  static fan::graphics::image_t pet_img{"images/duck.webp", image_presets::pixel_art()};
  pile.pet.open(pile.player.body.get_position(), pet_img);
}

void close() {
  for (auto& i : collisions) {
    i.destroy();
  }
  collisions.clear();
  pile.renderer.erase(main_map_id);
}

void update() {
  pile.pet.step(pile.player.body.get_position(), pile.pathfinder, pile.path_grid_size, pile.engine.get_delta_time());
  gui::begin("A");
  static bool enable_lightning = 0;
  gui::toggle_button("enable_lightning", &enable_lightning);
  gui::end();

  if (object_key.sensor && fan::physics::is_on_sensor(pile.player.body, object_key.sensor)) {
    pile.renderer.erase_visual(main_map_id, "object_key");
    object_key.sensor.destroy();
  }

  if (!pile.is_map_changing && fan::physics::is_on_sensor(pile.player.body, player_sensor_door)) {
    pile.is_map_changing = true;
    
    fan::vec3 fadein_color = pile.renderer.get_compiled("stage_shop")->lighting.ambient;
    
    pile.map_transition_task = pile.stage_loader.change_stage<stage_shop_t>(
      pile.engine.get_lighting(),
      pile.fadeout_target_color,
      fadein_color,
      stage_common.stage_id,
      pile.current_stage
    );
  } else if (enable_lightning) {
    pile.weather.lightning();
  }
  
  if (pile.is_map_changing && !pile.map_transition_task.valid()) {
    pile.is_map_changing = false;
  }

  pile.step();
  pile.renderer.update(main_map_id, pile.player.body.get_position());
}

std::vector<fan::physics::entity_t> collisions;
std::vector<fan::graphics::rectangle_t> rect_path;
fan::graphics::rectangle_t rect_dst;
fan::physics::body_id_t player_sensor_door;
tilemap_loader_t::id_t main_map_id;
tilemap_loader_t::compiled_map_t* main_compiled_map = nullptr;
equipable_t object_key;