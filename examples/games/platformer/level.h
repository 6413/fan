void load_map() {
  main_compiled_map = pile.renderer.compile("sample_level.fte");
  fan::vec2i render_size(16, 9);
  //render_size /= 0.01;
  fte_loader_t::properties_t p;
  p.size = render_size;
  p.position = pile.player.body.get_position();
  main_map_id = pile.renderer.add(&main_compiled_map, p);

  // Generate collisions for every tile in the map
  //for (auto& y : pile.renderer.map_list[main_map_id].compiled_map->compiled_shapes) {
  //  for (auto& x : y) {
  //    for (auto& tile : x) { // Depth
  //      if (tile.id.empty()) {
  //        collisions.emplace_back(pile.engine.physics_context.create_box(tile.position, tile.size, 0, fan::physics::body_type_e::static_body, {}));
  //      }
  //    }
  //  }
  //}

  // Set player spawn
  pile.player.body.set_physics_position(pile.renderer.get_position(main_map_id, "player_spawn"));

  sensor_spikes = pile.renderer.get_physics_body(main_map_id, "sensor_spikes");
}

void open(void* sod) {
  load_map();
}

void close() {
  for (auto& i : collisions) {
    i.erase();
  }
}

void reload_map() {
  for (auto& i : collisions) {
    i.erase();
  }
  collisions.clear();
  pile.renderer.erase(main_map_id);
  load_map();
}

void update() {
  if (fan::physics::is_on_sensor(pile.player.body, sensor_spikes)) {
    pile.player.body.set_physics_position(pile.renderer.get_position(main_map_id, "player_spawn"));
  }

  static bool pause = false;
  if (fan::window::is_key_pressed(fan::key_e)) {
    pause = !pause;
  }
  if (fan::window::is_key_pressed(fan::key_r)) {
    reload_map();
  }
  pile.renderer.update(main_map_id, pile.player.body.get_position());
  if (pause) {
    return;
  }
  pile.step();
}

fte_loader_t::id_t main_map_id;
fte_loader_t::compiled_map_t main_compiled_map;
std::vector<fan::physics::entity_t> collisions;

fan::physics::body_id_t sensor_spikes;