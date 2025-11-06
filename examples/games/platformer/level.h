void open(void* sod) {
  main_compiled_map = pile.renderer.compile("sample_level.fte");
  fan::vec2i render_size(16, 9);
  //render_size /= 0.01;
  fte_loader_t::properties_t p;
  p.size = render_size;
  p.position = pile.player.body.get_position();
  main_map_id = pile.renderer.add(&main_compiled_map, p);

  // Generate collisions for every tile in the map
  for (auto& y : pile.renderer.map_list[main_map_id].compiled_map->compiled_shapes) {
    for (auto& x : y) {
      for (auto& tile : x) { // Depth
        collisions.emplace_back(pile.engine.physics_context.create_box(tile.position, tile.size, 0, fan::physics::body_type_e::static_body, {}));
      }
    }
  }

  // Set player spawn
  pile.player.body.set_physics_position(pile.renderer.get_position(main_map_id, "player_spawn"));
}

void close() {

}

void update() {
  static bool pause = false;
  if (fan::window::is_key_pressed(fan::key_e)) {
    pause = !pause;
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