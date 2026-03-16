void open(void* sod) {
  main_map_id = pile.renderer.open_map(main_compiled_map, "sample_level.fte", {
    .position = pile.player.body.get_position(),
    .size = fan::vec2i(16, 9),
  });

  for (auto& y : pile.renderer.map_list[main_map_id].compiled_map->compiled_shapes)
    for (auto& x : y)
      for (auto& tile : x)
        collisions.emplace_back(pile.engine.physics_context.create_box(tile.position, tile.size, 0, fan::physics::body_type_e::static_body, {}));
}

void close() {}

void update() {
  pile.renderer.update(main_map_id, pile.player.body.get_position());
  pile.step();
}

tilemap_loader_t::id_t main_map_id;
tilemap_loader_t::compiled_map_t main_compiled_map;
std::vector<fan::physics::entity_t> collisions;