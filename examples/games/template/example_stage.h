void open(void* sod) {
  id = pile.renderer.open_map("sample_level.fte", {
    .position = pile.player.body.get_position(),
    .size = fan::vec2i(16, 9),
  });

  pile.renderer.iterate_tiles(id, [&](const auto& tile) {
    collisions.emplace_back(pile.get_physics_context().create_box(tile.position, tile.size, 0, fan::physics::body_type_e::static_body, {}));
  });
}

void close() {}

void update() {
  pile.renderer.update(id, pile.player.body.get_position());
  pile.update();
}

tilemap_renderer_t::id_t id;
std::vector<fan::physics::entity_t> collisions;