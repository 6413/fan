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
      for (auto& tile : x) { // depth
        collisions.emplace_back(pile.engine.physics_context.create_box(tile.position, tile.size, 0, fan::physics::body_type_e::static_body, {}));
      }
    }
  }

  fan::physics::body_id_t spawn_point_entity0 = pile.renderer.get_physics_body(main_map_id, "spawn_entity0");
  if (!spawn_point_entity0) {
    fan::throw_error("spawn_point_entity0 not found");
  }

  pile.entities.resize(1);

  pile.entities.back().body.set_physics_position(spawn_point_entity0.get_physics_position() - fan::vec2(0, 256));
  pile.entities.back().update_cb = [&] {
    static fan::time::timer left_right_timer{ (uint64_t)5e9, true };
    static f32_t movement_speed = 100.0f;
    static bool left = true;
    if (left_right_timer) {
      left = !left;
      left_right_timer.restart();
    }
    auto& body = pile.entities.back().body;
   // fan::print(fan::vec2(left ? -movement_speed : movement_speed, 0), pile.entities.back().body.get_linear_velocity());
    body.set_linear_velocity(fan::vec2(left ? -movement_speed : movement_speed, body.get_linear_velocity().y));
  };
}

void close() {

}

void update() {
  pile.renderer.update(main_map_id, pile.player.body.get_position());
  pile.step();

  if (fan::physics::is_colliding(pile.player.body, pile.entities.back().body)) {
    fan::print("is_colliding");
    pile.player.body.set_physics_position(pile.player.player_spawn);
  }
}

std::vector<fan::physics::body_id_t> collisions;

fte_loader_t::id_t main_map_id;
fte_loader_t::compiled_map_t main_compiled_map;