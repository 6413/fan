static inline constexpr f32_t spike_height = 32.0f;
static inline constexpr f32_t base_half_width = 32.0f;

static inline constexpr std::array<fan::vec2, 3> get_spike_points() {
  return {{
    {0.0f, -spike_height},         // top
    {-base_half_width, spike_height},     // left
    { base_half_width, spike_height}      // right
  }};
}

void load_enemies() {
  for (auto* e : pile->entity) {
    e->destroy();
    delete e;
    e = nullptr;
  }

  // pointers can change
  pile->entity.resize(2);
  pile->entity[0] = new entity_t;
  pile->entity[0]->set_initial_position(pile->renderer.get_position(main_map_id, "enemy0_spawn"));
  pile->entity[1] = new entity_t;
  pile->entity[1]->set_initial_position(pile->renderer.get_position(main_map_id, "enemy1_spawn"));
}

void load_map() {
  main_compiled_map = pile->renderer.compile("sample_level.fte");
  fan::vec2i render_size(16, 9);
  //render_size /= 0.01;
  fte_loader_t::properties_t p;
  p.size = render_size;
  p.position = pile->player.body.get_position();
  main_map_id = pile->renderer.add(&main_compiled_map, p);

  pile->player.body.set_physics_position(pile->renderer.get_position(main_map_id, "player_spawn") + fan::vec2(300, -200));

  load_enemies();

  pile->renderer.iterate_visual(main_map_id, [&](fte_loader_t::tile_t& tile) {
    if (tile.id == "spikes") {
      auto points = get_spike_points();
      spike_sensors.emplace_back(pile->engine.physics_context.create_polygon(
        tile.position,
        0.0f,
        points.data(),
        points.size(),
        fan::physics::body_type_e::static_body,
        {.is_sensor = true}
      ));
    }
    else if (tile.mesh_property == fte_loader_t::fte_t::mesh_property_t::none) {
      tile_collisions.emplace_back(pile->engine.physics_context.create_rectangle(
        tile.position,
        tile.size,
        0.0f,
        fan::physics::body_type_e::static_body,
        {.fixed_rotation = true}
      ));
    }
  });

  //iterate_physics_entities
}

void open(void* sod) {

}

void close() {}

void reload_map() {
  for (auto& i : spike_sensors) {
    i.destroy();
  }
  spike_sensors.clear();
  pile->renderer.erase(main_map_id);
  load_map();
}

void update() {
  for (auto& spike : spike_sensors) {
    if (fan::physics::is_on_sensor(pile->player.body, spike)) {
      pile->player.respawn();
      load_enemies();
    }
    for (auto& enemy : pile->entity) {
      if (fan::physics::is_on_sensor(enemy->body, spike)) {
        enemy->destroy();
      }
    }
  }

  static bool pause = false;
  if (!pile->engine.render_console) {
    if (fan::window::is_key_pressed(fan::key_e)) {
      pause = !pause;
    }
    if (fan::window::is_key_pressed(fan::key_r)) {
      reload_map();
    }
  }
  pile->renderer.update(main_map_id, pile->player.body.get_position());
  if (pause) {
    return;
  }
  pile->step();
}

fte_loader_t::id_t main_map_id;
fte_loader_t::compiled_map_t main_compiled_map;

std::vector<fan::physics::entity_t> spike_sensors;
std::vector<fan::physics::entity_t> tile_collisions;