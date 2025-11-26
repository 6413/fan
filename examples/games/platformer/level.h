static inline constexpr f32_t spike_height = 32.0f;
static inline constexpr f32_t base_half_width = 32.0f;

static inline constexpr std::array<fan::vec2, 3> get_spike_points() {
  return {{
    {0.0f, -spike_height},         // top
    {-base_half_width, spike_height},     // left
    { base_half_width, spike_height}      // right
  }};
}

void load_map() {
  fan::graphics::physics::debug_draw(true);
  main_compiled_map = pile->renderer.compile("sample_level.fte");
  fan::vec2i render_size(16, 9);
  //render_size /= 0.01;
  fte_loader_t::properties_t p;
  p.size = render_size;
  p.position = pile->player.body.get_position();
  main_map_id = pile->renderer.add(&main_compiled_map, p);
  // Set player spawn
  pile->player.body.set_physics_position(pile->renderer.get_position(main_map_id, "player_spawn") + fan::vec2(300, -200));
  pile->entity.resize(2);
  pile->entity[0].body.set_physics_position(pile->renderer.get_position(main_map_id, "enemy0"));
  pile->entity[1].body.set_physics_position(pile->renderer.get_position(main_map_id, "enemy1"));

  std::vector<fte_loader_t::tile_t> tiles;
  if (!pile->renderer.get_visual_bodies(main_map_id, "sensor_spikes", &tiles)) {
    fan::throw_error("spikes not found");
  }

  spike_sensors.resize(tiles.size());

  for (auto [i, spike] : fan::enumerate(tiles)) {
    auto points = get_spike_points();
    spike_sensors[i] = pile->engine.physics_context.create_polygon(
      spike.position,
      0.0f,
      points.data(),
      points.size(),
      fan::physics::body_type_e::static_body,
      { .is_sensor = true }
    );
  }
}

void open(void* sod) {
  load_map();
}

void close() {

}

void reload_map() {
  pile->renderer.erase(main_map_id);
  load_map();
}

void update() {
  for (auto& spike : spike_sensors) {
    if (fan::physics::is_on_sensor(pile->player.body, spike)) {
      pile->player.body.set_physics_position(pile->renderer.get_position(main_map_id, "player_spawn"));
    }
  }
  for (auto& enemy : pile->entity) {
    for (auto& spike : spike_sensors) {
      if (fan::physics::is_on_sensor(enemy.body, spike)) {
        enemy.body.set_physics_position(pile->renderer.get_position(main_map_id, "player_spawn"));
      }
    }
  }

  static bool pause = false;
  if (fan::window::is_key_pressed(fan::key_e)) {
    pause = !pause;
  }
  if (fan::window::is_key_pressed(fan::key_r)) {
    reload_map();
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