void load_map() {
  main_compiled_map = pile->renderer.compile("sample_level.fte");
  fan::vec2i render_size(16, 9);
  //render_size /= 0.01;
  fte_loader_t::properties_t p;
  p.size = render_size;
  p.position = pile->player.body.get_position();
  main_map_id = pile->renderer.add(&main_compiled_map, p);
  // Set player spawn
  pile->player.body.set_physics_position(pile->renderer.get_position(main_map_id, "player_spawn") + fan::vec2(300, -200));
  pile->entity.body.set_physics_position(pile->renderer.get_position(main_map_id, "player_spawn"));

  sensor_spikes = pile->renderer.get_physics_body(main_map_id, "sensor_spikes");
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
  if (fan::physics::is_on_sensor(pile->player.body, sensor_spikes)) {
    pile->player.body.set_physics_position(pile->renderer.get_position(main_map_id, "player_spawn"));
  }
  if (fan::physics::is_on_sensor(pile->entity.body, sensor_spikes)) {
    pile->entity.body.set_physics_position(pile->renderer.get_position(main_map_id, "player_spawn"));
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

fan::physics::body_id_t sensor_spikes;