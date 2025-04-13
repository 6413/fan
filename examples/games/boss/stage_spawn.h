void open(void* sod) {
  main_compiled_map = pile.renderer.compile("examples/games/boss/spawn.json");
  fan::vec2i render_size(16, 9);
  render_size /= 1.5;
  fte_loader_t::properties_t p;
  p.size = render_size;
  pile.player.player.set_position(fan::vec2{ 0, 0 });
  pile.player.player.set_physics_position(pile.player.player.get_position());
  p.position = pile.player.player.get_position();
  main_map_id = pile.renderer.add(&main_compiled_map, p);
  fan::vec2 dst = pile.player.player.get_position();
  pile.loco.camera_set_position(
    pile.loco.orthographic_camera.camera,
    dst
  );
}

void close() {

}

void update() {
  pile.renderer.update(main_map_id, pile.player.player.get_position());
  pile.step();
}

fan::physics::body_id_t player_sensor_door;

fte_loader_t::id_t main_map_id;
fte_loader_t::compiled_map_t main_compiled_map;