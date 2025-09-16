void open(void* sod) {
  main_compiled_map = pile.renderer.compile("race_track.fte");
  fan::vec2i render_size(16, 9);
  //render_size /= 0.01;
  render_size *= 4;
  fte_loader_t::properties_t p;
  p.size = render_size;
  p.position = pile.car.body.get_position();
  main_map_id = pile.renderer.add(&main_compiled_map, p);
  pile.engine.physics_context.set_gravity(0);
}

void close() {

}

void update() {
  pile.renderer.update(main_map_id, pile.car.body.get_position());
  pile.step();
}

fte_loader_t::id_t main_map_id;
fte_loader_t::compiled_map_t main_compiled_map;