void open(void* sod) {
  main_compiled_map = pile.renderer.compile("examples/games/forest game/shop/shop.json");
  fan::vec2i render_size(16, 9);
  render_size /= 1.5;
  fte_loader_t::properties_t p;
  p.size = render_size;
  pile.player.player.set_position(fan::vec2{ 320.384949, 382.723236 });
  pile.player.player.set_physics_position(pile.player.player.get_position());
  p.position = pile.player.player.get_position();
  main_map_id = pile.renderer.add(&main_compiled_map, p);
  fan::vec2 dst = pile.player.player.get_position();
  pile.loco.camera_set_position(
    pile.loco.orthographic_camera.camera,
    dst
  );
  pile.loco.lighting.ambient = -1;

  pile.renderer.iterate_physics_entities(main_map_id,
    [&]<typename T>(auto& entity, T& entity_visual) {
    if (entity.id == "player_sensor_door" &&
      std::is_same_v<T, fan::graphics::physics::rectangle_t>) {
      player_sensor_door = entity_visual;
    }
  });

  if (player_sensor_door.is_valid() == false) {
    fan::throw_error("sensor not found");
  }
}

void close() {

}

void update() {
  if (fan::physics::is_on_sensor(pile.player.player, player_sensor_door)) {
    if (pile.loco.lighting.ambient > -1) {
      pile.loco.lighting.ambient -= pile.loco.delta_time * 5;
    }
    else {
      if (main_map_id.iic() == false) {
        pile.renderer.erase(main_map_id);
        pile.stage_loader.erase_stage(this->stage_common.stage_id);
        auto node_id = pile.stage_loader.open_stage<stage_forest_t>();
        pile.current_stage = node_id.NRI;
        stage_forest_t* sf = (stage_forest_t*)pile.stage_loader.stage_list[node_id].stage;
        // why this doesnt work?
        pile.player.player.set_position((sf)->player_sensor_door.get_physics_position() * fan::physics::length_units_per_meter);
        return;
      }
    }
  }
  else {
    if (pile.loco.lighting.ambient < 1) {
      pile.loco.lighting.ambient += pile.loco.delta_time * 5;
    }
    else {
      pile.loco.lighting.ambient = 1;
    }
  }

  pile.renderer.update(main_map_id, pile.player.player.get_position());
  pile.step();
}

fan::physics::body_id_t player_sensor_door;

fte_loader_t::id_t main_map_id;
fte_loader_t::compiled_map_t main_compiled_map;