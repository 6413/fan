void create_manual_collisions(std::vector<fan::physics::entity_t>& collisions, fan::algorithm::path_solver_t& path_solver) {
  for (auto& x : main_compiled_map.compiled_shapes) {
    for (auto& y : x) {
      for (auto& z : y) {
        if (z.image_name == "tile0" || z.image_name == "tile1" || z.image_name == "tile2") {
          collisions.push_back(pile.loco.physics_context.create_circle(
            fan::vec2(z.position) + fan::vec2(0, -z.size.y / 6),
            z.size.y / 2.f,
            fan::physics::body_type_e::static_body,
            fan::physics::shape_properties_t{ .friction = 0 }
          ));
          /* visual_collisions.push_back(fan::graphics::circle_t{ {
          .position = fan::vec3(fan::vec2(z.position)+ fan::vec2(0, -z.size.y / 6), 50000),
          .radius = z.size.y / 2.f,
          .color = fan::colors::red.set_alpha(0.5),
          .blending = true
          }});*/
          path_solver.add_collision(fan::vec3(fan::vec2(z.position) + fan::vec2(0, -z.size.y / 6), 50000));
        }
      }
    }
  }
}

void open(void* sod) {
  main_compiled_map = pile.renderer.compile("examples/games/forest game/forest.json");
  fan::vec2i render_size(16, 9);
  render_size /= 1.5;
  fte_loader_t::properties_t p;
  p.size = render_size;
  p.position = pile.player.player.get_position();
  main_map_id = pile.renderer.add(&main_compiled_map, p);

  pile.path_solver = fan::algorithm::path_solver_t(main_compiled_map.map_size * 2, main_compiled_map.tile_size * 2);
  create_manual_collisions(collisions, pile.path_solver);

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

  rect_dst = { {
  .position = 0,
  .size = main_compiled_map.tile_size / 4,
  .color = fan::colors::red.set_alpha(0.3),
  .blending = true
  }};
}

void close() {

}

void update() {
  ImGui::Begin("A");
  static bool v = 0;
  ImGui::ToggleButton("lightning", &v);
  ImGui::End();
  if (fan::physics::is_on_sensor(pile.player.player, player_sensor_door)) {
    if (pile.loco.lighting.ambient > -1) {
      pile.loco.lighting.ambient -= pile.loco.delta_time * 5;
    }
    else {
      if (main_map_id.iic() == false) {
        pile.renderer.erase(main_map_id);
        pile.stage_loader.erase_stage(this->stage_common.stage_id);
        pile.current_stage = pile.stage_loader.open_stage<stage_shop_t>().NRI;
        return;
      }
    }
  }
  else if (v) {
    pile.weather.lightning();
  }
  else {
    if (pile.loco.lighting.ambient < 1) {
      pile.loco.lighting.ambient += pile.loco.delta_time * 5;
    }
    else {
      pile.loco.lighting.ambient = 1;
    }
  }

  if (pile.loco.input_action.is_action_clicked("move_to_position") && !ImGui::IsAnyItemHovered()) {
    rect_path.clear();
    fan::vec2 dst = pile.loco.get_mouse_position(pile.loco.orthographic_camera.camera, pile.loco.orthographic_camera.viewport);
    pile.path_solver.set_dst(dst);
    rect_dst.set_position(fan::vec3(dst, 50000));
    pile.path_solver.init(pile.player.player.get_position());

    rect_path.reserve(pile.path_solver.path.size());
    for (const auto& p : pile.path_solver.path) {
      fan::vec2i pe = p;
      rect_path.push_back({ {
        .position = fan::vec3(pe * main_compiled_map.tile_size * 2, 50000),
        .size = main_compiled_map.tile_size / 4,
        .color = fan::colors::cyan.set_alpha(0.3),
        .blending = true
      } });
    }
  }
  if (rect_path.size() && pile.path_solver.current_position < rect_path.size()) {
    rect_path[pile.path_solver.current_position].set_color(fan::colors::green);
  }
  pile.step();
  pile.renderer.update(main_map_id, pile.player.player.get_position());
}

std::vector<fan::physics::entity_t> collisions;
std::vector<fan::graphics::rectangle_t> rect_path;
fan::graphics::rectangle_t rect_dst;
fan::physics::body_id_t player_sensor_door;

fte_loader_t::id_t main_map_id;
fte_loader_t::compiled_map_t main_compiled_map;