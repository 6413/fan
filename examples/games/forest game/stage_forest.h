void create_manual_collisions(std::vector<fan::physics::entity_t>& collisions, fan::algorithm::path_solver_t& path_solver) {
  for (auto& x : getp().compiled_map0.compiled_shapes) {
    for (auto& y : x) {
      for (auto& z : y) {
        if (z.image_name == "tile0" || z.image_name == "tile1" || z.image_name == "tile2") {
          collisions.push_back(getp().loco.physics_context.create_circle(
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
  pile_t& pile_r = getp();

  pile_r.path_solver = fan::algorithm::path_solver_t(pile_r.compiled_map0.map_size*2, pile_r.compiled_map0.tile_size*2);
  create_manual_collisions(collisions, pile_r.path_solver);

  for (auto& i : pile_r.renderer.map_list[pile_r.map_id0].physics_entities) {
    if (i.id == "npc0_door") {
      std::visit([&]<typename T>(T& entity) { 
        if constexpr (std::is_same_v<T, fan::graphics::physics_shapes::rectangle_t>) {
          pile_r.npc0_door_sensor = entity;
        }
      }, i.visual);
      break;
    }
  }

  if (pile_r.npc0_door_sensor.is_valid() == false) {
    fan::throw_error("sensor not found");
  }
}

void close() {

}

void window_resize() {

}

void update() {
  pile_t& pile_r = getp();


  if (pile_r.loco.physics_context.is_on_sensor(pile_r.player.player, pile_r.npc0_door_sensor)) {
    if (pile_r.loco.lighting.ambient > -1) {
      pile_r.loco.lighting.ambient -= pile_r.loco.delta_time * 5;
    }
    else {
      if (pile_r.map_id0.iic() == false) {
        pile_r.renderer.erase(pile_r.map_id0);
        pile_r.compiled_map0 = pile_r.renderer.compile("examples/games/forest game/shop/shop.json");
        fan::vec2i render_size(16, 9);
        render_size /= 1.5;
        fte_loader_t::properties_t p;
        p.size = render_size;
        pile_r.player.player.set_position(fan::vec2{ 320.384949, 382.723236 });
        pile_r.player.player.set_physics_position(pile_r.player.player.get_position());
        p.position = pile_r.player.player.get_position();
        pile_r.map_id0 = pile_r.renderer.add(&pile_r.compiled_map0, p);
        fan::vec2 dst = pile_r.player.player.get_position();
        pile_r.loco.camera_set_position(
          pile_r.loco.orthographic_camera.camera,
          dst
        );
        pile_r.loco.lighting.ambient = -1;
        // close stage here
      }
    }
  }

  if (pile_r.loco.input_action.is_action_clicked("move_to_position") && !ImGui::IsAnyItemHovered()) {
    rect_path.clear();
    fan::vec2 dst = pile_r.loco.get_mouse_position(pile_r.loco.orthographic_camera.camera, pile_r.loco.orthographic_camera.viewport);
    pile_r.path_solver.set_dst(dst);
    rect_dst.set_position(fan::vec3(dst, 50000));
    pile_r.path_solver.init(pile_r.player.player.get_position());

    rect_path.reserve(pile_r.path_solver.path.size());
    for (const auto& p : pile_r.path_solver.path) {
      fan::vec2i pe = p;
      rect_path.push_back({ {
        .position = fan::vec3(pe * pile_r.compiled_map0.tile_size * 2, 50000),
        .size = pile_r.compiled_map0.tile_size / 4,
        .color = fan::colors::cyan.set_alpha(0.3),
        .blending = true
      } });
    }
  }
  if (rect_path.size() && pile_r.path_solver.current_position < rect_path.size()) {
    rect_path[pile_r.path_solver.current_position].set_color(fan::colors::green);
  }
}

std::vector<fan::physics::entity_t> collisions;
std::vector<fan::graphics::rectangle_t> rect_path;
fan::graphics::rectangle_t rect_dst{ {
  .position = 0,
  .size = getp().compiled_map0.tile_size / 4,
  .color = fan::colors::red.set_alpha(0.3),
  .blending = true
}};