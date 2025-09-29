void create_manual_collisions(std::vector<fan::physics::entity_t>& collisions, fan::algorithm::path_solver_t& path_solver) {
  for (auto& x : main_compiled_map->compiled_shapes) {
    for (auto& y : x) {
      for (auto& z : y) {
        if (gloco->texture_pack[z.texture_pack_unique_id].name == "tile0" || gloco->texture_pack[z.texture_pack_unique_id].name == "tile1" || gloco->texture_pack[z.texture_pack_unique_id].name == "tile2") {
          collisions.push_back(pile.loco.physics_context.create_circle(
            fan::vec2(z.position) + fan::vec2(0, z.size.y / 2),
            z.size.y / 2.f,
            0,
            fan::physics::body_type_e::static_body,
            fan::physics::shape_properties_t{ .friction = 0 }
          ));
          path_solver.add_collision(fan::vec3(fan::vec2(z.position) + fan::vec2(0, z.size.y / 2), 50000));
        }
      }
    }
  }
}

void open(void* sod) {
  fan::time::timer t{ true };
  main_compiled_map = &pile.maps_compiled[stage_name];
  pile.is_map_changing = false;
  fan::vec2i render_size(16, 9);
  render_size /= 1.5;
  fte_loader_t::properties_t p;
  p.size = render_size;
  p.position = pile.player.body.get_position();
  main_map_id = pile.renderer.add(main_compiled_map, p);
  pile.path_solver = fan::algorithm::path_solver_t(main_compiled_map->map_size * 2, main_compiled_map->tile_size * 2);
  create_manual_collisions(collisions, pile.path_solver);

  player_sensor_door = pile.renderer.get_physics_body(main_map_id, "player_sensor_door");
  if (player_sensor_door.is_valid() == false) {
    fan::throw_error("sensor not found");
  }

  rect_dst = { {
    .position = 0,
    .size = main_compiled_map->tile_size / 4,
    .color = fan::colors::red.set_alpha(0.3),
    .blending = true
  }};

  object_key.sensor = pile.renderer.add_sensor_rectangle(main_map_id, "object_key");
  if (!object_key.sensor) {
    fan::throw_error("key not found");
  }
  static bool initialized = false;
  if (!initialized) {
    initialized = true;
    pile.loco.lighting.ambient = 0;
  }

  if (pile.stage_loader.previous_stage_name == stage_shop_t::stage_name) {
    pile.player.body.set_physics_position(player_sensor_door.get_physics_position() + fan::vec2(0, player_sensor_door.get_aabb_size().y*2.f));
    pile.loco.camera_set_position(pile.loco.orthographic_render_view.camera, pile.player.body.get_position());
  }
  else {
    pile.player.body.set_physics_position(fan::vec2(1019.59076, 400.117065));
    pile.loco.camera_set_position(pile.loco.orthographic_render_view.camera, pile.player.body.get_position());
  }
  fan::graphics::gui::printf("The map was in: {:.4} seconds.", t.seconds());
}

void close() {
  for (auto& i : collisions) {
    i.destroy();
  }
  collisions.clear();
}

void update() {
  using namespace fan::graphics;
  if (!pile.is_map_changing && pile.loco.lighting.is_near(fan::vec3(pile.fadeout_target_color))) {
    gloco->lighting.set_target(main_compiled_map->lighting.ambient);
  }

  gui::begin("A");
  static bool enable_lightning = 0;
  fan::graphics::gui::toggle_button("enable_lightning", &enable_lightning);
  gui::end();

  if (object_key.sensor) {
    if (fan::physics::is_on_sensor(pile.player.body, object_key.sensor)) {
      pile.renderer.erase_id(main_map_id, "object_key");
      object_key.sensor.destroy();
    }
  }

  if (!pile.is_map_changing && fan::physics::is_on_sensor(pile.player.body, player_sensor_door)) {
    pile.loco.lighting.set_target(fan::vec3(pile.fadeout_target_color));
    pile.is_map_changing = true;
  }
  else if (pile.is_map_changing && pile.loco.lighting.is_near_target()) {
    pile.renderer.erase(main_map_id);
    pile.stage_loader.erase_stage(this->stage_common.stage_id);
    pile.current_stage = pile.stage_loader.open_stage<stage_shop_t>().NRI;
    return;
  }
  else if (enable_lightning) {
    pile.weather.lightning();
  }

  if (fan::window::is_mouse_clicked(fan::mouse_right) && !gui::is_any_item_hovered()) {
    rect_path.clear();
    fan::vec2 dst = pile.loco.get_mouse_position(pile.loco.orthographic_render_view.camera, pile.loco.orthographic_render_view.viewport);
    pile.path_solver.set_dst(dst);
    rect_dst.set_position(fan::vec3(dst, 50000));
    pile.path_solver.init(pile.player.body.get_position());

    rect_path.reserve(pile.path_solver.path.size());
    for (const auto& p : pile.path_solver.path) {
      fan::vec2i pe = p;
      rect_path.push_back({ {
        .position = fan::vec3(pe * main_compiled_map->tile_size * 2, 50000),
        .size = main_compiled_map->tile_size / 4,
        .color = fan::colors::cyan.set_alpha(0.3),
        .blending = true
      } });
    }
  }
  if (rect_path.size() && pile.path_solver.current_position < rect_path.size()) {
    rect_path[pile.path_solver.current_position].set_color(fan::colors::green);
  }
  pile.step();
  pile.renderer.update(main_map_id, pile.player.body.get_position());
}

std::vector<fan::physics::entity_t> collisions;
std::vector<fan::graphics::rectangle_t> rect_path;
fan::graphics::rectangle_t rect_dst;
fan::physics::body_id_t player_sensor_door;

fte_loader_t::id_t main_map_id;
fte_loader_t::compiled_map_t* main_compiled_map=0;

equipable_t object_key;