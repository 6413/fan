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

  initialize_entities();
}

void close() {

}

void update() {
  pile.renderer.update(main_map_id, pile.player.body.get_position());
  pile.step();

  update_entities();

  static fan::time::timer sword_hit_timer{ (uint64_t).5e9, true };
  if (pile.player.sword.visual.collides(entities.back().body)) {
    if (sword_hit_timer) {
      fan::print("sword hit");
      f32_t knockback_force = .5f;
      f32_t x_direction = fan::math::sgn(entities.back().body.get_position().x - pile.player.body.get_position().x);
      fan::vec2 knockback = fan::vec2(x_direction * knockback_force, -knockback_force);
      entities.back().body.apply_linear_impulse_center(knockback);
      sword_hit_timer.restart();
    }
  }
}

std::vector<fan::physics::body_id_t> collisions;

fte_loader_t::id_t main_map_id;
fte_loader_t::compiled_map_t main_compiled_map;

struct entity_t {

  fan::graphics::physics::character2d_t body{ fan::graphics::physics::capsule_sprite_t{{
    .position = fan::vec3(fan::vec2(109, 123) * 64, 10),
    // collision radius,
    .center0 = {0.f, -24.f},
    .center1 = {0.f, 24.f},
    .size = 12,
    /*.color = fan::color::from_rgba(0x715a5eff),*/
    .body_type = fan::physics::body_type_e::dynamic_body,
    //.mass_data{.mass = 0.01f},
    .shape_properties{
      .friction = 0.6f, 
      .density = 0.1f, 
      .fixed_rotation = true,
      .contact_events = true,
    },
  }}};

  std::function<void()> update_cb;
};

// entities

std::vector<entity_t> entities;

void initialize_entities() {
  fan::physics::body_id_t spawn_point_entity0 = pile.renderer.get_physics_body(main_map_id, "spawn_entity0");
  if (!spawn_point_entity0) {
    fan::throw_error("spawn_point_entity0 not found");
  }

  entities.resize(1);

  entities.back().body.set_physics_position(spawn_point_entity0.get_physics_position() - fan::vec2(0, 256));
  entities.back().update_cb = [&] {
    static fan::time::timer left_right_timer{ (uint64_t)3e9, true };
    static f32_t movement_speed = 10.0f;
    static bool left = true;
    if (left_right_timer) {
      left = !left;
      left_right_timer.restart();
    }
    auto& body = entities.back().body;
    // fan::print(fan::vec2(left ? -movement_speed : movement_speed, 0), pile.entities.back().body.get_linear_velocity());
    if (std::abs(body.get_linear_velocity().x) < 100.f) {
      body.apply_force_center(fan::vec2(left ? -movement_speed : movement_speed, 0));
    }
  };
}

void update_entities() {
  for (auto& entity : entities) {
    if (entity.update_cb) {
      entity.update_cb();
    }
  }

  if (fan::physics::is_colliding(pile.player.body, entities.back().body)) {
    pile.player.body.set_physics_position(pile.player.player_spawn);
  }
}