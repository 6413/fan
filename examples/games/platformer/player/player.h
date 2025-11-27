struct player_t {
  static inline constexpr fan::vec2 draw_offset{ 0.f, -38.f };
  static inline constexpr f32_t aabb_scale = 0.19f;
  static inline constexpr f32_t task_tick = 1000.f / 60.f;
  static inline constexpr f32_t rotation_speed = fan::math::two_pi / 60.f * 2.f;
  static inline constexpr int attack_hitbox_frame = 4;
  static inline constexpr f32_t sword_length = 100.f;

  static inline constexpr std::array<fan::vec2, 3> get_hitbox_points(f32_t direction) {
    return {{
      {sword_length * direction, 0.0f},
      {0.0f, -10.0f},
      {0.0f, 10.0f}
      }};
  }
  player_t() {
    body.stun = false; // set before loading
    body = fan::graphics::physics::character2d_t::from_json({
      .json_path = "player/player.json",
      .aabb_scale = aabb_scale,
      .draw_offset_override = draw_offset
    });
    body.jump_impulse = 75.f;
    body.enable_default_movement();
    body.sync_visual_angle(false);
    body.attack_state.knockback_force = 20.f;
    body.attack_state.damage = 10.f;
    body.attack_state.cooldown_duration = 0.1e9;
    body.attack_state.cooldown_timer = fan::time::timer(body.attack_state.cooldown_duration, true);
    body.attack_state.on_attack_start = [this]() {
      hit_enemies.clear();
      hitbox_spawned = false;
    };
    body.attack_state.on_attack_end = [this]() {
      if (attack_hitbox.is_valid()) {
        attack_hitbox.destroy();
      }
      hitbox_spawned = false;
    };
    mouse_click_handle = pile->engine.on_mouse_click(fan::mouse_left, [this](const auto& bdata) {
      body.cancel_animations();
      if (body.attack_state.try_attack(fan::vec2(0))) {
      }
    });
  }
  void spawn_hitbox() {
    if (attack_hitbox) {
      attack_hitbox.destroy();
    }
    auto points = get_hitbox_points(fan::math::sgn(body.get_tc_size().x));
    attack_hitbox = pile->engine.physics_context.create_polygon(
      get_center(),
      0.0f,
      points.data(),
      points.size(),
      fan::physics::body_type_e::static_body,
      { .is_sensor = true }
    );
    hitbox_spawned = true;
  }
  fan::event::task_t jump() {
    jump_cancelled = false;
    body.set_rotation_point(-body.get_draw_offset());
    f32_t start_time = fan::time::now();
    f32_t duration = 1.0e9 / 2.0f;
    while (!jump_cancelled) {
      f32_t elapsed = fan::time::now() - start_time;
      if (elapsed >= duration) {
        break;
      }
      f32_t progress = elapsed / duration;
      f32_t a = progress * fan::math::two_pi;
      body.set_angle(fan::vec3(0, 0, a * body.get_image_sign().x));
      co_await fan::co_sleep(task_tick);
    }
    body.set_angle(0.f); 
  }
  void step() {
    if (body.is_on_ground()) {
      jump_cancelled = true;
    }
    if (fan::window::is_action_clicked("move_up")) {
      //body.set_linear_velocity(fan::vec2(body.get_linear_velocity().x, -100.f));
      task_jump = jump();
    }
    if (body.attack_state.is_attacking && !hitbox_spawned) {
      if (body.animation_on("attack0", attack_hitbox_frame)) {
        spawn_hitbox();
      }
    }
    if (hitbox_spawned && attack_hitbox.is_valid()) {
      for (auto& enemy : pile->entity) {
        if (hit_enemies.find(&enemy) == hit_enemies.end()) {
          if (attack_hitbox.test_overlap(enemy.body)) {
            enemy.on_hit(&body, (enemy.body.get_position() - body.get_position()).normalized());
            hit_enemies.insert(&enemy);
          }
        }
      }
    }
    body.update_animations();
  }
  fan::vec2 get_center() const {
    return body.get_position() - draw_offset;
  }
  fan::vec2 get_physics_pos() {
    return body.get_physics_position();
  }
  void on_hit(fan::graphics::physics::character2d_t* source, const fan::vec2& hit_direction) {
    body.take_hit(source, hit_direction);
    if (body.health <= 0) {
      body.set_physics_position(pile->renderer.get_position(pile->get_level().main_map_id, "player_spawn"));
      body.health = body.max_health;
    }
  }

  fan::graphics::physics::character2d_t body;
  fan::graphics::engine_t::buttons_handle_t mouse_click_handle;
  fan::physics::entity_t attack_hitbox;
  std::unordered_set<void*> hit_enemies;
  bool hitbox_spawned = false;
  fan::event::task_t task_jump;
  bool jump_cancelled = false;
};