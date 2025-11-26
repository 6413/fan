//TODO use collision mask for player and entities
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
    body = fan::graphics::physics::character2d_t::from_json({
      .json_path = "player.json",
      .aabb_scale = aabb_scale,
      .draw_offset_override = draw_offset
    });
    body.enable_default_movement();
    // jump changes angle visually
    body.sync_visual_angle(false);
    body.hit_response.knockback_force = 10.f;

    mouse_click_handle = pile->engine.on_mouse_click(fan::mouse_left, [this](const auto& bdata) {
      task_attack = attack();
    });
  }

  fan::event::task_t jump() {
    jump_cancelled = false;
    body.set_rotation_point(-body.get_draw_offset());
    f32_t a = 0;
    while (a < fan::math::two_pi && !jump_cancelled) {
      a += rotation_speed;
      body.set_angle(fan::vec3(0, 0, a * body.get_image_sign().x));
      co_await fan::co_sleep(task_tick);
    }
    body.set_angle(0.f); 
  }
  fan::event::task_t attack() {
    co_await fan::graphics::animation_frame_awaiter(&body, "attack0", attack_hitbox_frame);
    auto points = get_hitbox_points(fan::math::sgn(body.get_tc_size().x));
    fan::physics::entity_t hitbox = pile->engine.physics_context.create_polygon(
      get_center(),
      0.0f,
      points.data(),
      points.size(),
      fan::physics::body_type_e::static_body,
      { .is_sensor = true }
    );

    if (hitbox.test_overlap(pile->entity.body)) {
      pile->entity.on_hit((pile->entity.body.get_position() - body.get_position()).normalized());
    }

    hitbox.destroy();
  }

  void step() {
    if (body.is_on_ground()) {
      jump_cancelled = true;
    }
    if (fan::window::is_action_clicked("move_up")) {
      //body.set_linear_velocity(fan::vec2(body.get_linear_velocity().x, -100.f));
      task_jump = jump();
    }
    body.update_animations();
  }

  fan::vec2 get_center() const {
    return body.get_position() - draw_offset;
  }
  fan::vec2 get_physics_pos() {
    return body.get_physics_position();
  }

  void on_hit(const fan::vec2& hit_direction) {
    body.take_hit(hit_direction);
  }

  fan::graphics::physics::character2d_t body;
  fan::graphics::engine_t::buttons_handle_t mouse_click_handle;
  fan::event::task_t task_attack;
  fan::event::task_t task_jump;
  bool jump_cancelled = false;
};