struct player_t {

  static inline constexpr fan::vec2 draw_offset{0.f, -42.5f / 1.6f};
  static inline constexpr f32_t aabb_scale = 0.17f / 2.f;

  player_t() {
    body.set_dynamic();

    body = fan::graphics::physics::character2d_t::from_json({
      .json_path = "player/player.json",
      .aabb_scale = aabb_scale,
    });
    body.set_physics_position(fan::vec2(48, -4) * 64);
    body.set_draw_offset(draw_offset);
    body.set_size(fan::vec2(8.f, 48.f) * 6.48179f);
    fan::print(body.get_size());
    fan::graphics::physics::character_movement_preset_t::setup_default_controls(body);
    body.set_jump_height(1.f * body.get_mass());
    body.movement_state.max_speed = 100.f;
        body.anim_controller.auto_update_animations = false;
    body.anim_controller.auto_flip_sprite = true;
    //light.set_dynamic();
  }

  void step() {
    body.update_animations();
    //fan::vec2 vel = body.get_linear_velocity();
    //fan::print("velocity.y: ", vel.y, " mass: ", body.get_mass());
    light.set_position(fan::vec2(body.get_position()));
  }

  fan::graphics::physics::character2d_t body;
  fan::graphics::light_t light{ {
    .size = 500,
    .color = fan::colors::white/*,
    .flags = 3*/
  } };
};