struct entity_t {
  static inline constexpr fan::vec2 draw_offset {0, -38};
  static inline constexpr f32_t aabb_scale = 0.19f;
  static constexpr fan::vec2 trigger_distance = {300, 50};
  static constexpr fan::vec2 closeup_distance = {120, 50};

  static bool attack_cb(fan::graphics::physics::character2d_t& c) {
    fan::vec2 d = c.ai_behavior.get_target_distance(&c);
    if (c.attack_state.try_attack(d)) {
      if (c.is_animation_finished()) {
        c.attack_state.end_attack();
      }
      else {
        auto* target = c.ai_behavior.target;
        if (!target) {
          return false;
        }
        if (target != &pile->player.body) {
          return false;
        }
        pile->player.on_hit(
          (target->get_position() - c.get_position()).normalized()
        );
        return true;
      }
    }
    return false;
  }


  entity_t(const fan::vec3& pos = 0) {
    body = fan::graphics::physics::character2d_t::from_json({
      .json_path = "player/player.json",
      .aabb_scale = aabb_scale,
      .draw_offset_override = draw_offset,
      .attack_cb = attack_cb
    });
    body.accelerate_force /= 2;
    body.set_color(fan::color(1, 1 / 3.f, 1 / 3.f, 1));
    body.enable_ai_follow(&pile->player.body, trigger_distance, closeup_distance);
    body.setup_attack(0.5f, closeup_distance);
    body.navigation.auto_jump_obstacles = true;
    body.navigation.jump_lookahead_tiles = 1.5f;
    body.hit_response.knockback_force = 50.f;
    body.hit_response.stun_duration = 500;
  }
  void update() {
    auto& level = pile->get_level();
    fan::vec2 tile_size = pile->renderer.get_tile_size(level.main_map_id) * 2.f;
    body.update_ai(tile_size);
  }
  void on_hit(const fan::vec2& hit_direction) {
    body.take_hit(hit_direction);
  }

  fan::graphics::physics::character2d_t body;
};