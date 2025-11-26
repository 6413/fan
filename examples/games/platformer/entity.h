struct entity_t {
  static inline constexpr fan::vec2 draw_offset {0, -18};
  static inline constexpr f32_t aabb_scale = 0.19f;
  static constexpr fan::vec2 trigger_distance = {600, 50};
  static constexpr fan::vec2 closeup_distance = {120, 50};
  static inline constexpr int attack_hitbox_frame = 4;
  static inline constexpr int attack_hitbox_frame2 = 8;

  bool attack_cb(fan::graphics::physics::character2d_t& c) {
    if (c.animation_on("attack0", {attack_hitbox_frame, attack_hitbox_frame2})) {
      auto* target = c.ai_behavior.target;
      pile->player.on_hit(
        &body,
        (target->get_position() - c.get_position()).normalized()
      );
    }
    if (!c.attack_state.cooldown_timer) {
      return false;
    }
    bool cond = c.get_current_animation_frame() == 0;
    if (cond) {
      c.attack_state.cooldown_timer.restart();
    }
    return cond;
  }

  entity_t(const fan::vec3& pos = 0) {
    body = fan::graphics::physics::character2d_t::from_json({
      .json_path = "skeleton.json",
      .aabb_scale = aabb_scale,
      .draw_offset_override = draw_offset,
      .attack_cb = [this](auto& c) { return attack_cb(c); }
    });
    body.accelerate_force /= 2.1f;
    body.set_size(body.get_size());
    body.set_color(fan::color(1, 1 / 3.f, 1 / 3.f, 1));
    body.enable_ai_follow(&pile->player.body, trigger_distance, closeup_distance);
    body.setup_attack(2.f, closeup_distance);
    body.navigation.auto_jump_obstacles = true;
    body.navigation.jump_lookahead_tiles = 1.5f;
    body.attack_state.knockback_force = 5.f;
  }
  void update() {
    auto& level = pile->get_level();
    fan::vec2 tile_size = pile->renderer.get_tile_size(level.main_map_id) * 2.f;
    body.update_ai(tile_size);
  }
  void on_hit(fan::graphics::physics::character2d_t* source, const fan::vec2& hit_direction) {
    body.take_hit(source, hit_direction);
  }

  fan::graphics::physics::character2d_t body;
};