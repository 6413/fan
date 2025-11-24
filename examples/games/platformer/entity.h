struct entity_t{
  static inline constexpr fan::vec2 draw_offset{ 0.f, -38.f };
  static inline constexpr f32_t aabb_scale = 0.19f;
  entity_t(const fan::vec3& pos = 0) {
    body = fan::graphics::physics::character2d_t::from_json({
      .json_path = "player/player.json",
      .aabb_scale = aabb_scale,
      .draw_offset_override = draw_offset
    });
  }
  void update() {
    body.update_animations();
  }

  void on_hit(fan::vec2 hit_direction) {
    body.apply_linear_impulse_center(-hit_direction * 50.f);
  }

  fan::graphics::physics::character2d_t body;
};