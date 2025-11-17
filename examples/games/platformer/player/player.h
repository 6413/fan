struct player_t {
  player_t() {
    body = fan::graphics::physics::character2d_t::from_json({
      .json_path = "player.json",
      .aabb_scale = 0.19f,
      .draw_offset_override = fan::vec2(0, -38)
    });
  }

  void step() {
    body.update_animations();
  }

  fan::graphics::physics::character2d_t body;
};