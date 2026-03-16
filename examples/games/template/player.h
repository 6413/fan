struct player_t {
  player_t() {
    body.enable_default_movement();
  }

  void step() {
    light.set_position(fan::vec2(body.get_position()));
  }

  fan::graphics::physics::character2d_t body = fan::graphics::physics::character_capsule({
    .position = fan::vec3(fan::vec2(109, 123) * 64, 10),
    .center0 = {0.f, -24.f},
    .center1 = {0.f, 24.f},
    .radius = 12,
    }, 
    {
      .friction = 0.6f, 
      .fixed_rotation = true
    }
  );
  fan::graphics::light_t light {
    body.get_position(), 
    200, 
    fan::colors::white
  };
};