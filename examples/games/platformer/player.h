struct player_t {

  player_t() {
    body = fan::graphics::physics::character_capsule({
      //.size = 12,
      .center0 = {0.f, -24.f},
      .center1 = {0.f, 24.f},
      .radius = 12,
      .body_type = fan::physics::body_type_e::dynamic_body,
    },
    {
      .friction = 0.6f,
      .density = 0.1f,
      .fixed_rotation = true
    });
    light.set_position(body.get_position());
    f32_t mass = body.get_mass();
    body.jump_impulse = 0.013 * mass;
    body.force = mass /10.0;
    body.max_speed = 0.0001;
    body.wall_jump.push_away_force = 0.013 * mass;
    body.wall_jump.slide_speed = 200;
    
  }

  void step() {
    body.process_movement(fan::graphics::physics::character2d_t::movement_e::side_view);
    light.set_position(fan::vec2(body.get_position()));
  }

  fan::graphics::physics::character2d_t body;
  fan::graphics::light_t light{ {
    .size = 200,
    .color = fan::colors::white/*,
    .flags = 3*/
  } };
};