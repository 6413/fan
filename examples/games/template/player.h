struct player_t {
  player_t() {
    body.enable_default_movement();
  }

  void step() {
    light.set_position(fan::vec2(body.get_position()));
  }

  fan::graphics::physics::character2d_t body{ fan::graphics::physics::capsule_t{{
    .position = fan::vec3(fan::vec2(109, 123) * 64, 10),
    // collision radius,
    .center0 = {0.f, -24.f},
    .center1 = {0.f, 24.f},
    .radius = 12,
    /*.color = fan::color::from_rgba(0x715a5eff),*/
    .blending = true,
    .body_type = fan::physics::body_type_e::dynamic_body,
    //.mass_data{.mass = 0.01f},
    .shape_properties{
      .friction = 0.6f, 
      .fixed_rotation = true
    },
  }}};
  fan::graphics::light_t light{ {
    .position = body.get_position(),
    .size = 200,
    .color = fan::colors::white/*,
    .flags = 3*/
  } };
};