struct player_t {

  player_t() {
    light = fan::graphics::light_t{ {
      .position = body.get_position(),
      .size = 200,
      .color = fan::colors::white,
    } };
    body.jump_impulse = 3;
    body.force = 15;
    body.max_speed = 270;
  }

  void step() {
    body.process_movement(fan::graphics::physics::character2d_t::movement_e::side_view);
    light.set_position(fan::vec2(body.get_position()));
  }

  static constexpr fan::vec2 player_spawn = fan::vec2(109, 123) * 64;

  fan::graphics::physics::character2d_t body{ fan::graphics::physics::capsule_t{{
    .position = fan::vec3(player_spawn, 10),
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
      .density = 0.1f, 
      .fixed_rotation = true,
      .contact_events = true,
    },
  }}};
  loco_t::shape_t light;
  fan::graphics::animator_t animator;
};