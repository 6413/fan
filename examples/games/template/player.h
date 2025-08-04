struct player_t {

  player_t() {
    light = fan::graphics::light_t{ {
      .position = player.get_position(),
      .size = 200,
      .color = fan::colors::white,
      .flags = 3
    } };
    player.impulse = 3;
    player.force = 15;
    player.max_speed = 270;
  }

  void step() {
    player.process_movement(fan::graphics::physics::character2d_t::movement_e::side_view);
    light.set_position(fan::vec2(player.get_position()));
  }

  fan::graphics::physics::character2d_t player{ fan::graphics::physics::capsule_t{{
    .position = fan::vec3(fan::vec2(109, 123) * 64, 10),
    // collision radius,
    .center0 = {0.f, -24.f},
    .center1 = {0.f, 24.f},
    .radius = 12,
    /*.color = fan::color::hex(0x715a5eff),*/
    .blending = true,
    .body_type = fan::physics::body_type_e::dynamic_body,
    //.mass_data{.mass = 0.01f},
    .shape_properties{
      .friction = 0.6f, 
      .density = 0.1f, 
      .fixed_rotation = true
    },
  }}};
  loco_t::shape_t light;
  fan::graphics::animator_t animator;
};