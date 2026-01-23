struct player_t {
  player_t() {
    //body.set_dynamic();
    fan::graphics::physics::character_movement_preset_t::setup_default_controls(body);
    //light.set_dynamic();
  }

  void step() {
    //fan::vec2 vel = body.get_linear_velocity();
    //fan::print("velocity.y: ", vel.y, " mass: ", body.get_mass());
    light.set_position(fan::vec2(body.get_position()));
  }

  fan::graphics::physics::character2d_t body{ fan::graphics::physics::capsule_t{{
    .position = fan::vec3(fan::vec2(48, -4) * 64, 10),
    //.center0 = {0, -16.f},
    //.center1 = {0, 16.f},
    //.radius=4.f,
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
    .size = 500,
    .color = fan::colors::white/*,
    .flags = 3*/
  } };
};