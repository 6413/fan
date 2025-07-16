struct player_t {
  fan::vec2 velocity = 0;
  std::array<loco_t::image_t, 4> img_idle;
  std::array<loco_t::image_t, std::size(fan::movement_e::_strings)> img_movement;

  player_t() {
    img_movement.fill(gloco->default_texture);

    light = fan::graphics::light_t{ {
      .position = player.get_position(),
      .size = 200,
      .color = fan::colors::white,
      .flags = 3
    } };
  }

  void step() {
    light.set_position(fan::vec2(player.get_position()));
    fan::vec2 dir = animator.prev_dir;
    uint32_t flag = 0;
    light.set_flags(flag);
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
      .fixed_rotation = true,
      .collision_multiplier = fan::vec2(1, 1)
    },
  }}};
  loco_t::shape_t light;
  fan::graphics::animator_t animator;
};