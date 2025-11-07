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

    sword.visual.set_image(sword.image);
    sword.visual.set_size(0);
  }

  void step() {
    body.process_movement(fan::graphics::physics::character2d_t::movement_e::side_view);
    light.set_position(fan::vec2(body.get_position()));

    if (fan::window::is_mouse_clicked() && !sword.is_visible) {
      sword.toggle_render();
    }
    if (sword.is_visible) {
      fan::vec2 size = fan::vec2(10, 32);
      sword.visual.set_size(size);
      sword.visual.set_rotation_point(fan::vec2(-size.x, size.y));
      fan::vec2 pos = body.get_position() + fan::vec2(size.x, 0);
      pos += sword.timer.seconds() * 50.f;
      pos.y += sword.timer.seconds() * 50.f;
      //pos.x += size.x*3.f;
      pos.y -= size.y*1.5f;
      fan::graphics::aabb(sword.visual);
      sword.visual.set_position(pos);

      f32_t cam_pos_x = gloco->camera_get_position(gloco->orthographic_render_view.camera).x;
      f32_t body_pos_x = body.get_position().x;
      f32_t diff = cam_pos_x - body_pos_x;
      diff += fan::window::get_size().x / 2.f;

      f32_t sign = fan::math::sgn(fan::window::get_mouse_position().x - diff);

      sword.visual.set_angle(fan::vec3(0, 0, sign * sword.timer.seconds()*15.f));
      f32_t hit_time_s = 0.2f;
      if (sword.timer.seconds() > hit_time_s) {
        sword.visual.set_size(0);
        sword.is_visible = false;
      }
    }
  }

  static constexpr fan::vec2 player_spawn = fan::vec2(109, 123) * 64;

  struct object_t {
    fan::graphics::image_t image;
    fan::graphics::sprite_t visual;
    fan::time::timer timer;
    bool is_visible = false;

    void toggle_render() {
      is_visible = true;
      timer.start();
    }
  };

  object_t sword;

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
  fan::graphics::shape_t light;
};