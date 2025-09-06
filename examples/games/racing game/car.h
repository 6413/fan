struct car_t {
  car_t() {

  }

  //
  fan::vec3 get_tire_position(const fan::vec2i& side_offset) const {
    fan::vec2 car_size = car.get_size();

    fan::vec3 local_offset = {
        static_cast<float>(side_offset.x) * car_size.x * 0.5f,
        static_cast<float>(side_offset.y) * car_size.y * 0.5f,
        0xffff - 0x101
    };

    return car.transform(local_offset);
  }

  void step() {
    f32_t dt = engine.delta_time;
    fan::vec2 input = engine.get_input_vector();
    // texture points up, so rotate to 0 angle
    fan::basis basis = car.get_basis();
    fan::vec2 forward = basis.forward;
    fan::vec2 right = basis.right;
    fan::vec2 vel = car.get_linear_velocity();

    // forward movement
    f32_t speed = 3.f;
    // (-input.y) sign conflict
    car.apply_force_center(forward * (-input.y) * speed);

    // tire grip
    fan::vec2 side = forward.perpendicular();
    f32_t lateral_speed = vel.dot(side);
    f32_t grip = 5.f;
    car.apply_force_center(-side * lateral_speed * grip * dt);

    f32_t drift_threshold = 70.0f;
    if (timer_drift_lines_add && fabs(lateral_speed) > drift_threshold) {
      
      fan::vec3 rear_left_pos  = get_tire_position({ -1, 1 });
      fan::vec3 rear_right_pos = get_tire_position({  1, 1 });
      
      // normalized
      f32_t drift_intensity = fmin(1.0f, (fabs(lateral_speed) - drift_threshold) / 100.0f);
      
      rear_left_trail.set_point(rear_left_pos, drift_intensity);
      rear_right_trail.set_point(rear_right_pos, drift_intensity);
      timer_drift_lines_add.restart();
    }
    
    f32_t steering = 0.5f;
    car.apply_angular_impulse(input.x * steering * dt);
    engine.camera_move_to(car);
  }

  fan::graphics::image_t car_image{ "images/car_red.png" };
  std::vector<fan::graphics::circle_t> drift_lines;
  fan::graphics::trail_t rear_left_trail;
  fan::graphics::trail_t rear_right_trail;
  fan::time::timer timer_drift_lines_add{ fan::time::timer(0.01e9, true) };

  fan::graphics::physics::character2d_t car{
    fan::graphics::physics::capsule_sprite_t{{
      .position = fan::vec3(1019.7828, 1618.1302, 0xffff - 0x100),
      .center0 = {0.f, -16.f},
      .center1 = {0.f, 16.f},
      .size = car_image.get_size().normalized() * 32.f,
      .angle = fan::math::pi / 2,
      .image = car_image,
      .body_type = fan::physics::body_type_e::dynamic_body,
      .shape_properties{
        .friction = 0.6f,
        .density = 0.1f,
        .fixed_rotation = false,
        .linear_damping = 1.f,
        .angular_damping = 10.f
      }
    }}
  };
};