struct car_t {
  static constexpr uint16_t car_draw_depth = 0xffff / 2;

  void open(
    const fan::vec3& initial_position = -0xffff, 
    const fan::color& color = fan::colors::white,
    const std::string &movement_keybinds_prefix = "",
    int forward = fan::key_w,
    int back = fan::key_s,
    int left = fan::key_a,
    int right =  fan::key_d
  )
  {
    this->movement_keybinds_prefix = movement_keybinds_prefix;
    if (initial_position != -0xffff) {
      body.set_position(initial_position);
      body.set_physics_position(initial_position);
    }
    if (!movement_keybinds_prefix.empty()) {
      fan::graphics::add_input_action(forward, movement_keybinds_prefix + "move_forward");
      fan::graphics::add_input_action(back, movement_keybinds_prefix + "move_back");
      fan::graphics::add_input_action(left, movement_keybinds_prefix + "move_left");
      fan::graphics::add_input_action(right, movement_keybinds_prefix + "move_right");
    }
    body.set_color(color);

    initialize_sensors();

    if (is_local) {
      piece_id = engine_sound.play(0, true);
    }
  }

  // initialize checkpoint sensors
  void initialize_sensors() {
    // dangerous if 'current_stage' is not racing_track_t
    auto& racing_track_stage_data = pile.stage_loader.get_stage_data<pile_t::racing_track_t>(pile.current_stage);

    {
      pile.renderer.iterate_physics_entities(racing_track_stage_data.main_map_id,
        [&]<typename T>(auto& entity, T & entity_visual) {
        if (entity.id == "goal" &&
          std::is_same_v<T, fan::graphics::physics::rectangle_t>) {
          goal_sensor = entity_visual;
        }
      });

      if (goal_sensor.is_valid() == false) {
        fan::throw_error("sensor not found");
      }
    }
    {
      pile.renderer.iterate_physics_entities(racing_track_stage_data.main_map_id,
        [&]<typename T>(auto& entity, T & entity_visual) {
        if (entity.id == "checkpoint_middle" &&
          std::is_same_v<T, fan::graphics::physics::rectangle_t>) {
          middle_checkpoint_sensor = entity_visual;
        }
      });

      if (middle_checkpoint_sensor.is_valid() == false) {
        fan::throw_error("sensor not found");
      }
    }
  }

  //
  fan::vec3 get_tire_position(const fan::vec2i& side_offset) const {
    fan::vec2 car_size = body.get_size();

    fan::vec3 local_offset = {
      static_cast<float>(side_offset.x) * car_size.x * 0.5f,
      static_cast<float>(side_offset.y) * car_size.y * 0.5f,
      0xffff - 0x101
    };

    f32_t z = body.get_position().z;
    return fan::vec3(fan::vec2(body.transform(local_offset)), z);
  }

  void handle_movement() {
    f32_t dt = engine.delta_time;
    fan::vec2 input = engine.get_input_vector(
      movement_keybinds_prefix + "move_forward",
      movement_keybinds_prefix + "move_back",
      movement_keybinds_prefix + "move_left",
      movement_keybinds_prefix + "move_right"
    );
    // texture points up, so rotate to 0 angle
    fan::basis basis = body.get_basis();
    fan::vec2 forward = basis.forward;
    fan::vec2 right = basis.right;
    fan::vec2 vel = body.get_linear_velocity();

    // forward movement
    f32_t speed = 2000.f * body.get_mass();
    // (-input.y) sign conflict
    body.apply_force_center(forward * (-input.y) * speed);

    // calculate using forward velocity
    f32_t velocity_volume = body.get_linear_velocity().abs().length() / 200.f;
    if (use_audio) {
      engine_sound.set_volume(std::min(0.2f, velocity_volume));
    }
    else {
      engine_sound.stop(piece_id);
    }

    // tire grip
    fan::vec2 side = forward.perpendicular();
    f32_t lateral_speed = vel.dot(side) * body.get_mass();
    f32_t grip = 0.01f;
    body.apply_force_center(-side * lateral_speed * grip * dt);

    f32_t drift_threshold = 60.0f * body.get_mass();
    if (timer_drift_lines_add && fabs(lateral_speed) > drift_threshold) {
      
      fan::vec3 rear_left_pos  = get_tire_position({ -1, 1 });
      fan::vec3 rear_right_pos = get_tire_position({  1, 1 });
      
      // normalized
      f32_t drift_intensity = fmin(1.0f, (fabs(lateral_speed) - drift_threshold) / 100.0f);
      
      rear_left_trail.set_point(rear_left_pos, drift_intensity);
      rear_right_trail.set_point(rear_right_pos, drift_intensity);
      timer_drift_lines_add.restart();
    }
    
    f32_t steering = 30.f * body.get_mass();
    if (input.y) {
      body.apply_angular_impulse(input.x * steering * dt * -input.y);
    }
  }

  void handle_laps() {
   // fan::print(fan::physics::is_on_sensor(body, goal_sensor));
    if (checkpoint_required) {
      if (fan::physics::is_on_sensor(body, middle_checkpoint_sensor)) {
        checkpoint_required = false;
      }
    }
    else {
      if (fan::physics::is_on_sensor(body, goal_sensor)) {
        checkpoint_required = true;
        ++laps;
      }
    }
  }

  void step() {
    if (is_local) {
      handle_movement();
    }
    handle_laps();

    fan::graphics::gui::text(laps);
  }

  bool is_local = true;
  bool checkpoint_required = true;
  fan::physics::body_id_t goal_sensor;
  fan::physics::body_id_t middle_checkpoint_sensor;
  uint16_t laps = 0;

  fan::graphics::image_t car_image{ "images/car_red.png" };
  std::vector<fan::graphics::circle_t> drift_lines;
  fan::graphics::trail_t rear_left_trail;
  fan::graphics::trail_t rear_right_trail;
  fan::time::timer timer_drift_lines_add{ fan::time::timer(0.01e9, true) };
  std::string movement_keybinds_prefix = "";

  static constexpr f32_t car_size = 32.f;

  fan::graphics::physics::character2d_t body{
    fan::graphics::physics::capsule_sprite_t{{
      .position = fan::vec3(1019.7828, 1618.1302, car_draw_depth),
      .center0 = {0.f, -car_image.get_size().normalized().y * car_size / 2.5f}, // collider size
      .center1 = {0.f,  car_image.get_size().normalized().y * car_size / 2.5f }, // collider size
      .size = car_image.get_size().normalized() * car_size,
      .angle = fan::math::pi / 2,
      .image = car_image,
      .body_type = fan::physics::body_type_e::dynamic_body,
      .shape_properties{
        .friction = 0.6f,
        .density = 0.1f,
        .fixed_rotation = false,
        .linear_damping = 7.f,
        .angular_damping = 10.f,
      }
    }}
  };


  // audio
  bool use_audio = true;
  fan::audio::piece_t engine_sound{ "audio/engine_running.sac" };
  fan::audio::sound_play_id_t piece_id;
};