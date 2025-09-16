struct car_ai_t : car_t {

  void open() {
    car_t::open(
      fan::vec3(1019.7828, 1578.1302, car_draw_depth)
    );
    use_audio = false;

    auto& racing_track_stage_data = pile.stage_loader.get_stage_data<pile_t::racing_track_t>(pile.current_stage);

    path_solver = fan::algorithm::path_solver_t(racing_track_stage_data.main_compiled_map.map_size * 2, racing_track_stage_data.main_compiled_map.tile_size * 2);

    fan::vec2 grid_size = racing_track_stage_data.main_compiled_map.tile_size;

    for (auto& x : racing_track_stage_data.main_compiled_map.physics_shapes) {
      if (x.physics_shapes.shape_properties.is_sensor) {
        continue;
      }
      // how many grid cells this shape spans
      fan::vec2i grid_count = fan::vec2i(
        std::ceil(x.size.x / grid_size.x),
        std::ceil(x.size.y / grid_size.y)
      );

      // top-left corner (since position is center)
      fan::vec2 top_left = fan::vec2(x.position) - x.size;

      for (int gx = 0; gx < grid_count.x; gx++) {
        for (int gy = 0; gy < grid_count.y; gy++) {
          // cell center (shift by +0.5 * grid_size so we hit the middle of each cell)
          fan::vec2 cell_center = top_left + fan::vec2(
            (gx + 0.5)*grid_size.x * 2.f,
            (gy + 0.5)*grid_size.y * 2.f
          );
          fan::vec3 p = cell_center;
          p.z = 50000;
          rect_path.push_back({ {
         .position = p,
         .size = grid_size,
         .color = fan::random::bright_color().set_alpha(0.8),
         .blending = true
       } });
          path_solver.add_collision(p);
        }
      }
    /*   rect_path.push_back({ {
         .position = fan::vec3(fan::vec2(x.position), 50000),
         .size = grid_size,
         .color = fan::random::bright_color().set_alpha(0.8),
         .blending = true
       } });*/
      //fan::vec3 p = fan::vec3(cell_center + fan::vec2(0, -grid_size.y / 6), 50000);
    /*  rect_path.push_back({ {
        .position = fan::vec3(fan::vec2(x.position), 50000),
        .size = x.size,
        .color = fan::random::bright_color().set_alpha(0.8),
        .blending = true
      } });*/
    }

    rect_dst = { {
.position = 0,
.size = racing_track_stage_data.main_compiled_map.tile_size / 4,
.color = fan::colors::red.set_alpha(0.3),
.blending = true
} };

    fan::vec2 new_target = fan::physics::physics_to_render(middle_checkpoint_sensor.get_physics_position());
    new_target.x -= rect_dst.get_size().x * 25.f; // hardcoded to this map

    set_new_target_position(new_target);

  }

  void handle_movement(const fan::vec2& input) {
    f32_t dt = engine.delta_time;
    // texture points up, so rotate to 0 angle
    fan::basis basis = body.get_basis();
    fan::vec2 forward = basis.forward;
    fan::vec2 right = basis.right;
    fan::vec2 vel = body.get_linear_velocity();

    // forward movement
    f32_t speed = 2000.f * body.get_mass();
    // (-input.y) sign conflict
    body.apply_force_center(forward * (-input.y) * speed);

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
    
    f32_t steering = 80.f * body.get_mass();
    if (input.y) {
      body.apply_angular_impulse(input.x * steering * dt * -input.y);
    }
  }

  void handle_ai_driving() {
    // Static variables for stuck detection and emergency handling
    static fan::time::timer stuck_timer(0.5e9, true); // Check every 500ms
    static fan::vec2 last_position = body.get_position();
    static int stuck_counter = 0;
    static bool in_emergency = false;
    static fan::time::timer emergency_timer(3.5e9, true);
    static bool was_reversing = false;

    fan::vec2 current_pos = body.get_position();
    f32_t current_speed = body.get_linear_velocity().length();

    // STUCK DETECTION
    if (stuck_timer.finished()) {
      f32_t distance_moved = current_pos.distance(last_position);

      if (distance_moved < 6.0f && current_speed < 10.0f) {
        stuck_counter++;
        fan::print("STUCK DETECTED - counter:", stuck_counter);
      }
      else if (distance_moved > 15.0f || current_speed > 20.0f) {
        stuck_counter = 0;
        in_emergency = false;
      }

      last_position = current_pos;
      stuck_timer.restart();
    }

    // EMERGENCY MODE
    if (stuck_counter >= 2) {
      if (!in_emergency) {
        in_emergency = true;
        emergency_timer.restart();
        fan::print("ENTERING EMERGENCY");
      }

      // Simple emergency: reverse + turn
      fan::vec2 emergency_input = fan::vec2(
        (stuck_counter % 2 == 0) ? -0.8f : 0.8f,  // Alternate turning
        0.7f  // Reverse
      );
      handle_movement(emergency_input);

      if (emergency_timer.finished() || current_speed > 50.0f) {
        in_emergency = false;
        stuck_counter = 0;
        fan::print("EXITING EMERGENCY");
      }

      car_t::step();
      return;
    }

    // NORMAL AI PATHFINDING
    fan::vec2 world_direction = path_solver.step(body.get_position(), 80.f);

    if (world_direction.length() > 0.1f) {
      world_direction = world_direction.normalized();

      fan::basis basis = body.get_basis();
      fan::vec2 forward = basis.forward;
      fan::vec2 right = basis.right;

      f32_t forward_alignment = world_direction.dot(forward);
      f32_t right_alignment = world_direction.dot(right);

      fan::vec2 ai_input;
      ai_input.x = std::clamp(right_alignment * 0.7f, -1.0f, 1.0f);

      // FIXED REVERSE LOGIC - much simpler and more reliable
      bool should_reverse = false;

      if (was_reversing) {
        // If we were reversing, switch to forward when:
        // 1. Target is clearly ahead, OR
        // 2. We have good speed (momentum to turn)
        if (forward_alignment > 0.5f || current_speed > 18.0f) {
          should_reverse = false;
        }
        else {
          should_reverse = true;
        }
      }
      else {
        // If we were going forward, only reverse when:
        // 1. Target is clearly behind AND we're moving slowly
        if (forward_alignment < -0.3f && current_speed < 12.0f) {
          should_reverse = true;
        }
      }

      was_reversing = should_reverse;

      if (should_reverse) {
        ai_input.y = 0.5f; // Reverse
        ai_input.x *= 0.8f; // Less steering when reversing
      }
      else {
        // Forward movement with speed control
        if (current_speed < 10.0f && std::abs(right_alignment) > 0.6f) {
          ai_input.y = -0.9f; // Build speed for turning
          ai_input.x *= 0.4f; // Minimal steering
        }
        else {
          f32_t turn_sharpness = std::abs(right_alignment);
          ai_input.y = turn_sharpness > 0.7f ? -0.5f : -0.8f; // Slow for sharp turns
        }
      }

      handle_movement(ai_input);
    }
    else {
      handle_movement(fan::vec2(0, 0));
    }
  }

  void set_new_target_position(const fan::vec2& new_pos) {
    auto& racing_track_stage_data = pile.stage_loader.get_stage_data<pile_t::racing_track_t>(pile.current_stage);

    rect_path.clear();
    fan::vec2 dst = new_pos;
    path_solver.set_dst(dst);
    rect_dst.set_position(fan::vec3(dst, 50000));
    path_solver.init(body.get_position());

    rect_path.reserve(path_solver.path.size());
    for (const auto& p : path_solver.path) {
      fan::vec2i pe = p;
      rect_path.push_back({ {
        .position = fan::vec3(pe * racing_track_stage_data.main_compiled_map.tile_size * 2, 50000),
        .size = racing_track_stage_data.main_compiled_map.tile_size / 4,
        .color = fan::colors::cyan.set_alpha(0.3),
        .blending = true
      } });
    }
  }

  void update() {
    static fan::time::timer f{ (uint64_t).5e9, true};
    if (f) {
      handle_ai_driving();
    }

    car_t::step();

    if (checkpoint_required && rect_path.empty()) {
      set_new_target_position(fan::physics::physics_to_render(middle_checkpoint_sensor.get_physics_position()));
    }
    if (!checkpoint_required) {
      set_new_target_position(0);
    }
    if (rect_path.size() && path_solver.current_position < rect_path.size()) {
      rect_path[path_solver.current_position].set_color(fan::colors::green);
    }
  }

  fan::algorithm::path_solver_t path_solver;
  std::vector<fan::graphics::rectangle_t> rect_path;
  fan::graphics::rectangle_t rect_dst;
};