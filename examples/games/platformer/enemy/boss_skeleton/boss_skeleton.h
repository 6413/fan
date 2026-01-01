struct boss_skeleton_t : boss_t<boss_skeleton_t> {
  boss_skeleton_t() = default;
  ~boss_skeleton_t() {
    if (body.get_health() <= 0) {
      pile->get_level().is_boss_dead = true;
    }
  }
  template<typename container_t>
  boss_skeleton_t(container_t& bll, typename container_t::nr_t nr, const fan::vec2& position) {
    draw_offset = {0, -135};
    aabb_scale = 0.1f;
    attack_hitbox_frames = {4};
    closeup_distance.x = 400;
    trigger_distance.x = 10000;

    open(bll, nr, "boss_skeleton.json");
    set_initial_position(position);

    body.set_max_health(10.f);
    body.set_health(body.get_max_health());
    body.attack_state.attack_range = {closeup_distance.x + 50, 200};
    body.movement_state.max_speed = 350.f;
    body.anim_controller.auto_update_animations = false;

    f32_t mass = body.get_mass();
    mass *= 500.f;
    body.set_mass(mass);

    attack_hitbox.setup({
      .spawns = {{
        .frame = attack_hitbox_frames[0],
        .create_hitbox = [](const fan::vec2& center, f32_t direction){
          fan::vec2 offset = fan::vec2((40.f + 380.f / 2.f) * direction, 0.f);
          return pile->engine.physics_context.create_box(
            center + offset,
            fan::vec2(380.f / 2.f, 20.f),
            0.f,
            fan::physics::body_type_e::static_body,
            {.is_sensor = true}
          );
        }
      }},
      .attack_animation = "attack0",
      .track_hit_targets = false
    });

    name = "Skeleton Lord";
    idle_movement_timer.start(fan::random::value_i64(2.0e9, 4.0e9));
    backstep_cooldown.start(5.0e9);

    physics_step_nr = fan::physics::add_physics_step_callback([
      &bll,
      nr,
      xdist = closeup_distance.x
    ]() mutable {
      std::visit([xdist](auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, boss_skeleton_t>) {
          fan::vec2 target_pos = pile->player.get_physics_pos();
          if (!node.allow_move) {
            return;
          }
          update_boss_logic(node, xdist, target_pos);
        }
      }, bll[nr]);
    });
  }

  static void update_boss_logic(
    boss_skeleton_t& node,
    f32_t ideal_distance,
    const fan::vec2& target_pos
  ) {
    auto& body = node.body;

    if (body.get_health() < body.get_max_health() / 2.f && !node.second_phase) {
      node.second_phase = true;
      node.task_pulse_red = node.pulse_red.animate([&body](auto c) {
        body.set_color(c);
      });
      body.attack_state.cooldown_duration = 0.5e9;
      node.backstep_cooldown.set_time(3.0e9);

      body.movement_state.max_speed = 500.f;
    }

    fan::vec2 distance = target_pos - body.get_physics_position();
    update_orientation(body, distance);

    if (node.is_backstepping) {
      body.movement_state.move_to_direction_raw(body, {(f32_t)node.backstep_dir, 0.f});
      if (node.backstep_timer.finished()) {
        node.end_backstep();
      }
      body.anim_controller.update(&body);
      return;
    }

    f32_t ax = std::abs(distance.x);

    if (!body.attack_state.is_attacking && ax < body.attack_state.attack_range.x && node.backstep_cooldown.finished()) {
      if (fan::random::value_f32(0, 1) > 0.3f) {
        node.perform_backstep(distance);
        return;
      }
    }

    fan::vec2 movement_direction = compute_movement_direction(distance, ideal_distance);

    if (movement_direction.x == 0 && node.idle_movement_timer.finished()) {
      node.perform_idle_movement(distance);
      node.idle_movement_timer.start(fan::random::value_i64(3.0e9, 6.0e9));
    }

    apply_movement(body, movement_direction);
    body.anim_controller.update(&body);
  }

private:

  static fan::vec2 compute_movement_direction(const fan::vec2& distance, f32_t ideal) {
    fan::vec2 dir{0.f, 0.f};
    if (std::abs(distance.x) > ideal + 20.f) {
      dir.x = (distance.x > 0.f) ? 1.f : -1.f;
    }
    return dir;
  }

  static void apply_movement(
    fan::graphics::physics::character2d_t& body,
    const fan::vec2& movement_direction
  ) {
    body.movement_state.move_to_direction_raw(body, {movement_direction.x, 0.f});
  }

  static void update_orientation(
    fan::graphics::physics::character2d_t& body,
    const fan::vec2& distance
  ) {
    if (body.attack_state.is_attacking) { return;}
    fan::vec2 sign = body.get_image_sign();
    int desired = (distance.x > 0.f) ? 1 : -1;
    if ((int)fan::math::sgn(sign.x) != desired) {
      body.set_image_sign({(f32_t)desired, sign.y});
    }
  }

  void perform_backstep(const fan::vec2& distance) {
    int dir_away = (distance.x > 0.f) ? -1 : 1;
    backstep_dir = dir_away;
    is_backstepping = true;
    backstep_timer.start(0.6e9);
    backstep_cooldown.start(second_phase ? 3.0e9 : 5.0e9);
  }

  void end_backstep() {
    is_backstepping = false;
    backstep_dir = 0;
  }

  void perform_idle_movement(const fan::vec2& distance) {
    f32_t r = fan::random::value_f32(0, 1);
    int dir = (distance.x > 0.f) ? 1 : -1;

    if (r < 0.3f) {
      body.movement_state.move_to_direction_raw(body, {(f32_t)dir, 0.f});
    }
    else if (r < 0.6f) {
      body.movement_state.move_to_direction_raw(body, {(f32_t)-dir, 0.f});
    }
  }

  fan::auto_color_transition_t pulse_red = fan::pulse_red();
  fan::event::task_t task_pulse_red;
  fan::time::timer idle_movement_timer;
  fan::time::timer backstep_timer;
  fan::time::timer backstep_cooldown;

  bool is_backstepping = false;
  bool second_phase = false;
  int backstep_dir = 0;

public:
  bool allow_move = false;
};
