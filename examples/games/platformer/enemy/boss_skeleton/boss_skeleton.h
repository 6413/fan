struct boss_skeleton_t : boss_t<boss_skeleton_t> {
  boss_skeleton_t() = default;

  template<typename container_t>
  boss_skeleton_t(container_t& bll, typename container_t::nr_t nr, const fan::vec3& position) {
    draw_offset = {0, -135};
    aabb_scale = 0.1f;
    attack_hitbox_frames = {4};
    closeup_distance.x = 400;
    trigger_distance.x = 10000;

    open(bll, nr, "boss_skeleton.json");
    set_initial_position(position);

    body.set_max_health(100.f);
    body.set_health(body.get_max_health());
    body.attack_state.attack_range = {closeup_distance.x + 50, 200};
    body.movement_state.max_speed = 350.f;

    attack_hitbox.setup({
      .spawns = {{
          .frame = attack_hitbox_frames[0],
          .create_hitbox = [](const fan::vec2& center, f32_t direction){

      fan::vec2 offset = fan::vec2((40.f + 380.f/2.f) * direction, 0.f);
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

    physics_step_nr = fan::physics::add_physics_step_callback([
      &bll,
      nr,
      xdist = closeup_distance.x
    ]() mutable {
      std::visit([xdist](auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, boss_skeleton_t>) {
          auto& level = pile->get_level();
          fan::vec2 tile_size = pile->renderer.get_tile_size(level.main_map_id) * 2.f;
          fan::vec2 target_pos = pile->player.get_physics_pos();
          update_boss_logic(node, xdist, target_pos, tile_size);
        }
      }, bll[nr]);
    });
  }

  static void update_boss_logic(
    boss_skeleton_t& node,
    f32_t ideal_distance,
    const fan::vec2& target_pos,
    const fan::vec2& tile_size
  ) {
    auto& body = node.body;
    auto& ai = node.ai_behavior;
    auto& nav = node.navigation;

    if (body.get_health() < body.get_max_health() / 2.f) {
      if (!node.second_phase) {
        node.second_phase = true;
        node.task_pulse_red = node.pulse_red.animate([&body](auto c) {
          body.set_color(c); 
        });
        body.movement_state.max_speed = 500.f;
        body.attack_state.cooldown_duration = 1.0e9;
      }
      //static fan::color c = body.get_color();
      //static f32_t t = 0;
      //body.set_color(c.lerp(fan::color(1, 0.2, 0.2), t));
    }

    ai.update_ai(&body, nav, target_pos, tile_size);
    fan::vec2 distance = ai.get_target_distance(body.get_physics_position());

    if (!body.raycast(pile->player.body) || body.attack_state.is_attacking) {
      return;
    }

    ai.type = fan::graphics::physics::ai_behavior_t::behavior_type_e::none;

    fan::vec2 movement_direction = compute_movement_direction(distance, ideal_distance);

    apply_movement(body, movement_direction);
    update_orientation(body, distance);
    update_animations(body);
    node.did_attack = body.attack_state.is_attacking;
  }

private:

  static fan::vec2 compute_movement_direction(const fan::vec2& distance, f32_t ideal) {
    fan::vec2 dir{0.f, 0.f};

    f32_t ax = std::abs(distance.x);
    f32_t margin = 20.f;
    bool too_far   = ax > ideal + margin;
    bool too_close = ax < ideal - margin;
    int dir_to_player = (distance.x > 0.f) ? 1 : -1;

    if (too_far) {
      dir.x = (f32_t)dir_to_player;

    }
    //else if (too_close) {
    //  dir.x = (f32_t)-dir_to_player;        

    //}

    return dir;
  }

  static void apply_movement(
    fan::graphics::physics::character2d_t& body,
    const fan::vec2& movement_direction
  ) {

    if (std::abs(body.get_linear_velocity().y) > 0.5f) {
      return;
    }
    body.movement_state.move_to_direction_raw(body, fan::vec2(movement_direction.x, 0.f));
  }

  static void update_orientation(
    fan::graphics::physics::character2d_t& body,
    const fan::vec2& distance
  ) {
    fan::vec2 vel  = body.get_linear_velocity();
    fan::vec2 sign = body.get_image_sign();

    int8_t desired;

    if (std::abs(vel.x) > 5.0f) {
      desired = (int8_t)fan::math::sgn(vel.x);
    }
    else {
      desired = (distance.x > 0.f) ? 1 : -1;
    }

    if ((int8_t)fan::math::sgn(sign.x) != desired && std::abs(vel.y) < 0.5f) {
      body.set_image_sign({(f32_t)desired, sign.y});
    }
  }

  static void update_animations(fan::graphics::physics::character2d_t& body) {
    if (body.anim_controller.auto_update_animations) {
      body.anim_controller.update(&body);
    }
  }

  fan::color_transition_t pulse_red = fan::pulse_red();
  fan::event::task_t task_pulse_red;
  int blocked_frames = 0;
  bool did_attack = false;
  bool second_phase = false;
};