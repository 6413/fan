struct boss_skeleton_t : boss_t<boss_skeleton_t> {

  f32_t attack_delay = 0.5e9;

  boss_skeleton_t() = default;
  ~boss_skeleton_t() {
    if (body.get_health() <= 0) {
      auto& level = pile->get_level();
      level.is_boss_dead = true;
      level.audio_skeleton_lord.stop();
      pile->audio_background_play_id = fan::audio::play(pile->audio_background);
    }
  }

  template<typename container_t>
  boss_skeleton_t(container_t& bll, typename container_t::nr_t nr, const fan::vec2& position) {
    draw_offset = {0, -135};
    aabb_scale = 0.1f;
    attack_hitbox_frames = {1, 3}; // or (3, 4)
    closeup_distance.x = 400;
    trigger_distance.x = 10000;
    density = 10000.f;

    open(&bll, nr, "boss_skeleton.json");
    set_initial_position(position);

    //body.set_max_health(10.f);
    body.set_max_health(300.f);
    body.set_health(body.get_max_health());
    body.attack_state.attack_range = {450, 200};
    body.movement_state.max_speed = 350.f;
    body.anim_controller.auto_update_animations = false;
    body.attack_state.knockback_force = 1200.f;
    body.attack_state.cooldown_timer.start(attack_delay),

    attack_hitbox.setup({
      .spawns = {
        {
          .frame = attack_hitbox_frames[1],
          .create_hitbox = [](const fan::vec2& center, f32_t direction) {
            return pile->engine.physics_context.create_box(
              center + fan::vec2((40.f + 190.f) * direction, 0.f),
              fan::vec2(190.f, 20.f), 0.f,
              fan::physics::body_type_e::static_body,
              {.is_sensor = true}
            );
          }
        }
      },
      .attack_animation = "attack0",
      .track_hit_targets = false
    });

    name = "Skeleton Lord";
    behavior.init(behavior_config);

    physics_step_nr = fan::physics::add_physics_step_callback([&bll, nr]() {
      std::visit([](auto& node) {
        if (node.body.get_health() <= 0) {
          return;
        }
        node.attack_hitbox.update(&node.body);
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, boss_skeleton_t>) {
          if (node.allow_move) {
            node.update_logic();
          }
        }
      }, bll[nr]);
    });
  }

  void attack_blocked() override {
    base_t::attack_blocked();
    body.attack_state.cooldown_timer.start(attack_delay * 1.5f);
  }

private:
  void update_logic() {
    // recover from stun
    if (!second_phase && body.attack_state.cooldown_timer.duration() != attack_delay && body.attack_state.cooldown_timer.finished()) {
      body.attack_state.cooldown_timer.start(attack_delay);
    }

    // rage
    if (body.get_health() < body.get_max_health() / 2.f && !second_phase) {
      second_phase = true;
      task_pulse_red = pulse_red.animate([this](auto c) { body.set_color(c); });
      attack_delay = 0.25e9;
      body.attack_state.cooldown_timer.start(attack_delay);
      body.movement_state.max_speed = 500.f;
      behavior_config.backstep_cooldown = 3.0e9;
    }
    behavior.update(body, pile->player.get_physics_pos(), behavior_config); 
    body.anim_controller.update(body, body.get_linear_velocity());
  }

  fan::graphics::physics::boss_behavior_t behavior;
  fan::graphics::physics::boss_behavior_t::config_t behavior_config;
  fan::auto_color_transition_t pulse_red = fan::pulse_red();
  fan::event::task_t task_pulse_red;
  bool second_phase = false;

public:
  bool allow_move = false;
};