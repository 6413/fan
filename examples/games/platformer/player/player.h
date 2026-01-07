struct player_t {
  static inline constexpr fan::vec2 draw_offset{0.f, -42.5f};
  static inline constexpr f32_t aabb_scale = 0.17f;
  static inline constexpr f32_t sword_length = 120.f;
  static inline constexpr int attack_hitbox_frame = 3;

  static inline constexpr std::array<fan::vec2, 3> get_hitbox_points(f32_t direction){
    return {{
      {sword_length * direction, 0.0f},
      {0.0f, -20.0f},
      {0.0f, 20.0f}
    }};
  }

  player_t(){
    player_light.set_dynamic();
    player_light.set_color(fan::color(0.610, 0.550, 0.340, 1.0));
    player_light.set_size(256);

    potion_particles.from_json("effects/drink_potion.json");

    auto image_star = pile->engine.image_load("images/waterdrop.webp");
    particles = fan::graphics::shape_from_json("effects/explosion.json");
    particles.set_image(image_star);

    body = fan::graphics::physics::character2d_t::from_json({
      .json_path = "player/player.json",
      .aabb_scale = aabb_scale,
      .attack_cb = [](fan::graphics::physics::character2d_t& c) -> bool {
        const bool attack_input =
          fan::window::is_input_action_active(fan::actions::light_attack) ||
          fan::window::is_key_pressed(fan::gamepad_right_bumper);

        if (!attack_input || fan::graphics::gui::want_io()) {
          return false;
        }

        return c.attack_state.try_attack(&c);
      },
    });

    body.set_draw_offset(draw_offset);
    body.set_flags(fan::graphics::sprite_flags_e::use_hsl);
    body.set_color(fan::color::hsl(56.7f, 18.3f, -58.4f));
    body.set_dynamic();

    fan::graphics::physics::character_movement_preset_t::setup_default_controls(body);

    body.movement_state.jump_state.on_jump = [&](int jump_type) {
      task_jump = jump(jump_type == 1);
    };

    combat.hitbox.setup({
      .spawns = {{
        .frame = attack_hitbox_frame,
        .create_hitbox = [](const fan::vec2& center, f32_t direction){
          auto points = get_hitbox_points(direction);
          return pile->engine.physics_context.create_polygon(
            center, 0.0f, points.data(), points.size(),
            fan::physics::body_type_e::static_body, {.is_sensor = true}
          );
        }
      }},
      .attack_animation = "attack0",
      .track_hit_targets = true
    });

    body.setup_attack_properties({
      .max_health = 50.f,
      .damage = 10.f,
      .knockback_force = 20.f,
      .cooldown_duration = 0.05e9,
      .on_attack_end = [this]() { combat.did_attack = false; },
    });

    physics_step_nr = fan::physics::add_physics_step_callback([this]() {
      handle_attack();
    });
  }

  fan::event::task_t jump(bool is_double_jump) {
    audio_jump.play();
    if (!is_double_jump) {
      co_return;
    }
    body.set_rotation_point(-body.get_draw_offset());

    fan::time::timer jump_timer {1.0e9f / 2.f, true};
    while (!jump_timer) {
      f32_t progress = jump_timer.seconds() / jump_timer.duration_seconds();
      f32_t angle = progress * fan::math::two_pi * body.get_image_sign().x;
      body.set_angle(fan::vec3(0, 0, angle));
      co_await fan::graphics::co_next_frame();
    }
    body.set_angle(0.f);
  }

  void respawn(){
    body.set_angle(0.f);
    
    fan::vec3 spawn_pos = pile->checkpoint_system.get_respawn_position(pile->renderer, pile->get_level().main_map_id);
    if (spawn_pos == fan::vec3(0)) {
      spawn_pos = fan::graphics::tilemap::helpers::get_spawn_position_or_default(
        pile->renderer, pile->get_level().main_map_id
      );
    }
    body.set_physics_position(spawn_pos);

    body.set_health(body.get_max_health());
    pile->get_level().load_enemies();
  }

  void handle_attack(){
    if (!combat.did_attack && body.animation_crossed("attack0", attack_hitbox_frame)) {
      audio_attack.play();
      combat.did_attack = true;
    }

    combat.handle_attack(body, pile->enemies());
  }

  void drink_potion(){
    if (!potion_count) return;
    if (!potion_consume_timer) return;
    --potion_count;

    static constexpr f32_t potion_heal = 20.f;
    body.set_health(std::min(body.get_health() + potion_heal, body.get_max_health()));

    audio_drink_potion.play();

    fan::vec3 player_pos = body.get_center() - fan::vec2(0, body.get_size().y / 4.f);
    potion_particles.spawn_at(fan::vec3(fan::vec2(player_pos), player_pos.z + 1));

    potion_consume_timer.restart();
  }

  void update(){
    combat.hitbox.process_destruction();
    if (fan::window::is_input_action_active(actions::drink_potion)) {
      drink_potion();
    }

    if (body.is_on_ground()) {
      if (body.get_angle().z != 0.f) {
        body.set_angle(0.f);
      }
    }

    static f32_t v = 0;
    v += pile->engine.delta_time * 100.f;

    player_light.set_position(body.get_center());
    body.update_animations();

    if (!pile->level_stage) {
      return;
    }

    auto& level = pile->get_level();
    if (level.is_entering_door) {
      static f32_t moved = body.get_position().x;
      if (body.get_position().x - moved < 128.f) {
        body.movement_state.ignore_input = true;
        body.movement_state.move_to_direction_raw(body, fan::vec2(1, 0));
      }
      else {
        body.movement_state.ignore_input = false;
        level.enter_boss();
      }
    }

    pile->checkpoint_system.check_and_update(body, [this](auto& cp) {
      audio_checkpoint.play();
      fan::graphics::gui::print("Checkpoint reached!");
    });

    auto& map_compiled = level.main_compiled_map;
    if (get_physics_pos().y > map_compiled.map_size.y * (map_compiled.tile_size.y * 2.f)) {
      respawn();
    }
  }

  fan::vec2 get_physics_pos(){
    return body.get_physics_position();
  }

  bool on_hit(fan::graphics::physics::character2d_t* source, const fan::vec2& hit_direction){
    audio_enemy_hits_player.play();
    body.take_hit(source, hit_direction);
    if (body.get_health() <= 0) {
      body.cancel_animation();
      pile->get_level().reload_map();
      return true;
    }
    return false;
  }

  fan::graphics::physics::character2d_t& get_body(){
    return body;
  }

  fan::graphics::physics::character2d_t body;
  fan::graphics::physics::combat_controller_t combat;
  fan::event::task_t task_jump;
  fan::audio::sound_t
    audio_jump{"audio/jump.sac"},
    audio_attack{"audio/player_attack.sac"},
    audio_enemy_hits_player{"audio/enemy_hits_player.sac"},
    audio_checkpoint{"audio/checkpoint.sac"},
    audio_drink_potion{"audio/drink_potion.sac"};
  fan::graphics::shape_t particles;
  fan::event::task_t task_particles;
  uint16_t potion_count = 1;
  fan::time::timer potion_consume_timer{0.1e9, true};
  fan::graphics::effects::particle_pool_t::pool_t<4> potion_particles;
  fan::graphics::light_t player_light{{.position = 0}};
  fan::physics::step_callback_nr_t physics_step_nr;
};