struct player_t {
  static inline constexpr fan::vec2 draw_offset{ 0.f, -42.5f };
  static inline constexpr f32_t aabb_scale = 0.17f;
  static inline constexpr f32_t task_tick = 1000.f / 144.f;
  static inline constexpr int attack_hitbox_frame = 4;
  static inline constexpr f32_t sword_length = 100.f;

  static inline constexpr std::array<fan::vec2, 3> get_hitbox_points(f32_t direction){
    return {{
      {sword_length * direction, 0.0f},
      {0.0f, -10.0f},
      {0.0f, 10.0f}
      }};
  }
  player_t(){
    //particles = fan::graphics::extract_single_shape("explosion.json");

    std::string data;
    fan::io::file::read(fan::io::file::find_relative_path("effects/explosion.json"), &data);
    fan::json in = fan::json::parse(data);
    fan::graphics::shape_deserialize_t it;
    while (it.iterate(in, &particles)){}
    auto image_star = pile->engine.image_load("images/waterdrop.webp");
    particles.set_image(image_star);

    body = fan::graphics::physics::character2d_t::from_json({
      .json_path = "player/player.json",//
      .aabb_scale = aabb_scale,
      .draw_offset_override = draw_offset,
      .attack_cb = [](fan::graphics::physics::character2d_t& c) -> bool{
        if (!fan::window::is_mouse_clicked() || pile->engine.render_settings_menu){
          return false;
        }
        return c.attack_state.try_attack(&c);
      },
    });
    body.enable_default_movement();
    body.set_jump_height(60.f);
    body.enable_double_jump();
    body.sync_visual_angle(false);

    attack_hitbox.setup({
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
      .damage = 10.f,
      .knockback_force = 20.f,
      .cooldown_duration = 0.1e9,
      .on_attack_end = [this]() { did_attack = false;  }
    });

    mouse_click_handle = pile->engine.on_mouse_click(fan::mouse_left, [this](const auto& bdata){
      body.cancel_animation();
      body.attack_state.try_attack(&body);
    });
  }
  fan::event::task_t jump(bool is_double_jump){
    fan::audio::play(audio_jump);
    jump_cancelled = false;
    if (!is_double_jump){
      co_return;
    }
    body.set_rotation_point(-body.get_draw_offset());

    fan::time::timer jump_timer{1.0e9f / 2.f, true};
    while (!jump_cancelled && !jump_timer){
      f32_t progress = jump_timer.seconds() / jump_timer.duration_seconds();
      f32_t angle = progress * fan::math::two_pi * body.get_image_sign().x;
      body.set_angle(fan::vec3(0, 0, angle));
      co_await fan::graphics::co_next_frame();
    }
    body.set_angle(0.f);
  }
  void respawn(){
    if (current_checkpoint == -1){
      body.set_physics_position(pile->renderer.get_position(pile->get_level().main_map_id, "player_spawn"));
    }
    else{
      body.set_physics_position(pile->get_level().player_checkpoints[current_checkpoint].get_position());
    }
    body.set_health(body.get_max_health());
    pile->get_level().load_enemies();
  }
  fan::event::task_t particles_explode(){
    pile->player.particles.set_position(fan::vec3(pile->get_level().player_checkpoints[current_checkpoint].get_position() + fan::vec2(-32, 80), 0));
    fan::time::timer jump_timer{4.0e9f / 2.f, true};
    while (!jump_timer){
      f32_t t = jump_timer.seconds() / jump_timer.duration_seconds();
      t = std::clamp(t, 0.0f, 1.0f);
      f32_t progress = 1.0f - std::fabs(t * 2.0f - 1.0f);
      f32_t progress2 = 1.0f - std::fabs(t * 4.0f - 1.0f);
      pile->player.particles.set_color(fan::color::hsv(340.9f, 88.9f, progress * 100.f));
      ((fan::graphics::shapes::particles_t::ri_t*)pile->player.particles.GetData(fan::graphics::g_shapes->shaper))->position_velocity.y = -progress2 * 1000.f;
      co_await fan::graphics::co_next_frame();
    }
    pile->player.particles.set_color(0);
  }
  void update(){
    if (body.is_on_ground()){
      did_double_jump = false;
      jump_cancelled = true;
      did_wall_jump = false;
    }

    if (body.movement_state.jump_state.double_jump_consumed && !did_double_jump){
      did_double_jump = true;
      jump_cancelled = true;
    }
    if (fan::window::is_action_clicked("move_up") && 
      (!body.movement_state.jump_state.consumed 
        || jump_cancelled 
        || (body.wall_jump.consumed && !did_wall_jump))){
      if (body.wall_jump.consumed){
        did_wall_jump = true;
      }
      task_jump = jump(body.movement_state.jump_state.double_jump_consumed);
    }

    if (!did_attack && body.animation_on("attack0", attack_hitbox_frame)) {
      fan::audio::play(audio_attack);
      did_attack = true;
    }

    for (enemy_t& enemy : pile->enemy_skeleton){
      if (!attack_hitbox.check_hit(&body, 0, &enemy.body)) {
        continue;
      }
      if (enemy.on_hit(&body, (enemy.body.get_position() - body.get_position()).normalized())){
        break;
      }
    }

    attack_hitbox.update(&body);

    body.update_animations();

    for (auto [i, checkpoint] : fan::enumerate(pile->get_level().player_checkpoints)){
      if (fan::physics::is_on_sensor(pile->player.body, checkpoint) && pile->player.current_checkpoint > i){
        pile->player.current_checkpoint = i;
        fan::audio::play(audio_checkpoint);
        task_particles = particles_explode();
        fan::graphics::gui::print("Checkpoint reached!!!!!");
      }
    }
  }
  fan::vec2 get_physics_pos(){
    return body.get_physics_position();
  }
  bool on_hit(fan::graphics::physics::character2d_t* source, const fan::vec2& hit_direction){
    fan::audio::play(audio_enemy_hits_player);
    body.take_hit(source, hit_direction);
    if (body.get_health() <= 0){
      body.cancel_animation();
      respawn();
      return true;
    }
    return false;
  }

  fan::graphics::physics::character2d_t body;
  fan::graphics::physics::attack_hitbox_t attack_hitbox;
  fan::graphics::engine_t::buttons_handle_t mouse_click_handle;
  fan::event::task_t task_jump;
  bool jump_cancelled = false;
  bool did_double_jump = false, did_wall_jump = false, did_attack = false;
  fan::audio::piece_t audio_jump{"audio/jump.sac"}, 
    audio_attack{"audio/player_attack.sac"}, 
    audio_enemy_hits_player{"audio/enemy_hits_player.sac"},
    audio_checkpoint{"audio/checkpoint.sac"};
  int current_checkpoint = -1;
  fan::graphics::shape_t particles;
  fan::event::task_t task_particles;
};