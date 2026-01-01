struct player_t {
  static inline constexpr fan::vec2 draw_offset{ 0.f, -42.5f };
  static inline constexpr f32_t aabb_scale = 0.17f;
  static inline constexpr f32_t task_tick = 1000.f / 144.f;
  static inline constexpr int attack_hitbox_frame = 4;
  static inline constexpr f32_t sword_length = 120.f;

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

    auto drink_potion = fan::graphics::shape_from_json("effects/drink_potion.json");
    drink_potion.stop_particles();
    drink_potion.set_dynamic();
    std::fill(particles_drink_potion, particles_drink_potion + std::size(particles_drink_potion), drink_potion);
    //for (auto& i : particles_drink_potion) {
    //  i.set_dynamic();
    //}
    //particles = fan::graphics::extract_single_shape("explosion.json");
    //particles_drink_potion = 
    auto image_star = pile->engine.image_load("images/waterdrop.webp");
    particles = fan::graphics::shape_from_json("effects/explosion.json");
    particles.set_image(image_star);

    body = fan::graphics::physics::character2d_t::from_json({
      .json_path = "player/player.json",//
      .aabb_scale = aabb_scale,
      .attack_cb = [](fan::graphics::physics::character2d_t& c) -> bool{
        if (!fan::window::is_mouse_clicked() && !fan::window::is_key_pressed(fan::gamepad_right_bumper) || fan::graphics::gui::want_io()){
          return false;
        }
        return c.attack_state.try_attack(&c);
      },
    });
    body.set_draw_offset(draw_offset);
    body.set_flags(fan::graphics::sprite_flags_e::use_hsl);
    body.set_color(fan::color::hsl(56.7f, 18.3f, -58.4f));

    body.set_dynamic();
    body.enable_default_movement();
    body.set_jump_height(60.f);
    body.enable_double_jump();
    body.sync_visual_angle(false);
    body.movement_state.jump_state.on_jump = [&] (int jump_state) {
      task_jump = jump(jump_state);
    };

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

    { // keybinds handling
      int key_handle_index = 0;
      key_click_handles[key_handle_index++] = pile->engine.on_key_click(fan::key_r, [this](const auto&) {
        if (!potion_count) return;
        if (!potion_consume_timer) return;
        --potion_count;
        static constexpr f32_t potion_heal = 20.f;
        fan::audio::play(audio_drink_potion);
        body.set_health(std::min(body.get_health() + potion_heal, body.get_max_health()));
        {
          fan::vec3 player_pos = body.get_center() - fan::vec2(0, body.get_size().y / 4.f);
          static int potion_particle_index = 0;
          particles_drink_potion[potion_particle_index].start_particles();

          
          particles_drink_potion[potion_particle_index].set_position(fan::vec3(fan::vec2(player_pos), player_pos.z + 1));
         // particles_drink_potion[potion_particle_index].get_shape_data<fan::graphics::shapes::particles_t>().begin_angle = -0.777 + 0.777 * -body.get_linear_velocity().sign().x;
          potion_particle_index = (potion_particle_index + 1) % std::size(particles_drink_potion);

        }
        potion_consume_timer.restart();
      });
    }
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
      body.set_physics_position(pile->renderer.get_all_spawn_positions(pile->get_level().main_map_id, tilemap_loader_t::fte_t::mesh_property_t::player_spawn)[0]);
    }
    else{
      body.set_physics_position(pile->get_level().player_checkpoints[current_checkpoint].visual.get_position());
    }
    //body.set_max_health(200);
    body.set_health(body.get_max_health());
    //body.set_health(10.f);

    pile->get_level().load_enemies();
    //body.set_dynamic();// if player is always in the camera, no need to call update_dynamic i think
  }
  fan::event::task_t particles_explode(){
    particles.set_position(fan::vec3(pile->get_level().player_checkpoints[current_checkpoint].entity.get_position() + fan::vec2(-32, 80), 0));
    fan::time::timer jump_timer{4.0e9f / 2.f, true};
    while (!jump_timer){
      f32_t t = jump_timer.seconds() / jump_timer.duration_seconds();
      t = std::clamp(t, 0.0f, 1.0f);
      f32_t progress = 1.0f - std::fabs(t * 2.0f - 1.0f);
      f32_t progress2 = 1.0f - std::fabs(t * 4.0f - 1.0f);
      particles.set_color(fan::color::hsv(340.9f, 88.9f, progress * 100.f));
      ((fan::graphics::shapes::particles_t::ri_t*)particles.GetData(fan::graphics::g_shapes->shaper))->position_velocity.y = -progress2 * 1000.f;
      co_await fan::graphics::co_next_frame();
    }
    particles.set_color(0);
  }
  void handle_attack() {
    if (!did_attack && body.animation_on("attack0", attack_hitbox_frame)) {
      fan::audio::play(audio_attack);
      did_attack = true;
    }

    for (auto enemy : pile->enemies()) {
      if (!attack_hitbox.check_hit(&body, 0, &enemy.get_body())) {
        continue;
      }
      if (enemy.on_hit(&body, (enemy.get_body().get_position() - body.get_position()).normalized())) {
        break;
      }
    }

    attack_hitbox.update(&body);
  }

  void update() {
    static f32_t v = 0;
    v += pile->engine.delta_time * 100.f;
    body.set_color(fan::color::hsl(v, 18.3f, -58.4f));

    player_light.set_position(body.get_center());
    if (pile->get_level().is_entering_door) {
      static f32_t moved = body.get_position().x;
      if (body.get_position().x - moved < 128.f) {
        body.movement_state.ignore_input = true;
        body.movement_state.move_to_direction_raw(body, fan::vec2(1, 0));
      }
      else {
        body.movement_state.ignore_input = false;
        auto& level = pile->get_level();
        level.enter_boss();
      }
    }

    //body.update_dynamic();
    handle_attack();

    body.update_animations();

    for (auto [i, checkpoint] : fan::enumerate(pile->get_level().player_checkpoints)) {
      if (fan::physics::is_on_sensor(body, checkpoint.entity) && current_checkpoint < (int)i) {
        current_checkpoint = i;
        fan::audio::play(audio_checkpoint);
        //task_particles = particles_explode();
        fan::graphics::gui::print("Checkpoint reached!!!!!");
        break;
      }
    }

    auto& map_compiled = pile->get_level().main_compiled_map;
    if (get_physics_pos().y > map_compiled.map_size.y * (map_compiled.tile_size.y*2.f)) {
      respawn();
    }
  }
  fan::vec2 get_physics_pos() {
    return body.get_physics_position();
  }
  bool on_hit(fan::graphics::physics::character2d_t* source, const fan::vec2& hit_direction) {
    fan::audio::play(audio_enemy_hits_player);
    body.take_hit(source, hit_direction);
    if (body.get_health() <= 0) {
      body.cancel_animation();
      pile->get_level().reload_map();
      return true;
    }
    return false;
  }

  fan::graphics::physics::character2d_t& get_body() {
    return body;
  }

  fan::graphics::physics::character2d_t body;
  fan::graphics::physics::attack_hitbox_t attack_hitbox;

  fan::event::task_t task_jump;
  bool jump_cancelled = false, did_attack = false;

  fan::audio::piece_t 
    audio_jump{"audio/jump.sac"},
    audio_attack{"audio/player_attack.sac"},
    audio_enemy_hits_player{"audio/enemy_hits_player.sac"},
    audio_checkpoint{"audio/checkpoint.sac"},
    audio_drink_potion{"audio/drink_potion.sac"};

  fan::graphics::shape_t particles;
  fan::event::task_t task_particles;

  fan::graphics::engine_t::key_handle_t key_click_handles[10];

  int current_checkpoint = 1;
  
  uint16_t potion_count = 10;
  fan::time::timer potion_consume_timer {0.1e9, true};
  fan::graphics::shape_t particles_drink_potion[4];
  fan::graphics::light_t player_light {{.position=0}};
};