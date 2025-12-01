struct enemy_t {
  static inline constexpr fan::vec2 draw_offset{0, -18};
  static inline constexpr f32_t aabb_scale = 0.19f;
  static constexpr fan::vec2 trigger_distance = {500, 500};
  static constexpr fan::vec2 closeup_distance = {150, 100};
  //TODO use collision mask for player and entities
  static inline constexpr int attack_hitbox_frames[] = {4, 8};

  enemy_t(const std::string& path, const std::source_location& caller_path = std::source_location::current()){
    f32_t density = 4.f;

    body = fan::graphics::physics::character2d_t::from_json({
      .json_path = path,
      .aabb_scale = aabb_scale,
      .draw_offset_override = draw_offset,
      .attack_cb = [this](auto& c){ return should_attack(c); },
      .physics_properties={.density=density, .fixed_rotation=true, .linear_damping=2.0f}
    }, caller_path);
    body.set_jump_height(75.f * density);
    body.movement_state.accelerate_force = 120.f / 2.3f;
    body.movement_state.max_speed = 610.f;

    body.set_size(body.get_size());
    //body.set_color(fan::color(1, 1 / 3.f, 1 / 3.f, 1));

    attack_hitbox.setup({
      .spawns = {{
        .frame = attack_hitbox_frames[0],
        .create_hitbox = [this](const fan::vec2& center, f32_t direction){
          fan::vec2 offset = fan::vec2(50.f * direction, 0);
          return pile->engine.physics_context.create_box(
            center + offset, fan::vec2(60, 40), 0,
            fan::physics::body_type_e::static_body, {.is_sensor = true}
          );
        }},
        {
          .frame = attack_hitbox_frames[1],
          .create_hitbox = [this](const fan::vec2& center, f32_t direction){
            fan::vec2 offset = fan::vec2(50.f * direction, 0);
            return pile->engine.physics_context.create_box(
              center + offset, fan::vec2(60, 40), 0,
              fan::physics::body_type_e::static_body, {.is_sensor = true}
            );
          }
        }
      },
      .attack_animation = "attack0",
      .track_hit_targets = false
    });

    body.setup_attack_properties({
      .damage = 10.f,
      .knockback_force = 10.f,
      .attack_range = {closeup_distance.x, body.attack_state.attack_range.y},
      .cooldown_duration = 1.0e9,
      .cooldown_timer = fan::time::timer(body.attack_state.cooldown_duration, true),
      .stun = false
    });
    ai_behavior.target = &pile->player.body;

    auto& level = pile->get_level();

    navigation.auto_jump_obstacles = true;
    navigation.jump_lookahead_tiles = 1.5f;

    navigation.on_check_obstacle = [this](const fan::vec2& check_pos){
      return is_spike_at(check_pos);
    };
    physics_step_nr = fan::physics::add_physics_step_callback([&](){
      fan::vec2 tile_size = pile->renderer.get_tile_size(level.main_map_id) * 2.f;
      fan::vec2 target_pos = pile->player.get_physics_pos();
      ai_behavior.update_ai(&body, navigation, target_pos, tile_size);
      fan::vec2 distance = ai_behavior.get_target_distance(body.get_physics_position());
      if (!((std::abs(distance.x) < trigger_distance.x && std::abs(distance.y) < trigger_distance.y))){
        ai_behavior.enable_ai_patrol({initial_position - fan::vec2(400, 0), initial_position + fan::vec2(400, 0)});
      }
      else{
        ai_behavior.enable_ai_follow(&pile->player.body, trigger_distance, closeup_distance);
      }
    });
  }
  bool should_attack(fan::graphics::physics::character2d_t& c){
    fan::vec2 distance = ai_behavior.get_target_distance(c.get_physics_position());
    return c.attack_state.try_attack(&c, distance);
  }
  bool update(){
    if (remove_this){
      auto found = std::find_if(pile->enemy_skeleton.begin(), pile->enemy_skeleton.end(), [&](enemy_t* e){ return e->body.NRI == body.NRI; });
      if (found == pile->enemy_skeleton.end()){
        fan::throw_error("trying to remove non existing enemy");
      }
      delete *found;
      *found = nullptr;
      pile->enemy_skeleton.erase(found);
      return true;
    }
    
    for (int i = 0; i < attack_hitbox.hitbox_count(); ++i){
      if (attack_hitbox.check_hit(&body, i, &pile->player.body)){
        if (pile->player.on_hit(&body, (pile->player.body.get_position() - body.get_position()).normalized())){
          return true;
        }
      }
    }

    attack_hitbox.update(&body);

    body.update_animations();

    render_health();
    return false;
  }
  void render_health(){
    int heart_count = body.get_max_health() / 10.f;
    for (int i = 0; i < heart_count; ++i){
      fan::graphics::image_t hp_image = pile->get_gui().health_empty;
      //0-1
      f32_t progress = body.get_health() / body.get_max_health();
      if (progress * heart_count > i){
        hp_image = pile->get_gui().health_full;
      }
      f32_t image_size = 8.f;
      fan::graphics::sprite({
        .position = fan::vec3(fan::vec2(body.get_physics_position() - fan::vec2(heart_count / 2.f * image_size - i * (image_size * 2.f) + image_size + image_size / 2.f, body.get_size().y / 1.5f)), 0xFF00),
        .size = image_size, 
        .image = hp_image,
        });
    }
  }
  void destroy(){
    fan::physics::remove_physics_step_callback(physics_step_nr);
    remove_this = true;
  }
  bool on_hit(fan::graphics::physics::character2d_t* source, const fan::vec2& hit_direction){
    fan::audio::play(audio_player_hits_enemy);
    body.take_hit(source, hit_direction);
    if (body.is_dead()){
      destroy();
      return true;
    }
    return false;
  }
  bool is_spike_at(const fan::vec2& pos){
    for (auto& spike : pile->get_level().spike_sensors){
      fan::vec2 spike_pos = spike.get_position();
      fan::vec2 spike_size = pile_t::level_t::spike_height * 2.f;
      if (std::abs(pos.x - spike_pos.x) < spike_size.x / 2.f && std::abs(pos.y - spike_pos.y) < spike_size.y / 2.f){
        return true;
      }
    }
    return false;
  }
  void respawn(){
    body.set_physics_position(initial_position);
    ai_behavior.enable_ai_patrol({initial_position - fan::vec2(400, 0), initial_position + fan::vec2(400, 0)});
    body.reset_health();
  }
  void set_initial_position(const fan::vec2& position){
    initial_position = position;
    respawn();
  }

  fan::graphics::physics::character2d_t body;
  fan::graphics::physics::attack_hitbox_t attack_hitbox;
  fan::graphics::physics::ai_behavior_t ai_behavior;
  fan::graphics::physics::navigation_helper_t navigation;

  fan::physics::physics_step_callback_nr_t physics_step_nr;
  fan::vec2 initial_position = 0;
  bool remove_this = false;
  fan::audio::piece_t audio_attack{"audio/enemy_attack.sac"}, audio_player_hits_enemy{"audio/player_hits_enemy.sac"};
};