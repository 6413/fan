struct entity_t {
  static inline constexpr fan::vec2 draw_offset {0, -18};
  static inline constexpr f32_t aabb_scale = 0.19f;
  static constexpr fan::vec2 trigger_distance = {500, 500};
  static constexpr fan::vec2 closeup_distance = {150, 100};
  //TODO use collision mask for player and entities
  static inline constexpr int attack_hitbox_frames[] = {4, 8};

  fan::physics::physics_step_callback_nr_t physics_step_nr;

  entity_t(const fan::vec3& pos = 0) {
    f32_t density = 4.f;

    body = fan::graphics::physics::character2d_t::from_json({
      .json_path = "skeleton.json",
      .aabb_scale = aabb_scale,
      .draw_offset_override = draw_offset,
      .attack_cb = [this](auto& c) { return should_attack(c); },
      .physics_properties={.density=density, .fixed_rotation=true, .linear_damping=2.0f}
    });
    body.set_jump_height(75.f * density);
    body.movement_state.accelerate_force = 120.f / 2.3f;
    body.movement_state.max_speed = 610.f;
    body.set_size(body.get_size());
    //body.set_color(fan::color(1, 1 / 3.f, 1 / 3.f, 1));
    body.setup_attack_properties({
      .damage = 10.f,
      .knockback_force = 10.f,
      .attack_range = {closeup_distance.x, body.attack_state.attack_range.y},
      .cooldown_duration = 1.0e9,
      .cooldown_timer = fan::time::timer(body.attack_state.cooldown_duration, true),
      .stun = false,
      .on_attack_start = [this]() {
        std::fill(hitbox_spawned.begin(), hitbox_spawned.end(), false);
      },
      .on_attack_end = [this]() {
        for (auto& hitbox : attack_hitboxes) {
          if (hitbox.is_valid()) {
            hitbox.destroy();
          }
        }
        std::fill(hitbox_spawned.begin(), hitbox_spawned.end(), false);
      }
    });
    ai_behavior.target = &pile->player.body;

    body.attack_state.on_attack_start = [this]() {
      std::fill(hitbox_spawned.begin(), hitbox_spawned.end(), false);
    };
    body.attack_state.on_attack_end = [this]() {
      for (auto& hitbox : attack_hitboxes) {
        if (hitbox.is_valid()) {
          hitbox.destroy();
        }
      }
      std::fill(hitbox_spawned.begin(), hitbox_spawned.end(), false);
      std::fill(hitbox_used.begin(), hitbox_used.end(), false);
    };
    attack_hitboxes.resize(std::size(attack_hitbox_frames));
    hitbox_spawned.resize(std::size(attack_hitbox_frames));
    hitbox_used.resize(std::size(attack_hitbox_frames));
    auto& level = pile->get_level();
    
    navigation.auto_jump_obstacles = true;
    navigation.jump_lookahead_tiles = 1.5f;

    navigation.on_check_obstacle = [this](const fan::vec2& check_pos) {
      return is_spike_at(check_pos);
    };
    physics_step_nr = fan::physics::add_physics_step_callback(
      [&]() { 
      fan::vec2 tile_size = pile->renderer.get_tile_size(level.main_map_id) * 2.f;
      fan::vec2 target_pos = pile->player.get_physics_pos();
      ai_behavior.update_ai(&body, navigation, target_pos, tile_size);
      fan::vec2 distance = ai_behavior.get_target_distance(body.get_physics_position());
      if (!( (std::abs(distance.x) < trigger_distance.x &&
        std::abs(distance.y) < trigger_distance.y))) {
        ai_behavior.enable_ai_patrol({initial_position - fan::vec2(400, 0), initial_position + fan::vec2(400, 0)});
      }
      else {
        ai_behavior.enable_ai_follow(&pile->player.body, trigger_distance, closeup_distance);
      }
    }
  );
  }
  void spawn_hitbox(int index) {
    f32_t direction = fan::math::sgn(body.get_tc_size().x);
    fan::vec2 center = body.get_center();
    fan::vec2 hitbox_offset = fan::vec2(50.f * direction, 0);
    attack_hitboxes[index] = pile->engine.physics_context.create_box(
      center + hitbox_offset,
      fan::vec2(60, 40),
      0,
      fan::physics::body_type_e::static_body,
      { .is_sensor = true }
    );
    hitbox_spawned[index] = true;
  }
  bool should_attack(fan::graphics::physics::character2d_t& c) {
    fan::vec2 distance = ai_behavior.get_target_distance(c.get_physics_position());
    return c.attack_state.try_attack(&c, distance);
  }
  bool update() {
    if (remove_this) {
      auto found = std::find_if(pile->entity.begin(), pile->entity.end(), [&](entity_t* e) { return e->body.NRI == body.NRI; });
      if (found == pile->entity.end()) {
        fan::throw_error("trying to remove non existing enemy");
      }
      delete *found;
      *found = nullptr;
      pile->entity.erase(found);
      return true;
    }
    if (body.attack_state.is_attacking) {
      for (int i = 0; i < std::size(attack_hitbox_frames); ++i) {
        if (!hitbox_spawned[i] && body.animation_on("attack0", attack_hitbox_frames[i])) {
          spawn_hitbox(i);
        }
      }
    }
    for (int i = 0; i < attack_hitboxes.size(); ++i) {
      if (hitbox_spawned[i] && attack_hitboxes[i].is_valid()) {
        if (!hitbox_used[i] && attack_hitboxes[i].test_overlap(pile->player.body)) {
          if (pile->player.on_hit(&body, (pile->player.body.get_position() - body.get_position()).normalized())) {
            return true; // if player dies, enemies respawn
          }
          hitbox_used[i] = true;
        }
      }
    }

    body.update_animations();

    render_health();
    return false;
  }

  void render_health() {
    int heart_count = body.get_max_health() / 10.f;
    for (int i = 0; i < heart_count; ++i) {
      fan::graphics::image_t hp_image = pile->get_gui().health_empty;
      //0-1
      f32_t progress = body.get_health() / body.get_max_health();
      if (progress * heart_count > i) {
        hp_image = pile->get_gui().health_full;
      }
      f32_t image_size = 8.f;
      fan::graphics::sprite({
        .position = fan::vec3(fan::vec2(body.get_physics_position() - fan::vec2(heart_count / 2.f * image_size - i * (image_size * 2.f) + image_size + image_size / 2.f, body.get_size().y / 1.5f)), 0xFF00), // force to be rendered on top
        .size = image_size, 
        .image = hp_image,
      });
    }
  }

  void destroy() {
    fan::physics::remove_physics_step_callback(physics_step_nr);
    remove_this = true;
  }
  bool on_hit(fan::graphics::physics::character2d_t* source, const fan::vec2& hit_direction) {
    fan::audio::play(player_hits_enemy);
    body.take_hit(source, hit_direction);
    if (body.is_dead()) {
      destroy();
      return true;
      //set_initial_position(initial_position);
//      body.health = body.max_health;
    }
    return false;
  }
  bool is_spike_at(const fan::vec2& pos) {
    for (auto& spike : pile->get_level().spike_sensors) {
      fan::vec2 spike_pos = spike.get_position();
      fan::vec2 spike_size = pile_t::level_t::spike_height * 2.f;
      if (std::abs(pos.x - spike_pos.x) < spike_size.x / 2.f &&
        std::abs(pos.y - spike_pos.y) < spike_size.y / 2.f) {
        return true;
      }
    }
    return false;
  }
  void respawn() {
    body.set_physics_position(initial_position);
    ai_behavior.enable_ai_patrol({initial_position - fan::vec2(400, 0), initial_position + fan::vec2(400, 0)});
    body.reset_health();
  }
  void set_initial_position(const fan::vec2& position) {
    initial_position = position;
    respawn();
  }

  fan::vec2 initial_position = 0;
  fan::graphics::physics::character2d_t body;
  fan::graphics::physics::ai_behavior_t ai_behavior;
  fan::graphics::physics::navigation_helper_t navigation;
  std::vector<fan::physics::entity_t> attack_hitboxes;
  std::vector<bool> hitbox_spawned;
  std::vector<bool> hitbox_used;
  bool remove_this = false;
  fan::audio::piece_t attack {"enemy_attack.sac"}, player_hits_enemy{"player_hits_enemy.sac"};
};