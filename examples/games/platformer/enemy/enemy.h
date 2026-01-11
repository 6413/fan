struct enemy_base_t {
  enemy_base_t() = default;
  virtual ~enemy_base_t() = default;
  virtual bool update() = 0;
  virtual void destroy() = 0;
  virtual bool on_hit(fan::graphics::physics::character2d_t* source, const fan::vec2& hit_direction) = 0;
  virtual void respawn() = 0;
  virtual void set_initial_position(const fan::vec2& position) = 0;
  virtual bool should_attack(fan::graphics::physics::character2d_t& c) = 0;
  virtual bool is_spike_at(const fan::vec2& pos) = 0;
  virtual void render_health() = 0;
  virtual fan::graphics::physics::character2d_t& get_body() = 0;
};

template<typename derived_t>
struct enemy_t : enemy_base_t {
  fan::vec2 draw_offset{0, -18};
  f32_t aabb_scale = 0.19f;
  fan::vec2 trigger_distance = {500, 150};
  fan::vec2 closeup_distance = {150, 100};
  //TODO use collision mask for player and entities
  std::vector<int> attack_hitbox_frames;
  f32_t density = 100.f;

  enemy_t() {}
  ~enemy_t() {
    attack_hitbox.cleanup(&body);
  }

  template<typename container_t>
  void open(container_t* bll, typename container_t::nr_t nr, const std::string& path, const std::source_location& caller_path = std::source_location::current()) {
    body = fan::graphics::physics::character2d_t::from_json({
      .json_path = path,
      .aabb_scale = aabb_scale,
      .attack_cb = [bll, nr](auto& c){ 
        return std::visit([&c](auto& e) { return e.should_attack(c); }, (*bll)[nr]); 
      },
      .physics_properties={.density=density, .fixed_rotation=true, .linear_damping=2.0f}
    }, caller_path);

    body.set_draw_offset(draw_offset);
    body.set_dynamic();

    body.set_jump_height(75.f * density);
    body.movement_state.accelerate_force = 120.f / 3.f;
    body.movement_state.max_speed = 500.f;
    body.movement_state.check_gui = false;
    ai_behavior.trigger_distance = trigger_distance;
    ai_behavior.closeup_distance = closeup_distance;

    if (attack_hitbox_frames.size()) {
      std::vector<fan::graphics::physics::attack_hitbox_t::hitbox_spawn_t> spawns;
  
      for (size_t i = 0; i < attack_hitbox_frames.size(); ++i) {
        spawns.push_back({
          .frame = attack_hitbox_frames[i],
          .create_hitbox = [](const fan::vec2& center, f32_t direction){
            fan::vec2 offset = fan::vec2(50.f * direction, 0);
            return pile->engine.physics_context.create_box(
              center + offset, fan::vec2(60, 40), 0,
              fan::physics::body_type_e::static_body, {.is_sensor = true}
            );
          }
        });
      }
  
      attack_hitbox.setup({
        .spawns = spawns,
        .attack_animation = "attack0",
        .track_hit_targets = false
      });
    }

    body.setup_attack_properties({
      .damage = 10.f,
      .knockback_force = 5.f,
      .attack_range = {closeup_distance.x, body.attack_state.attack_range.y},
      .cooldown_duration = 2.0e9,
      .cooldown_timer = fan::time::timer(body.attack_state.cooldown_duration, true),
      .stun = false
    });
    ai_behavior.target = &pile->player.body;

    auto& level = pile->get_level();

    navigation.auto_jump_obstacles = true;
    navigation.jump_lookahead_tiles = 1.5f;
    navigation.on_check_obstacle = [bll, nr](const fan::vec2& check_pos){
      return false;
      //return std::visit([&check_pos](auto& e) { return e.is_spike_at(check_pos); }, (*bll)[nr]);
    };
    physics_step_nr = fan::physics::add_physics_step_callback([bll, nr](){
      std::visit([](auto& node){
        if (node.body.get_health() <= 0) {
          return;
        }
        node.attack_hitbox.update(&node.body);
        auto& level = pile->get_level();
        fan::vec2 tile_size = pile->renderer.get_tile_size(level.main_map_id) * 2.f;
        fan::vec2 target_pos = pile->player.get_physics_pos();
        node.ai_behavior.update_ai(&node.body, node.navigation, target_pos, tile_size);
        fan::vec2 distance = node.ai_behavior.get_target_distance(node.body.get_physics_position());
        if (!((std::abs(distance.x) < node.trigger_distance.x && std::abs(distance.y) < node.trigger_distance.y))) {
          node.ai_behavior.enable_ai_patrol({node.initial_position - fan::vec2(400, 0), node.initial_position + fan::vec2(400, 0)});
        }
        else if (node.body.raycast(pile->player.body)){
          node.ai_behavior.enable_ai_follow(&pile->player.body, node.trigger_distance, node.closeup_distance);
        }
        else {
          node.ai_behavior.enable_ai_patrol({node.initial_position - fan::vec2(400, 0), node.initial_position + fan::vec2(400, 0)});
        }
      }, (*bll)[nr]);
    });
    {
      hearts.clear();

      int heart_count = body.get_max_health() / 10.f;
      hearts.reserve(heart_count);

      f32_t image_size = 8.f;

      for (int i = 0; i < heart_count; ++i) {
        hearts.emplace_back(fan::graphics::unlit_sprite_t({
          .position = fan::vec3(0, 0, 0xFFF0),
          .size = image_size,
          .image = pile->get_gui().health_empty
        }));
        hearts.back().set_dynamic();
      }
    }
  }
  bool should_attack(fan::graphics::physics::character2d_t& c) override {
    fan::vec2 distance = ai_behavior.get_target_distance(c.get_physics_position());
    return c.attack_state.try_attack(&c, distance) && std::abs(c.get_linear_velocity().y) < 10.f;
  }
  bool base_update() {
    for (int i = 0; i < attack_hitbox.hitbox_count(); ++i) {
      if (attack_hitbox.check_hit(&body, i, &pile->player.body)) {
        if (pile->player.on_hit(&body, (pile->player.body.get_position() - body.get_position()).normalized())) {
          return true;
        }
      }
    }
    attack_hitbox.process_destruction();
    if (body.get_health() > 0) {
      body.update_animations();
    }
    render_health();
    return false;
  }
  bool update() override {
    return base_update();
  }
  void render_health() override {
    int heart_count = hearts.size();
    if (!heart_count) {
      return;
    }

    f32_t progress = body.get_health() / body.get_max_health();
    f32_t half_size = hearts[0].get_size().x;
    f32_t full_size = half_size * 2.f;       

    fan::vec2 base_pos = body.get_physics_position();
    base_pos.y -= body.get_size().y / 1.5f;

    f32_t total_width = heart_count * full_size;
    f32_t start_x = -total_width * 0.5f + full_size * 0.5f;

    for (int i = 0; i < heart_count; ++i) {
      if (progress * heart_count > i) {
        hearts[i].set_image(pile->get_gui().health_full);
      }
      else {
        hearts[i].set_image(pile->get_gui().health_empty);
      }

      f32_t x_offset = start_x + i * full_size;
      hearts[i].set_position(fan::vec2(base_pos.offset_x(x_offset)));
    }
  }

  void destroy() override {
    for (auto [nr, enemy] : fan::enumerate(pile->enemies())) {
      if (enemy.get_body().NRI == body.NRI) {
        pile->enemy_list.remove(nr);
        break;
      }
    }
  }
  bool on_hit(fan::graphics::physics::character2d_t* source, const fan::vec2& hit_direction) override {
    audio_player_hits_enemy.play();
    body.take_hit(source, hit_direction);
    if (body.is_dead()) {
      static constexpr f32_t drop_chance = 0.33f;

      if (fan::random::value(0.0f, 1.0f) < drop_chance) {
        fan::vec2 tile_size = pile->tilemaps_compiled[pile->get_level().stage_name].tile_size;
        fan::vec3i drop_pos = (body.get_center() / tile_size).floor() * tile_size;

        struct drop_t { const char* name; const char* texture; };
        static constexpr drop_t drops[] = {
          {"pickupable_health", "tile1217"},
          {"pickupable_health_potion", "tile1218"}
        };

        constexpr size_t count = std::size(drops);
        size_t index = (size_t)(fan::random::value(0.0f, 1.0f) * count);
        auto sensor = fan::physics::create_sensor_rectangle(
          drop_pos,
          tile_size / 1.2f
        );

        size_t idx = pile->get_level().pickupable_spatial.add(
          drops[index].name,
          sensor
        );

        pile->get_level().dropped_pickupables[drop_pos] = fan::graphics::sprite_t {{
          .position = drop_pos,
          .size = tile_size,
          .texture_pack_unique_id = pile->engine.texture_pack[drops[index].texture]
        }};
      }
      destroy();
      return true;
    }
    return false;
  }
  bool is_spike_at(const fan::vec2& pos) override {
    for (auto& spike : pile->get_level().spike_sensors) {
      fan::vec2 spike_pos = spike.get_position();
      fan::vec2 spike_size = pile_t::level0_t::spike_spatial_t::spike_height * 2.f;
      if (std::abs(pos.x - spike_pos.x) < spike_size.x / 2.f && std::abs(pos.y - spike_pos.y) < spike_size.y / 2.f) {
        return true;
      }
    }
    return false;
  }
  void respawn() override {
    body.set_physics_position(fan::vec3(initial_position, 5));
    ai_behavior.enable_ai_patrol({initial_position - fan::vec2(400, 0), initial_position + fan::vec2(400, 0)});
    body.reset_health();
  }
  void set_initial_position(const fan::vec2& position) override {
    initial_position = position;
    respawn();
  }
  fan::graphics::physics::character2d_t& get_body() override {
    return body;
  }

  fan::graphics::physics::character2d_t body;
  fan::graphics::physics::attack_hitbox_t attack_hitbox;
  fan::graphics::physics::ai_behavior_t ai_behavior;
  fan::graphics::physics::navigation_helper_t navigation;
  std::vector<fan::graphics::unlit_sprite_t> hearts;
  fan::vec2 initial_position = 0;
  fan::physics::step_callback_nr_t physics_step_nr;
  fan::audio::sound_t 
    audio_attack{"audio/enemy_attack.sac"}, 
    audio_player_hits_enemy{"audio/player_hits_enemy.sac"}
  ;
};