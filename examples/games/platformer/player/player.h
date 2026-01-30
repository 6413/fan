#define gui fan::graphics::gui
#define gameplay fan::graphics::gameplay
struct player_t {
  static inline constexpr fan::vec2 draw_offset {0.f, -42.5f};
  static inline constexpr f32_t aabb_scale = 0.17f;
  static inline constexpr f32_t sword_length = 120.f;
  static inline constexpr int attack_hitbox_frame = 3;

  static inline constexpr std::array<fan::vec2, 3> get_hitbox_points(f32_t direction) {
    return {{
      {sword_length * direction, 0.0f},
      {0.0f, -20.0f},
      {0.0f, 20.0f}
    }};
  }

  player_t() {
    player_light.set_dynamic();
    player_light.set_color(fan::color(0.610, 0.550, 0.340, 1.0) * 5.f);
    player_light.set_size(256);
    potion_particles.from_json("effects/drink_potion.json");

    auto image_star = pile->engine.image_load("images/waterdrop.webp");
    particles = fan::graphics::shape_from_json("effects/explosion.json");
    particles.set_image(image_star);
    body = fan::graphics::physics::character2d_t::from_json({
      .json_path = "player/player.json",
      .aabb_scale = aabb_scale,
      .attack_cb = [this](fan::graphics::physics::character2d_t& c) -> bool {
        if (is_blocking()) {
            return false;
        }

        const bool attack_input =
          fan::window::is_input_action_active(fan::actions::light_attack) ||
          fan::window::is_key_pressed(fan::gamepad_right_bumper)
        ;

        bool attack_pressed = attack_input;

        if (!attack_pressed || gui::want_io()) {
            return false;
        }

        return c.attack_state.try_attack(&c);
      },
    });

    body.set_draw_offset(draw_offset);
    body.set_flags(fan::graphics::sprite_flags_e::use_hsl);

    body.set_color(fan::color::hsl(56.7f, 18.3f, -58.4f));
    //body.set_dynamic();
    /*
      template <typename T>
      typename T::vi_t& get_shape_vdata() {
        return *(typename T::vi_t*)GetRenderData(g_shapes->shaper);
      }
      template <typename T>
      typename T::ri_t& get_shape_rdata() {
        return *(typename T::ri_t*)GetData(g_shapes->shaper);
      }
    */


    fan::graphics::physics::character_movement_preset_t::setup_default_controls(body);

    body.movement_state.jump_state.on_jump = [&](int jump_type) {
      task_jump = jump(jump_type == 1);
    };

    combat.hitbox.setup({
      .spawns = {{
        .frame = attack_hitbox_frame,
        .create_hitbox = [](const fan::vec2& center, f32_t direction) {
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
      .cooldown_timer = fan::time::timer{0.05e9, true},
      .on_attack_end = [this]() { combat.did_attack = false; },
      });

    body.set_restitution(0);

    physics_step_nr = fan::physics::add_physics_step_callback([this]() {
      handle_attack();
    });

    sprite_shield.set_position(body.get_center());
    body.anim_controller.auto_update_animations = false;
    body.anim_controller.auto_flip_sprite = true;

  }

  fan::event::task_t jump(bool is_double_jump) {
    //audio_jump.play();
    audio_attack.play();
    body.set_angle(0.f);
    sprite_shield.set_angle(0.f);
    if (!is_double_jump) {
      co_return;
    }
    body.set_rotation_point(-body.get_draw_offset());
    fan::time::timer jump_timer {1.0e9f / 2.f, true};
    while (!jump_timer) {
      f32_t progress = jump_timer.seconds() / jump_timer.duration_seconds();
      f32_t angle = progress * fan::math::two_pi * body.get_image_sign().x;
      body.set_angle(fan::vec3(0, 0, angle));
      sprite_shield.set_angle(fan::vec3(0, 0, angle));
      co_await fan::graphics::co_next_frame();
    }
    body.set_angle(0.f);
  }

  void respawn() {
    if (!checkpoint_position) {
      checkpoint_position = pile->checkpoint_system.get_respawn_position(pile->renderer, pile->get_level().main_map_id);
      if (!checkpoint_position) {
        fan::throw_error("no checkpoint found");
      }
    }

    body.set_physics_position(checkpoint_position);
    body.set_linear_velocity(fan::vec2(0));
    body.set_angular_velocity(0);

    f32_t max_health = body.get_max_health();
    body.set_health(max_health);
    particles.set_position(body.get_position());
    pile->get_level().load_enemies();

    static bool once = true;
    if (once) {
      auto& lgui = pile->get_gui();
      auto potion = gameplay::items::create(items::id_e::health_potion);
      lgui.inventory.add_item(potion, 5);
      auto shield = gameplay::items::create(items::id_e::iron_shield);
      lgui.inventory.add_item(shield, 1);
      once = false;
    }
  }

  void handle_attack() {
    if (!combat.did_attack && body.sprite_sheet_crossed("attack0", attack_hitbox_frame)) {
      audio_attack.play();
      combat.did_attack = true;
    }

    combat.handle_attack(body, pile->enemies());
  }

  void drink_potion() {
    if (!potion_consume_timer) {
      return;
    }

    auto& inv = pile->get_gui().inventory;

    if (!inv.remove_item(items::id_e::health_potion, 1)) {
      return;
    }

    auto& registry = gameplay::items::get_registry();
    auto* def = registry.get_definition(items::id_e::health_potion);

    f32_t new_health = std::min(
      body.get_health() + def->effects.front().value,
      body.get_max_health()
    );
    body.set_health(new_health);

    audio_drink_potion.play();

    fan::vec3 player_pos = body.get_center() - fan::vec2(0, body.get_size().y / 4.f);
    potion_particles.spawn_at(fan::vec3(fan::vec2(player_pos), player_pos.z + 1));

    potion_consume_timer.restart();
  }

  void use_item(const gameplay::item_t& item) {
    switch (item.id) {
    case items::id_e::health_potion:
      drink_potion();
      break;

    case items::id_e::mana_potion:
      // drink_mana_potion();
      break;

    default:
      break;
    }
  }


  bool is_blocking() const {
    return fan::window::is_input_down(fan::actions::block_attack) && pile->get_gui().equipment.has_item(items::id_e::iron_shield);
  }

  void process_hotbar() {
    if (fan::window::is_input_action_active(actions::drink_potion)) {
      auto& hotbar = pile->get_gui().hotbar;
      auto& slot = hotbar.slots[hotbar.selected_slot];

      hotbar.consume_slot(hotbar.selected_slot, hotbar.on_item_use);
      /*else inventory*/
    }
  }

  void update() {
    fan::print(body.get_flags());
    combat.hitbox.process_destruction();

    process_hotbar();

    if (body.is_on_ground() || body.movement_state.is_wall_sliding) {
      if (body.get_angle().z != 0.f) {
        body.set_angle(0.f);
      }
      sprite_shield.set_angle(0.f);
    }

    player_light.set_position(body.get_center());

    bool blocking = is_blocking();

    if (fan::window::is_input_released(fan::actions::block_attack)) {
      body.cancel_animation();
    }

    if (blocking) {
      fan::vec2 input_vector = fan::window::get_input_vector();
      if (input_vector.x != 0) {
        fan::vec2 sign = body.get_image_sign();
        int desired_sign = fan::math::sgn(input_vector.x);
        if (fan::math::sgn(sign.x) != desired_sign) {
         // body.set_image_sign(fan::vec2(desired_sign, sign.y));
        }
      }

      fan::vec2 sign = body.get_image_sign();
      f32_t facing_direction = sign.x;

      fan::vec2 shield_offset = fan::vec2(body.get_size().x * facing_direction, 0);
      fan::vec3 shield_pos = body.get_center().offset_z(1) + shield_offset;

      sprite_shield.set_position(shield_pos);
      sprite_shield.set_rotation_point(body.get_center() - shield_pos);
      sprite_shield.set_tc_size(fan::vec2(facing_direction, 1.0f));

      if (!body.attack_state.is_attacking) {
        body.cancel_animation();
        body.set_sprite_sheet("attack0");
        body.set_current_sprite_sheet_frame(3);
      }

      body.movement_state.max_speed = max_player_speed / 3.f;
    }
    else {
      body.movement_state.max_speed = max_player_speed;
      sprite_shield.set_position(fan::vec2(-0xfffff));
      body.update_animations();
    }

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
      gui::print("Checkpoint reached!");
      checkpoint_position = pile->checkpoint_system.get_respawn_position(pile->renderer, pile->get_level().main_map_id);
    });

    auto& map_compiled = pile->tilemaps_compiled[pile->get_level().stage_name];
    if (get_physics_pos().y > map_compiled.map_size.y * (map_compiled.tile_size.y * 2.f)) {
      respawn();
    }
  }

  fan::vec2 get_physics_pos() {
    return body.get_physics_position();
  }

  int on_hit(fan::graphics::physics::character2d_t* source, const fan::vec2& hit_direction) {
    if (is_blocking()) {
      fan::vec2 sign = body.get_image_sign();
      f32_t shield_direction = sign.x;
      f32_t attack_direction = fan::math::sgn(hit_direction.x);

      // check opposite signs
      if (shield_direction != attack_direction) {
        // give stun
        body.take_knockback(source, hit_direction);
        audio_shield_block.play();
        return attack_result_e::blocked;
      }
    }

    audio_enemy_hits_player.play();
    body.take_hit(source, hit_direction);
    if (body.get_health() <= 0) {
      body.cancel_animation();
      pile->get_level().reload_map();
      return attack_result_e::hit;
    }
    return attack_result_e::miss;
  }

  fan::graphics::physics::character2d_t& get_body() {
    return body;
  }

  fan::graphics::physics::character2d_t body;
  fan::graphics::physics::combat_controller_t combat;
  fan::event::task_t task_jump;
  fan::audio::sound_t
    audio_jump {"audio/jump.sac"},
    audio_attack {"audio/player_attack.sac"},
    audio_enemy_hits_player {"audio/enemy_hits_player.sac"},
    audio_checkpoint {"audio/checkpoint.sac"},
    audio_drink_potion {"audio/drink_potion.sac"},
    audio_shield_block {"audio/shield_block.sac"}
  ;
  fan::graphics::shape_t particles;
  fan::event::task_t task_particles;
  fan::time::timer potion_consume_timer {0.1e9, true};
  fan::graphics::effects::particle_pool_t::pool_t<4> potion_particles;
  fan::graphics::light_t player_light {{.position = 0}};
  fan::physics::step_callback_nr_t physics_step_nr;
  fan::graphics::sprite_t sprite_shield {{
    .image = "images/shield.webp"
  }};
  fan::vec2 checkpoint_position = 0;
  f32_t max_player_speed = 600.f;
};
#undef gui
#undef gameplay