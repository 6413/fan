struct player_t {
  player_t() {
    body.set_size(fan::vec2(8, 16));
    body.set_draw_offset(fan::vec2(0, -body.get_size().y));
    body.set_linear_damping(1500.f);
    body.add_child(light);
    animator.load_animations(body, "npc/player");

    fan::vec2 sword_size = sword.get_image().get_size().normalize() * body.get_size().min() * 2.f;
    f32_t sword_scale = 0.8;
    sword.set_size(sword_size * sword_scale);
  }

  void step() {
    if (pile.is_map_changing) return;

    static fan::time::timer weapon_visible_timer = fan::time::seconds_timer(0.2);
    if (is_mouse_clicked() && weapon_visible_timer.finished()) {
      weapon_visible_timer.restart();
    }
    if (!weapon_visible_timer.finished()) {
      fan::vec2 df = animator.desired_facing;
      f32_t ang_45 = fan::math::radians(45.f);
      sword.set_attachment(body, df, (sword.get_angle().z != -ang_45) * 2 - 1, ang_45 + df.angle());
    }
    else {
      sword.set_position(fan::vec2(-0xfffff));
    }

    animator.update(body, body.get_linear_velocity());

    fan::vec3 pos = body.get_position();
    gloco()->camera_move_to_smooth(body);
    pos.z = pile.renderer.get_dynamic_depth(pile.active_map_id, pos, body.get_size().y) + 0.7f;
    body.set_position(pos);
  }

  fan::vec2 velocity = 0;
  fan::graphics::physics::character2d_t body = fan::graphics::physics::character_circle_sprite(fan::vec3(1019.59f, 500.f, 10.f), 4.f);
  static constexpr fan::vec2 light_size = 200.f;
  light_t light{{}, light_size, fan::colors::white / 4.f};
  sprite_sheet_controller_t animator;
  sprite_t sword{image_t("npc/player/sword.png", image_presets::pixel_art())};
};