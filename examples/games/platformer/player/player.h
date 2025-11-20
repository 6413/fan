//TODO use collision mask for player and entities
struct player_t {
  static inline constexpr fan::vec2 draw_offset{ 0.f, -38.f };
  player_t() {
    body = fan::graphics::physics::character2d_t::from_json({
      .json_path = "player.json",
      .aabb_scale = 0.19f,
      .draw_offset_override = draw_offset
    });
    mouse_click_handle = pile->engine.on_mouse_click(fan::mouse_left, [this](const auto& bdata) {
      task_attack = attack();
    });
    //body.get_animation("attack0")->fps = 3;
    //body.anim_controller.states.find("attack0")->second.fps = 3;
  }

  void step() {
   // fan::graphics::circle(body.get_position() - draw_offset, 32, fan::colors::white);
    body.update_animations();
  }

  fan::event::task_t attack() {
    co_await fan::graphics::animation_frame_awaiter(&body, "attack0", 4);
    fan::vec2 pts[3] = {
      { 100.0f * fan::math::sgn(body.get_tc_size().x), 0.0f},
      { -0.0f, -10.0f },
      { -0.0f, 10.0f }
    };
    fan::physics::entity_t hitbox = fan::physics::gphysics->create_polygon(
      body.get_position() - draw_offset,
      0.0f,
      pts,
      3,
      fan::physics::body_type_e::static_body,
      { .is_sensor = true }
    );
    
    if (hitbox.test_overlap(pile->entity.body)) {
      pile->entity.on_hit((body.get_position() - pile->entity.body.get_position()).normalized());
    }

    hitbox.destroy();
  }

  fan::graphics::physics::character2d_t body;
  fan::graphics::engine_t::buttons_handle_t mouse_click_handle;
  fan::event::task_t task_attack;
};