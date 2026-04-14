struct player_t {
  player_t() {
    body.set_draw_offset(fan::vec2(0, body.get_size().y / 1.5f));
    body.set_size(fan::vec2(8, 16));
    body.set_linear_damping(1500.f);
    body.add_child(light);
    animator.load_animations(body, "npc/player");
  }

  void step() {
    if (pile.is_map_changing) {
      return; 
    }

    animator.update(body, body.get_linear_velocity());

    fan::vec3 pos = body.get_position();
    gloco()->camera_move_to_smooth(body);
  
    pos.z = pile.renderer.get_dynamic_depth(pile.active_map_id, pos.xy(), body.get_size().y);
    body.set_position(pos);
  }

  fan::vec2 velocity = 0;
  fan::graphics::physics::character2d_t body = fan::graphics::physics::character_circle_sprite(fan::vec3(1019.59f, 500.f, 10.f), 4.f);
  fan::graphics::light_t light{{}, 200.f, fan::colors::white / 4.f};
  fan::graphics::sprite_sheet_controller_t animator;
};