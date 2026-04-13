struct player_t {
  fan::vec2 velocity = 0;
  fan::graphics::physics::character2d_t body = 
    fan::graphics::physics::character_circle_sprite(fan::vec3(1019.59f, 500.f, 10.f), 4.f);
  fan::graphics::light_t light;
  fan::graphics::sprite_sheet_controller_t animator;

  player_t() {
    body.set_draw_offset(fan::vec2(0, body.get_size().y / 1.5f));
    body.set_size(fan::vec2(8, 16));
    body.set_linear_damping(1500.f);
    light = fan::graphics::light_t(fan::graphics::light_properties_t{
      .position = body.get_position(), .size = 200.f, .color = fan::colors::white / 4.f
    });

    animator.load_animations(body, "npc/player");
  }

  void step() {
    if (pile.is_map_changing) return;
    animator.update(body, body.get_linear_velocity());

    fan::vec3 pos = body.get_position();
    light.set_position(pos);
    gloco()->camera_move_to_smooth(body);
    if (auto w = gui::window("A")){
      static f32_t v = 0.f;
      gui::drag(&v);
    float feet_y = pos.y + body.get_size().y / 2.f;
    pos.z = fan::graphics::get_depth_from_y(fan::vec2(pos.x, feet_y), 32.f * 3.f) + v;
    }
    body.set_position(pos);
  }
};