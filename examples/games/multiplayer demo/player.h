struct player_t {

  player_t() {
    // init client animations
    fan::json json_data = fan::graphics::read_json("entities/player/player.json");
    gloco()->parse_animations(json_data);
    fan::graphics::map_animations(anims);
    body.set_shape(fan::graphics::extract_single_shape(json_data));
    body.set_size(body.get_size() / 2.5);
    body.play_sprite_sheet();
    
    body.jump_impulse = 3.5;
    body.force = 15;
    body.max_speed = 3000;
  }

  void step() {
    body.process_movement(fan::graphics::physics::character2d_t::movement_e::side_view);
    body.update_animation();
  }

  fan::graphics::physics::character2d_t body{ 
    fan::graphics::physics::capsule_sprite_t{{
      .position =  fan::vec3(fan::vec2(109, 123) * 64, 0xffff - 10),
      .center0 = {0.f, -24.f},
      .center1 = {0.f, 32.f},
      .radius = 12,
      .blending = true,
      .body_type = fan::physics::body_type_e::dynamic_body,
      .shape_properties{
        .friction = 0.6f,
        .density = 0.1f,
        .fixed_rotation = true,
      },
    }
  }};
  fan::graphics::shape_t light;
  fan::graphics::animator_t animator;
};