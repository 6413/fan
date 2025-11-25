struct entity_t{
  static inline constexpr fan::vec2 draw_offset{0, -38};
  static inline constexpr f32_t aabb_scale = 0.19f;
  static constexpr fan::vec2 distance_when_to_move = {100, 50};

  bool attack_cb(fan::graphics::physics::character2d_t& c){
    static fan::time::timer attack_delay{0.5e9, true};
    static bool did_attack = false;
    if(did_attack){
      if(c.is_animation_finished()){
        did_attack = false;
      }
      else{
        return true;
      }
    }
    fan::vec2 d = get_player_distance();
    bool attack = attack_delay && d.x <= distance_when_to_move.x && 
      std::abs(d.y) < distance_when_to_move.y;
    if(attack){
      attack_delay.restart();
      did_attack = true;
    }
    return attack;
  }
  entity_t(const fan::vec3& pos = 0){
    body = fan::graphics::physics::character2d_t::from_json({
      .json_path = "player/player.json",
      .aabb_scale = aabb_scale,
      .draw_offset_override = draw_offset,
      .attack_cb = [this](auto& c){ return attack_cb(c); }
      });
    body.accelerate_force /= 2;
    body.set_color(fan::color(1, 1/3.f, 1/3.f, 1));
  }
  fan::vec2 get_center() const{
    return body.get_position() - draw_offset;
  }
  fan::vec2 get_physics_pos(){
    return body.get_physics_position();
  }
  fan::vec2 get_player_distance() const{
    return pile->player.get_center() - get_center();
  }
  void update(){
    fan::vec2 d = get_player_distance();
    auto tc = body.get_tc_size();
    fan::vec2 new_tc{std::copysign(std::abs(tc.x), d.x), tc.y};
    if(new_tc != tc){
      body.set_tc_size(new_tc);
    }
    if(std::abs(d.x) > distance_when_to_move.x && move && !is_stuck){
      body.move_to_direction(fan::vec2(d.square_normalize().x, 0));
      if(stuck_timer && d.y < -distance_when_to_move.y){
        body.perform_jump(true);
        stuck_timer.restart();
      }
    }
    if(prev_x == d.x && body.jumping){
      did_jump = true;
    }
    else if(!body.jumping){
      did_jump = false;
    }
    is_stuck = body.jumping && did_jump;
    if(prev_x != d.x){
      is_stuck = false;
    }
    prev_x = d.x;
    body.update_animations();
    auto& level = pile->get_level();
    fan::vec2 tile_size = pile->renderer.get_tile_size(level.main_map_id) * 2;
    tile_size.x = std::copysign(tile_size.x / 1.5f, new_tc.x);
    if(!fan::physics::is_point_overlapping(get_physics_pos() + tile_size) && !is_stuck){
      body.perform_jump(true);
    }
  }
  void on_hit(const fan::vec2& hit_direction){
    move = false;
    body.apply_linear_impulse_center(-hit_direction * 50);
    fan::event::after(500, [this]{
      move = true;
    });
  }

  fan::graphics::physics::character2d_t body;
  bool move = true;
  bool is_stuck = false;
  bool did_jump = false;
  f32_t prev_x = 0;
  fan::time::timer stuck_timer{0.1e9, true};
};