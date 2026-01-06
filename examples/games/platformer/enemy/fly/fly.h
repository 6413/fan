struct fly_t : enemy_t<fly_t> {
  using base_t = enemy_t<fly_t>;

  fly_t() = default;
  template<typename container_t>
  fly_t(container_t& bll, typename container_t::nr_t nr, const fan::vec3& position) {
    open(&bll, nr, "fly.json");

    set_initial_position(position);
    body.set_jump_height(body.get_jump_height() / 10.f);
    body.set_max_health(100);
    body.set_health(body.get_max_health());
    body.attack_state.cooldown_duration = 1.0e9;
    body.attack_state.cooldown_timer = fan::time::timer{body.attack_state.cooldown_duration, true};
    body.attack_state.knockback_force = 600.f;
    body.set_gravity_scale(0);
    body.attack_state.attack_range = {80, 25};
    body.attack_state.attack_requires_facing_target = false;
    body.attack_state.damage = 0;
    ai_behavior.trigger_distance = {500, 300};
    ai_behavior.closeup_distance = {75, 75};
    
    physics_step_nr = fan::physics::add_physics_step_callback([b = &bll, nr](){
      std::visit([nr](auto& node){
        if (node.body.get_health() <= 0) {
          return;
        }
        auto& level = pile->get_level();
        fan::vec2 tile_size = pile->renderer.get_tile_size(level.main_map_id) * 2.f;
        fan::vec2 target_pos = pile->player.get_physics_pos();
        node.ai_behavior.update_ai(&node.body, node.navigation, target_pos, tile_size);
        fan::vec2 distance = node.ai_behavior.get_target_distance(node.body.get_physics_position());
        if (node.ai_behavior.should_move(distance)) {
          node.ai_behavior.type = fan::graphics::physics::ai_behavior_t::behavior_type_e::none;
          fan::vec2 movement_direction = distance.sign();
          node.body.movement_state.move_to_direction_raw(node.body, movement_direction);
          //node.ai_behavior.enable_ai_follow(&pile->player.body, trigger_distance, closeup_distance);
        }
        else  {
          node.ai_behavior.enable_ai_patrol({node.initial_position - fan::vec2(400, 0), node.initial_position + fan::vec2(400, 0)});
        }
      }, (*b)[nr]);
    });
  }
  bool update() override {
    fan::vec2 distance = body.get_position() - pile->player.body.get_position();
    if (body.attack_state.try_attack(&pile->player.body, distance)) {
      pile->player.body.take_hit(&body, -distance.normalized());
      //pile->player.body.apply_linear_impulse_center(fan::vec2(-distance.normalized().x * body.attack_state.knockback_force, -
// body.attack_state.knockback_force / 5.f));
      body.attack_state.end_attack();
    }
    const std::string& anim_name = body.get_sprite_sheet_animation().name;
    if (body.get_health() <= 0 && anim_name != "die") {
      body.play_sprite_sheet_once("die");
      body.anim_controller.auto_update_animations = false;
    }
    if (destroy_this && body.get_health() <= 0 && anim_name == "die" && body.is_animation_finished()) {
      base_t::destroy();
      return true;
    }
    return base_t::update();
  }
  void destroy() override {
    destroy_this = true;
  }
  bool destroy_this = false;
};