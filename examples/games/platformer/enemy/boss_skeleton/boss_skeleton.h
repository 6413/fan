struct boss_skeleton_t : boss_t<boss_skeleton_t> {
  boss_skeleton_t() = default;
  template<typename container_t>
  boss_skeleton_t(container_t& bll, typename container_t::nr_t nr, const fan::vec3& position) {
    draw_offset = {0, -55};
    aabb_scale = 0.3f;
    attack_hitbox_frames = {4};
    closeup_distance.x = 400;
    trigger_distance.x = 10000;
    open(bll, nr, "boss_skeleton.json");
    set_initial_position(position);
    body.set_max_health(300.f);
    body.set_health(body.get_max_health());

    body.attack_state.attack_range = {closeup_distance.x + 10, 200};

    attack_hitbox.setup({
      .spawns = {{
        .frame = attack_hitbox_frames[0],
        .create_hitbox = [range = body.attack_state.attack_range.x](const fan::vec2& center, f32_t direction){
          fan::vec2 offset = fan::vec2(range * direction, 0);
          return pile->engine.physics_context.create_box(
            center + offset, fan::vec2(60, 60), 0,
            fan::physics::body_type_e::static_body, {.is_sensor = true}
          );
        }}
      },
      .attack_animation = "attack0",
      .track_hit_targets = false
    });

    name = "Skeleton Lord";

    physics_step_nr = fan::physics::add_physics_step_callback([&bll, nr, xdist = closeup_distance.x](){
      auto& level = pile->get_level();
      fan::vec2 tile_size = pile->renderer.get_tile_size(level.main_map_id) * 2.f;
      fan::vec2 target_pos = pile->player.get_physics_pos();
      std::visit([&, xdist](auto& node){
        node.ai_behavior.update_ai(&node.body, node.navigation, target_pos, tile_size);
        fan::vec2 distance = node.ai_behavior.get_target_distance(node.body.get_physics_position());
        if (node.body.raycast(pile->player.body) && !node.body.attack_state.is_attacking){
          node.ai_behavior.type = fan::graphics::physics::ai_behavior_t::behavior_type_e::none;
          fan::vec2 movement_direction{0, 0};

          float ax = std::abs(distance.x);
          if (ax > xdist + 5.f) {
            movement_direction.x = (distance.x > 0) ? 1.f : -1.f;
          }
          else if (ax < xdist - 5.f) {
            movement_direction.x = (distance.x > 0) ? -1.f : 1.f;
          }
          node.body.movement_state.move_to_direction_raw(node.body, movement_direction);
        }
      }, bll[nr]);
    });
  }
};