struct boss_skeleton_t : enemy_t<boss_skeleton_t> {
  boss_skeleton_t() = default;
  template<typename container_t>
  boss_skeleton_t(container_t& bll, typename container_t::nr_t nr, const fan::vec3& position) {
    draw_offset = {0, -55};
    aabb_scale = 0.3f;
    attack_hitbox_frames = {4, 8};
    open(bll, nr, "boss_skeleton.json");
    set_initial_position(position);

    body.attack_state.attack_range = {300, 200};
    //physics_step_nr.~raii_nr_t();
  }
};