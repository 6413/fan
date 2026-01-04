struct skeleton_t : enemy_t<skeleton_t> {
  skeleton_t() = default;
  template<typename container_t>
  skeleton_t(container_t& bll, typename container_t::nr_t nr, const fan::vec3& position) {
    draw_offset = {0, -2};
    aabb_scale = 0.25f;
    attack_hitbox_frames = {4, 8};
    body.set_max_health(1000);
    open(bll, nr, "skeleton.json");
    set_initial_position(position);
  }
};