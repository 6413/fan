struct skeleton_t : enemy_t<skeleton_t> {
  skeleton_t() = default;
  template<typename container_t>
  skeleton_t(container_t& bll, typename container_t::nr_t nr, const fan::vec3& position) 
    : enemy_t<skeleton_t>(bll, nr, "skeleton.json", {0, -2}, 0.25f) {

    set_initial_position(position);
  }
};