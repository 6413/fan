struct skeleton_t : enemy_t {
  skeleton_t(const fan::vec3& position) : enemy_t("skeleton.json") {
    set_initial_position(position);
  }
};