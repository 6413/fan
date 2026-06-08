module;

module fan.graphics.voxel;

#if defined (FAN_WINDOW)

#if defined(FAN_3D)

import fan.graphics;

namespace fan::graphics {
  void voxel_world_t::set_generator(generator_t gen) {
    generator = std::move(gen);
  }

  void voxel_world_t::clear() {
    blocks.clear();
  }

  void voxel_world_t::update(const fan::vec3& camera_pos, int view_dist) {
    fan::vec3i center = (camera_pos / block_size).floor();
    if (center == last_center) { return; }
    last_center = center;

    static std::vector<fan::vec3i> to_remove;
    to_remove.clear();

    for (auto& entry : blocks.table) {
      if (entry.state != decltype(blocks)::state_t::occupied) { continue; }
      if (std::abs(entry.key.x - center.x) > view_dist ||
          std::abs(entry.key.z - center.z) > view_dist) {
        to_remove.push_back(entry.key);
      }
    }
    for (const auto& key : to_remove) { blocks.erase(key); }

    fan::vec3i c_min = center - view_dist;
    fan::vec3i c_max = center + view_dist;

    for (int x = c_min.x; x <= c_max.x; ++x) {
      for (int z = c_min.z; z <= c_max.z; ++z) {
        if (!generator) { continue; }
        auto props = generator(fan::vec3i{x, 0, z});
        if (!props) { continue; }
        fan::vec3i key{x, (int)(props->position.y / block_size), z};
        if (blocks.contains(key)) { continue; }
        blocks.try_emplace(key, fan::graphics::rectangle3d_t{*props});
      }
    }
  }

} // namespace fan::graphics

#endif

#endif