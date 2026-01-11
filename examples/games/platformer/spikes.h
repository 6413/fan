struct spike_spatial_t {
  static inline constexpr f32_t spike_height = 32.0f;
  static inline constexpr f32_t base_half_width = 32.0f;

  static inline constexpr std::array<fan::vec2, 3> get_spike_points(std::string_view dir) {
    static constexpr f32_t h = spike_height;
    static constexpr f32_t w = base_half_width;

    fan::vec2 a {0, -h}, b {-w, h}, c {w, h};
    if (dir == "down") {
      a.y = -a.y; b.y = -b.y; c.y = -c.y;
    }
    else if (dir == "left") {
      a = {h, 0}; b = {-h, -w}; c = {-h,  w};
    }
    else if (dir == "right") {
      a = {-h, 0}; b = {h, -w}; c = {h,  w};
    }
    return {{a, b, c}};
  }

  fan::vec2 world_min = 0;
  fan::vec2 cell_size = 256;
  fan::vec2i grid_size = {4096, 4096};

  std::unordered_map<uint32_t, std::vector<fan::physics::entity_t>> cells;

  void add(fan::physics::entity_t spike) {
    auto aabb = spike.get_aabb();
    auto minc = fan::graphics::spatial::world_to_cell_clamped(aabb.min, world_min, cell_size, grid_size);
    auto maxc = fan::graphics::spatial::world_to_cell_clamped(aabb.max, world_min, cell_size, grid_size);

    for (int y = minc.y; y <= maxc.y; ++y) {
      for (int x = minc.x; x <= maxc.x; ++x) {
        uint32_t idx = fan::graphics::spatial::cell_index({x, y}, grid_size);
        cells[idx].push_back(spike);
      }
    }
  }

  fan::physics::entity_t* query(fan::physics::entity_t& entity) {
    auto aabb = entity.get_aabb();
    auto minc = fan::graphics::spatial::world_to_cell_clamped(aabb.min, world_min, cell_size, grid_size);
    auto maxc = fan::graphics::spatial::world_to_cell_clamped(aabb.max, world_min, cell_size, grid_size);

    for (int y = minc.y; y <= maxc.y; ++y) {
      for (int x = minc.x; x <= maxc.x; ++x) {
        uint32_t idx = fan::graphics::spatial::cell_index({x, y}, grid_size);
        auto it = cells.find(idx);
        if (it == cells.end()) continue;

        for (auto& spike : it->second) {
          if (fan::physics::is_on_sensor(entity, spike)) {
            return &spike;
          }
        }
      }
    }
    return nullptr;
  }

  void clear() {
    cells.clear();
  }
};