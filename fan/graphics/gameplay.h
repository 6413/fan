namespace fan::graphics::gameplay {
  struct checkpoint_system_t {
    struct checkpoint_data_t {
      fan::graphics::sprite_t visual;
      fan::physics::entity_t sensor;
      int index;
    };

    std::vector<checkpoint_data_t> checkpoints;
    int current_checkpoint = -1;

    void load_from_map(tilemap_renderer_t& renderer, tilemap_renderer_t::id_t map_id, auto visual_setup_cb) {
      clear();
      renderer.iterate_physics_entities(map_id, [&](auto& entity, auto& visual) {
        if (entity.id.contains("checkpoint")) {
          int idx = std::stoi(entity.id.substr(10));
          if (checkpoints.size() <= idx) checkpoints.resize(idx + 1);

          checkpoints[idx].sensor = visual;
          checkpoints[idx].index = idx;
          visual_setup_cb(checkpoints[idx].visual, visual);
        }
        return false;
      });
    }

    bool check_and_update(fan::physics::body_id_t player, auto on_checkpoint_cb) {
      for (auto& cp : checkpoints) {
        if (fan::physics::is_on_sensor(player, cp.sensor) && current_checkpoint < cp.index) {
          current_checkpoint = cp.index;
          on_checkpoint_cb(cp);
          return true;
        }
      }
      return false;
    }

    fan::vec3 get_respawn_position(int checkpoint) const {
      if (checkpoint == -1 || checkpoints.empty()) return 0;
      return checkpoints[checkpoint].visual.get_position();
    }
    fan::vec3 get_respawn_position() const {
      return get_respawn_position(current_checkpoint);
    }

    void set_checkpoint(int index) {
      if (index < 0) {
        current_checkpoint = -1;
        return;
      }

      if (index >= (int)checkpoints.size()) {
        current_checkpoint = checkpoints.size() - 1;
        return;
      }

      current_checkpoint = index;
    }

    void clear() {
      checkpoints.clear();
    }
  };
}