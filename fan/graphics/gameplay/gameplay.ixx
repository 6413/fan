module;

#include <vector>
#include <string>
#include <unordered_map>

export module fan.graphics.gameplay;

import fan.graphics;
import fan.physics.b2_integration;
import fan.graphics.physics_shapes;
import fan.graphics.gui.tilemap_editor.renderer;
import fan.graphics.spatial;

export namespace fan::graphics::gameplay {
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

    fan::vec3 get_respawn_position(tilemap_renderer_t& renderer, tilemap_renderer_t::id_t map_id, int checkpoint) const {
      if (checkpoint == -1 || checkpoints.empty()) {
        fan::vec3 p = 0;
        p = renderer.get_spawn_position(map_id);
        return p;
      }
      return checkpoints[checkpoint].visual.get_position();
    }
    fan::vec3 get_respawn_position(tilemap_renderer_t& renderer, tilemap_renderer_t::id_t map_id) const {
      return get_respawn_position(renderer, map_id, current_checkpoint);
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

  struct pickupable_spatial_t {
    struct pickupable_data_t {
      std::string id;
      fan::physics::body_id_t sensor;
    };

    fan::graphics::spatial::dynamic_grid_t<size_t> grid;
    fan::graphics::spatial::registry_t<size_t> registry;
    std::unordered_map<size_t, pickupable_data_t> pickupables;
    size_t next_id = 0;

    void init(const fan::vec2& world_min, const fan::vec2& world_size, f32_t cell_size = 128.f) {
      fan::vec2i grid_size = (world_size / cell_size).ceil();
      fan::graphics::spatial::dynamic_grid_init(grid, world_min, fan::vec2(cell_size), grid_size);
    }

    size_t add(const std::string& id, fan::physics::body_id_t sensor) {
      size_t idx = next_id++;

      fan::vec2 pos = sensor.get_position();
      fan::vec2 size = sensor.get_size();
      fan::physics::aabb_t aabb {pos - size, pos + size};

      fan::graphics::spatial::static_grid_t<size_t> dummy_static;
      fan::graphics::spatial::add_object(
        registry,
        dummy_static,
        grid,
        idx,
        aabb,
        fan::graphics::spatial::movement_dynamic
      );

      pickupables[idx] = {id, sensor};
      return idx;
    }

    void remove(size_t id) {
      auto it = pickupables.find(id);
      if (it == pickupables.end()) {
        return;
      }

      fan::graphics::spatial::static_grid_t<size_t> dummy_static;

      fan::graphics::spatial::remove_object(
        registry,
        dummy_static,
        grid,
        id
      );

      it->second.sensor.destroy();
      pickupables.erase(it);
    }

    std::vector<size_t> query_radius(const fan::vec2& center, f32_t radius) {
      std::vector<size_t> result;
      fan::graphics::spatial::query_radius(grid, center, radius, [&](size_t idx) {
        result.push_back(idx);
      });
      return result;
    }

    pickupable_data_t* get(size_t id) {
      auto it = pickupables.find(id);
      if (it == pickupables.end()) {
        return nullptr;
      }
      return &it->second;
    }

    void clear() {
      for (auto& [id, data] : pickupables) {
        data.sensor.destroy();
      }

      pickupables.clear();
      pickupables.rehash(0);

      grid.objects.clear();
      grid.cells.clear();
      grid.id_to_object.clear();

      registry.id_to_dynamic.clear();
      registry.id_to_dynamic.rehash(0);

      registry.id_to_movement.clear();
      registry.id_to_movement.rehash(0);

      registry.aabb_cache.clear();
      registry.aabb_cache.rehash(0);

      next_id = 0;
    }
  };
}