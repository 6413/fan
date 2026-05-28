module;

#if defined(FAN_PHYSICS_2D)
#endif

export module fan.graphics.gameplay;

import std;

#if defined(FAN_PHYSICS_2D)

import fan.types;
import fan.types.vector;
import fan.math;
import fan.graphics.common_context;
import fan.graphics.shapes;
import fan.physics.types;
import fan.graphics.tilemap_editor.renderer;
import fan.spatial;
import fan.graphics.physics_shapes;
import fan.graphics;

import fan.physics.b2_integration;

export namespace fan::graphics::gameplay {
  struct checkpoint_system_t {
    struct checkpoint_data_t {
      fan::graphics::shape_t visual;
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
        p = renderer.get_spawn(map_id);
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

    fan::spatial::dynamic_grid_t<std::size_t> grid;
    fan::spatial::registry_t<std::size_t> registry;
    std::unordered_map<std::size_t, pickupable_data_t> pickupables;
    std::size_t next_id = 0;

    void init(const fan::vec2& world_min, const fan::vec2& world_size, f32_t cell_size = 128.f) {
      fan::vec2i grid_size = (world_size / cell_size).ceil();
      fan::spatial::dynamic_grid_init(grid, world_min, fan::vec2(cell_size), grid_size);
    }

    std::size_t add(const std::string& id, fan::physics::body_id_t sensor) {
      std::size_t idx = next_id++;

      fan::vec2 pos = sensor.get_position();
      fan::vec2 size = sensor.get_size();
      fan::physics::aabb_t aabb {pos - size, pos + size};

      fan::spatial::static_grid_t<std::size_t> dummy_static;
      fan::spatial::add_object(
        registry,
        dummy_static,
        grid,
        idx,
        aabb,
        fan::spatial::movement_dynamic
      );

      pickupables[idx] = {id, sensor};
      return idx;
    }

    void remove(std::size_t id) {
      auto it = pickupables.find(id);
      if (it == pickupables.end()) {
        return;
      }

      fan::spatial::static_grid_t<std::size_t> dummy_static;

      fan::spatial::remove_object(
        registry,
        dummy_static,
        grid,
        id
      );

      it->second.sensor.destroy();
      pickupables.erase(it);
    }

    std::vector<std::size_t> query_radius(const fan::vec2& center, f32_t radius) {
      std::vector<std::size_t> result;
      fan::spatial::query_radius(grid, center, radius, [&](std::size_t idx) {
        result.push_back(idx);
      });
      return result;
    }

    pickupable_data_t* get(std::size_t id) {
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

  struct spikes_t {
    static inline constexpr std::array<fan::vec2, 3> get_spike_points(fan::vec2 size, std::string_view dir) {
      f32_t w = size.x;
      f32_t h = size.y;

      fan::vec2 a {0, -h}, b {-w, h}, c {w, h};
      if (dir == "down") {
        a.y = -a.y; b.y = -b.y; c.y = -c.y;
      }
      else if (dir == "left") {
        a = {h, 0}; b = {-h, -w}; c = {-h, w};
      }
      else if (dir == "right") {
        a = {-h, 0}; b = {h, -w}; c = {h, w};
      }
      return {{a, b, c}};
    }

    struct spike_t {
      fan::graphics::sprite_t visual;
      fan::physics::entity_t sensor;
    };

    fan::vec2 world_min = 0;
    fan::vec2 cell_size = 256;
    fan::vec2i grid_size = {4096, 4096};

    std::unordered_map<std::uint32_t, std::vector<fan::physics::entity_t>> cells;
    std::vector<spike_t> spikes;
    fan::graphics::image_t img {"images/gameplay/spike.webp"};

    void add_to_spatial(fan::physics::entity_t spike) {
      auto aabb = spike.get_aabb();
      auto minc = fan::spatial::world_to_cell_clamped(aabb.min, world_min, cell_size, grid_size);
      auto maxc = fan::spatial::world_to_cell_clamped(aabb.max, world_min, cell_size, grid_size);

      for (int y = minc.y; y <= maxc.y; ++y) {
        for (int x = minc.x; x <= maxc.x; ++x) {
          std::uint32_t idx = fan::spatial::cell_index({x, y}, grid_size);
          cells[idx].push_back(spike);
        }
      }
    }

    void add(fan::vec2 position, fan::vec2 size, std::string_view direction) {
      static constexpr f32_t dir_to_angle[] = {0.f, fan::math::pi, fan::math::pi * 1.5f, fan::math::pi * 0.5f};
      static constexpr std::string_view dirs[] = {"up", "down", "left", "right"};
      int di = 0;
      for (int i = 0; i < 4; ++i) if (dirs[i] == direction) { di = i; break; }

      auto pts = get_spike_points(size, direction);
      spikes.push_back({
        fan::graphics::sprite_t(
          fan::vec3(position, 0),
          size,
          fan::vec3(0, 0, dir_to_angle[di]),
          img
        ),
        fan::physics::gphysics()->create_polygon(
          position, dir_to_angle[di], pts.data(), pts.size(),
          fan::physics::body_type_e::static_body,
          {.is_sensor = true}
        )
      });
      add_to_spatial(spikes.back().sensor);
    }

    fan::physics::entity_t* query(fan::physics::entity_t& entity) {
      auto aabb = entity.get_aabb();
      auto minc = fan::spatial::world_to_cell_clamped(aabb.min, world_min, cell_size, grid_size);
      auto maxc = fan::spatial::world_to_cell_clamped(aabb.max, world_min, cell_size, grid_size);

      for (int y = minc.y; y <= maxc.y; ++y) {
        for (int x = minc.x; x <= maxc.x; ++x) {
          std::uint32_t idx = fan::spatial::cell_index({x, y}, grid_size);
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

    bool query_and_kill(physics::character2d_t& character) {
      if (!query(character)) { return false; }
      character.instant_kill();
      return true;
    }

    bool is_at(fan::vec2 world_pos) const {
      auto minc = fan::spatial::world_to_cell_clamped(world_pos, world_min, cell_size, grid_size);
      std::uint32_t idx = fan::spatial::cell_index(minc, grid_size);
      auto it = cells.find(idx);
      if (it == cells.end()) return false;
      for (auto& spike : it->second) {
        auto aabb = spike.get_aabb();
        if (world_pos.x >= aabb.min.x && world_pos.x <= aabb.max.x &&
          world_pos.y >= aabb.min.y && world_pos.y <= aabb.max.y) {
          return true;
        }
      }
      return false;
    }

    void clear() {
      spikes.clear();
      cells.clear();
    }
  };

}

#endif