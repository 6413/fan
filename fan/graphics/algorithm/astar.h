#pragma once

#include <fan/graphics/algorithm/AStar.hpp>

namespace fan {
  namespace algorithm {
    struct path_solver_t {
      using path_t = AStar::CoordinateList;
      path_solver_t(const fan::vec2i& map_size_, const fan::vec2& tile_size_) {
        this->map_size = map_size_;
        generator.setWorldSize(map_size);
        generator.setHeuristic(AStar::Heuristic::euclidean);
        generator.setDiagonalMovement(false);
        tile_size = tile_size_;
      }
      path_t get_path(const fan::vec2& src_ = -1) {
        if (src_ != -1) {
          src = get_grid(src_);
        }
        path_t v = generator.findPath(src, dst);
        v.pop_back();
        std::reverse(v.begin(), v.end());
        return v;
      }
      fan::vec2i get_grid(const fan::vec2& p) const {
        fan::vec2i fp = (p / tile_size).floor();
        fp = fp.clamp(0, map_size - 1);
        return fp;
      }
      // takes raw position and converts it to grid automatically
      void set_src(const fan::vec2& src_) {
        src = get_grid(src_);
      }
      void set_dst(const fan::vec2& dst_) {
        dst = get_grid(dst_);
      }
      void add_collision(const fan::vec2& p) {
        generator.addCollision(get_grid(p));
      }
      AStar::Generator generator;
      fan::vec2i src = 0;
      fan::vec2i dst = 0;
      fan::vec2i map_size;
      fan::vec2 tile_size;

      void init(const fan::vec2& src_) {
        path = get_path(src_);
        move = true;
        init_solve = true;
      }

      // returns direction to go from current position in normalized grid format
      // when function returns {0, 0}, its finished
      fan::vec2i step(const fan::vec2& src_) {
        if (move) {
          fan::vec2i src_pos = get_grid(src_);
          if (init_solve) {
            current_position = 0;
            init_solve = false;
          }
          if (src_pos == fan::vec2i(path[current_position])) {
            ++current_position;
          }
          if (src_pos == dst) {
            move = false;
          }
          else {
            fan::vec2i current = current_position >= path.size() ? dst : fan::vec2i(path[current_position]);
            fan::vec2i direction = current - src_pos;
            return direction;
          }
        }
        return 0;
      }

      bool move = false;
      bool init_solve = true;

      path_solver_t::path_t path;
      int current_position = 0;
    };
  }
}