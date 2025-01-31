#pragma once

#include <fan/graphics/algorithm/AStar.hpp>

namespace fan {
  namespace algorithm {
    struct path_solver_t {
      using path_t = AStar::CoordinateList;
      path_solver_t() = default;
      path_solver_t(const fan::vec2i& map_size_, const fan::vec2& tile_size_) {
        this->map_size = map_size_;
        generator.setWorldSize(map_size);
        generator.setHeuristic(AStar::Heuristic::euclidean);
        generator.setDiagonalMovement(true);
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
        fan::vec2i fp = ((p + tile_size / 2) / tile_size).floor();
        return fp.clamp(0, map_size - 1);
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
      fan::vec2 step(const fan::vec2& src_) {
        if (move) {
          fan::vec2 srcp = src_ - tile_size/2;
          //srcp.y += tile_size.y/2;
          fan::vec2 src_pos = get_grid(srcp);
          if (init_solve) {
            current_position = 0;
            init_solve = false;
          }
          
          if (srcp.is_near(fan::vec2i(path[current_position]) * tile_size - tile_size / 2, 5)) {
            ++current_position;
          }
          if (srcp.is_near(fan::vec2i(dst) * tile_size - tile_size / 2, 5)) {
            move = false;
          }
          else {
            fan::vec2i currenti = current_position >= path.size() ? dst : fan::vec2i(path[current_position]);
            fan::vec2 current = currenti * tile_size-tile_size/2;
            fan::vec2 direction = current - srcp;
            if (std::abs(direction.x) < 1) {
              direction.x = 0;
            }
            if (std::abs(direction.y) < 1) {
              direction.y = 0;
            }
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