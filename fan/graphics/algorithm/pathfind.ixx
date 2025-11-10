module;

#include <fan/graphics/algorithm/AStar.hpp>

export module fan.graphics.algorithm.pathfind;

export namespace fan::graphics::algorithm::pathfind {
  using uint = ::AStar::uint;
  using heuristic_function = ::AStar::HeuristicFunction;
  using coordinate_list = ::AStar::CoordinateList;

  struct node {
    uint G;
    uint H;
    vec2i coordinates;
    node* parent;

    node(vec2i coord_, node* parent_ = nullptr)
      : G(0), H(0), coordinates(coord_), parent(parent_) {
    }

    uint get_score() {
      return G + H;
    }
  };

  using node_set = ::AStar::NodeSet;

  struct generator {
    ::AStar::Generator impl;

    void set_world_size(vec2i world_size) {
      impl.setWorldSize(::AStar::Vec2i(world_size));
    }

    void set_diagonal_movement(bool enable) {
      impl.setDiagonalMovement(enable);
    }

    void set_heuristic(heuristic_function h) {
      impl.setHeuristic(h);
    }

    coordinate_list find_path(vec2i src, vec2i dst) {
      return impl.findPath(::AStar::Vec2i(src), ::AStar::Vec2i(dst));
    }

    void add_collision(vec2i c) {
      impl.addCollision(::AStar::Vec2i(c));
    }

    void remove_collision(vec2i c) {
      impl.removeCollision(::AStar::Vec2i(c));
    }

    void clear_collisions() {
      impl.clearCollisions();
    }

    coordinate_list& get_walls() {
      return impl.getWalls();
    }
  };

  struct heuristic {
    static uint manhattan(vec2i a, vec2i b) {
      return ::AStar::Heuristic::manhattan(::AStar::Vec2i(a), ::AStar::Vec2i(b));
    }

    static uint euclidean(vec2i a, vec2i b) {
      return ::AStar::Heuristic::euclidean(::AStar::Vec2i(a), ::AStar::Vec2i(b));
    }

    static uint octagonal(vec2i a, vec2i b) {
      return ::AStar::Heuristic::octagonal(::AStar::Vec2i(a), ::AStar::Vec2i(b));
    }
  };

}
