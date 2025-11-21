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
    node(vec2i coord_, node* parent_ = nullptr);
    uint get_score();
  };

  using node_set = ::AStar::NodeSet;

  struct generator {
    ::AStar::Generator impl;

    void set_world_size(vec2i world_size);
    void set_diagonal_movement(bool enable);
    void set_heuristic(heuristic_function h);
    coordinate_list find_path(vec2i src, vec2i dst);
    void add_collision(vec2i c);
    void remove_collision(vec2i c);
    void clear_collisions();
    coordinate_list& get_walls();
  };

  struct heuristic {
    static uint manhattan(vec2i a, vec2i b);
    static uint euclidean(vec2i a, vec2i b);
    static uint octagonal(vec2i a, vec2i b);
  };
}