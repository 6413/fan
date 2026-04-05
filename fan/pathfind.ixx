module;

#if defined(FAN_2D)
  #include <fan/graphics/2D/algorithm/AStar.hpp>
#endif

export module fan.pathfind;

#if defined(FAN_2D)

import fan.types;
import fan.types.vector;

export namespace fan::pathfind {
  using uint = ::AStar::uint;
  using heuristic_function = std::function<uint(fan::vec2i, fan::vec2i)>;
  using coordinate_list = std::vector<fan::vec2i>;

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

    void init(const fan::vec2i& world_size, bool diagonal = false) {
      set_world_size(world_size);
      set_diagonal_movement(diagonal);
      clear_collisions();
    }

    // returns false if any border origin can't reach it
    bool is_fully_connected(const fan::vec2i& goal) {
      fan::vec2i gc = impl.getWorldSize();
      const fan::vec2i origins[] = {
        {0,       0      }, {gc.x / 2,  0      }, {gc.x - 1,  0      },
        {0,       gc.y / 2 }, {gc.x - 1,  gc.y / 2 },
        {0,       gc.y - 1 }, {gc.x / 2,  gc.y - 1 }, {gc.x - 1,  gc.y - 1}
      };
      for (auto& o : origins) if (find_path(o, goal).empty()) return false;
      return true;
    }


    std::vector<::AStar::Vec2i>& get_walls();
  };

  struct heuristic {
    static uint manhattan(vec2i a, vec2i b);
    static uint euclidean(vec2i a, vec2i b);
    static uint octagonal(vec2i a, vec2i b);
  };

  struct follow_result_t { vec2 target; bool bash; };

  follow_result_t follow(std::vector<vec2i>& path, vec2 pos, vec2 goal, f32_t grid, int bash_thresh, f32_t arrive_r2 = 1600.f) {
    if (!path.empty()) {
      vec2 tgt = vec2(path.back()) * grid + grid / 2.f;
      if ((pos - tgt).length_squared() < arrive_r2) path.pop_back();
    }
    if ((int)path.size() > bash_thresh) return {goal, true};
    if (!path.empty()) return {vec2(path.back()) * grid + grid / 2.f, false};
    return {goal, false};
  }
}

#endif