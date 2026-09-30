module;

#if defined(FAN_2D)
  #include <fan/graphics/2D/algorithm/AStar.hpp>
#endif

module fan.pathfind;

import std;

#if defined(FAN_2D)

fan::pathfind::node::node(vec2i coord_, node* parent_)
  : G(0), H(0), coordinates(coord_), parent(parent_) {
}

fan::pathfind::uint
fan::pathfind::node::get_score() {
  return G + H;
}

void fan::pathfind::generator::set_world_size(vec2i world_size) {
  impl.setWorldSize(::AStar::Vec2i(world_size));
}

void fan::pathfind::generator::set_diagonal_movement(bool enable) {
  impl.setDiagonalMovement(enable);
}

void fan::pathfind::generator::set_heuristic(heuristic_function h) {
  impl.setHeuristic(h);
}

fan::pathfind::coordinate_list
fan::pathfind::generator::find_path(vec2i src, vec2i dst) {
  auto p = impl.findPath(::AStar::Vec2i(src), ::AStar::Vec2i(dst));
  std::vector<fan::vec2i> res;
  res.resize(p.size());
  std::memcpy(res.data(), p.data(), sizeof(fan::vec2i) * res.size());
  return res;
}

void fan::pathfind::generator::add_collision(vec2i c) {
  impl.addCollision(::AStar::Vec2i(c));
}

void fan::pathfind::generator::remove_collision(vec2i c) {
  impl.removeCollision(::AStar::Vec2i(c));
}

void fan::pathfind::generator::clear_collisions() {
  impl.clearCollisions();
}

std::vector<::AStar::Vec2i>&
fan::pathfind::generator::get_walls() {
  return impl.getWalls();
}

void fan::pathfind::generator::init(const fan::vec2i& world_size, bool diagonal) {
  set_world_size(world_size);
  set_diagonal_movement(diagonal);
  clear_collisions();
}

bool fan::pathfind::generator::is_fully_connected(const fan::vec2i& goal) {
  fan::vec2i gc = impl.getWorldSize();
  const fan::vec2i origins[] = {
    {0,       0      }, {gc.x / 2,  0      }, {gc.x - 1,  0      },
    {0,       gc.y / 2 }, {gc.x - 1,  gc.y / 2 },
    {0,       gc.y - 1 }, {gc.x / 2,  gc.y - 1 }, {gc.x - 1,  gc.y - 1}
  };
  for (auto& o : origins) if (find_path(o, goal).empty()) return false;
  return true;
}

fan::pathfind::uint
fan::pathfind::heuristic::manhattan(vec2i a, vec2i b) {
  return ::AStar::Heuristic::manhattan(::AStar::Vec2i(a), ::AStar::Vec2i(b));
}

fan::pathfind::uint
fan::pathfind::heuristic::euclidean(vec2i a, vec2i b) {
  return ::AStar::Heuristic::euclidean(::AStar::Vec2i(a), ::AStar::Vec2i(b));
}

fan::pathfind::uint
fan::pathfind::heuristic::octagonal(vec2i a, vec2i b) {
  return ::AStar::Heuristic::octagonal(::AStar::Vec2i(a), ::AStar::Vec2i(b));
}

#endif