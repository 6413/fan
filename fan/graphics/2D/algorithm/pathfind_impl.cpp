module;

#if defined(FAN_2D)
#include <fan/graphics/2D/algorithm/AStar.hpp>
#endif

module fan.graphics.algorithm.pathfind;

#if defined(FAN_2D)

fan::graphics::algorithm::pathfind::node::node(vec2i coord_, node* parent_)
  : G(0), H(0), coordinates(coord_), parent(parent_) {
}

fan::graphics::algorithm::pathfind::uint
fan::graphics::algorithm::pathfind::node::get_score() {
  return G + H;
}

void fan::graphics::algorithm::pathfind::generator::set_world_size(vec2i world_size) {
  impl.setWorldSize(::AStar::Vec2i(world_size));
}

void fan::graphics::algorithm::pathfind::generator::set_diagonal_movement(bool enable) {
  impl.setDiagonalMovement(enable);
}

void fan::graphics::algorithm::pathfind::generator::set_heuristic(heuristic_function h) {
  impl.setHeuristic(h);
}

fan::graphics::algorithm::pathfind::coordinate_list
fan::graphics::algorithm::pathfind::generator::find_path(vec2i src, vec2i dst) {
  return impl.findPath(::AStar::Vec2i(src), ::AStar::Vec2i(dst));
}

void fan::graphics::algorithm::pathfind::generator::add_collision(vec2i c) {
  impl.addCollision(::AStar::Vec2i(c));
}

void fan::graphics::algorithm::pathfind::generator::remove_collision(vec2i c) {
  impl.removeCollision(::AStar::Vec2i(c));
}

void fan::graphics::algorithm::pathfind::generator::clear_collisions() {
  impl.clearCollisions();
}

fan::graphics::algorithm::pathfind::coordinate_list&
fan::graphics::algorithm::pathfind::generator::get_walls() {
  return impl.getWalls();
}

fan::graphics::algorithm::pathfind::uint
fan::graphics::algorithm::pathfind::heuristic::manhattan(vec2i a, vec2i b) {
  return ::AStar::Heuristic::manhattan(::AStar::Vec2i(a), ::AStar::Vec2i(b));
}

fan::graphics::algorithm::pathfind::uint
fan::graphics::algorithm::pathfind::heuristic::euclidean(vec2i a, vec2i b) {
  return ::AStar::Heuristic::euclidean(::AStar::Vec2i(a), ::AStar::Vec2i(b));
}

fan::graphics::algorithm::pathfind::uint
fan::graphics::algorithm::pathfind::heuristic::octagonal(vec2i a, vec2i b) {
  return ::AStar::Heuristic::octagonal(::AStar::Vec2i(a), ::AStar::Vec2i(b));
}

#endif