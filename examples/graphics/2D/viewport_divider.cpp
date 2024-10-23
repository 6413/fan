#include <fan/pch.h>
#include <fan/graphics/divider.h>


int main() {

  fan::graphics::viewport_divider_t d;

  loco_t loco;

  loco.camera_set_ortho(
    loco.orthographic_camera.camera,
    fan::vec2(-1, 1),
    fan::vec2(-1, 1)
  );

  struct node_t {
    fan::graphics::rectangle_t rect;
  };

  std::vector<node_t> nodes;

  auto get_random_direction = []() {
    return (fan::graphics::viewport_divider_t::direction_e)(fan::random::value_i64(0, 1));
  };

  uint16_t main_viewports = 30;
  uint16_t sub_viewports = 30;

  for (int i = 0; i < main_viewports; ++i) {
    fan::graphics::viewport_divider_t::direction_e dir = get_random_direction();
    fan::graphics::viewport_divider_t::iterator_t it = d.insert(dir);
    for (int j = 0; j < sub_viewports; ++j) {
      it = d.insert(it, get_random_direction());
    }
  }

  d.iterate([&](fan::graphics::viewport_divider_t::window_t& node) {
    static int depth = 0;
    nodes.push_back(node_t{
      .rect{{
        .position = fan::vec3(node.position * 2 - 1, depth++),
        .size = node.size,
      //.color = fan::color::hsv(fan::random::value_i64(0, 360), 100, 100),
        .color = fan::random::color() - fan::color(0, 0, 0, .5), // have some alpha to see viewports dont collide
        .blending = true
      }}
    });
  });

  loco.loop([&] {

  });

  return 0;
}
