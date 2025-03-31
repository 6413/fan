#include <fan/pch.h>
#include <fan/imgui/implot_internal.h>

fan_track_allocations();

int main() {
  loco_t loco;
  fan::print(sizeof(loco_t::polygon_vertex_t), (offsetof(loco_t::polygon_vertex_t, angle)));
  fan::graphics::polygon_t hexagon = {{
    .position = fan::vec3(600, 300, 1),
    .vertices = fan::graphics::create_hexagon(50, fan::colors::blue).vertices,
    .angle = fan::vec3(0, 0, fan::math::pi / 2)
  }};

  fan::graphics::polygon_t hexagon2 = {{
    .position = fan::vec3(600, 300, 2),
    .vertices = fan::graphics::create_hexagon(100, fan::colors::red).vertices,
  }};
  f32_t a = 0;
  loco.loop([&] {
    hexagon2.set_angle(fan::vec3(0, 0, a));
    a += loco.delta_time / 2;
    hexagon.set_position(fan::graphics::get_mouse_position());
    hexagon2.set_position(fan::graphics::get_mouse_position() + 300);
  });
}