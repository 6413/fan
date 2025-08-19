#include <fan/pch.h>

int main() {
  fan::graphics::engine_t engine;

  std::vector<fan::graphics::physics::rectangle_t> walls;
  fan::vec2 window_size = fan::window::get_size();
  for (int i = 0; i < 20; ++i) {
    walls.emplace_back(
    fan::graphics::physics::rectangle_t{ {
      .position = fan::random::vec2(0, window_size),
      .size = 50,
      .color = fan::color::from_rgba(0x30a6b6ff),
      .outline_color = fan::color::from_rgba(0x30a6b6ff) * 2,
      .body_type = fan::physics::body_type_e::static_body,
      .shape_properties{.presolve_events = true},
      } }
    );
  }
  
  int ray_count = 4000;
  std::vector<fan::graphics::line_t> rays(ray_count);

  fan_window_loop{

    f32_t angle = 0;
    int ray_counter = 0;
    f32_t ray_length = 4000;

    fan::vec2 ray_start = fan::graphics::get_mouse_position();

    f32_t r = 0;

    for (auto& ray : rays) {
      angle = (f32_t)ray_counter++ / ray_count * fan::math::two_pi;
      fan::vec2 angle_vector = fan::math::direction_vector<fan::vec2>(angle) * ray_length;
      fan::physics::ray_result_t result = fan::physics::raycast(ray_start, ray_start + angle_vector);
      r += result.point.x * result.point.y;
      if (result.hit) {
        ray.set_line(ray_start, result.point);
      }
      else {
        ray.set_line(ray_start, ray_start + angle_vector);
      }
    }
  };
  return 0;
}