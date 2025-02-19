#include <fan/pch.h>

void modify_reflect_depth(std::vector<fan::graphics::rectangle_t>& ray_hit_point, std::vector<fan::graphics::line_t>& rays, int& prev_reflect_depth);

int reflect_depth = 4;

fan_language{
  std::vector<vertex_t> triangle_vertices{
    {vec2(400, 400), colors::orange},
    {vec2(400, 600), colors::orange},
    {vec2(700, 600), colors::orange},
  };
  physics_shapes::polygon_t triangle{{
    .vertices = triangle_vertices
  }};
  std::vector<physics_shapes::circle_t> circles;
  for (std::size_t i = 0; i < 5; ++i) {
    circles.push_back({ {
      .position = fan::random::vec2(0, gloco->window.get_size()),
      .radius = fan::random::f32(12, 84),
      .color = colors::orange,
    }});
  }

  auto box = physics_shapes::create_stroked_rectangle(gloco->window.get_size() / 2, gloco->window.get_size() / 2, 3);

  line_t ray{ { .src={0, 500, 0xfff}, .color = fan::colors::white}};

  std::vector<rectangle_t> ray_hit_point(reflect_depth, {{.size=4, .color=fan::colors::red}});
  std::vector<line_t> rays(reflect_depth, { { .src={0, 0, 0xfff}, .color = fan::colors::green}});

  int prev_reflect_depth = reflect_depth;
  fan_window_loop{
    modify_reflect_depth(ray_hit_point, rays, prev_reflect_depth);
  
    fan::vec2 src = ray.get_src();
    fan::vec2 dst = ray.get_dst();
    ray.set_line(src, dst);
    if (is_mouse_down(fan::mouse_right)) {
      ray.set_line(get_mouse_position(), dst);
    }
    if (is_mouse_down()) {
      ray.set_line(src, get_mouse_position());
    }
    for (auto [i, d] : fan::enumerate(ray_hit_point)) {
      d.set_position(-1000);
      rays[i].set_line(0, 0);
    }
    
    int depth = 0;
    fan::vec2 current_src = src;
    fan::vec2 current_dst = dst;

    while (depth < reflect_depth) {
      if (auto result = physics::raycast(current_src, current_dst)) {
        ray_hit_point[depth].set_position(result.point);

        fan::vec2 direction = (current_dst - current_src).normalize();
        fan::vec2 reflection = direction - result.normal * 2 * direction.dot(result.normal);
        rays[depth].set_line(current_src, result.point);
        rays[depth].set_color(color::hsv(360.f * (depth / (f32_t)reflect_depth), 100, 100));

        current_src = result.point + reflection * 0.5f;
        current_dst = result.point + reflection * 10000.f;

        depth++;
      }
      else {
        break;
      }
    }

    text("reflect depth: " + std::to_string(reflect_depth), {5, 25});
  };
};

void modify_reflect_depth(std::vector<fan::graphics::rectangle_t>& ray_hit_point, std::vector<fan::graphics::line_t>& rays, int& prev_reflect_depth) {
  if (gloco->window.key_state(fan::mouse_scroll_up) != -1) {
    reflect_depth += 1;
  }
  else if (gloco->window.key_state(fan::mouse_scroll_down) != -1) {
    reflect_depth = std::max(reflect_depth - 1, 0);
  }
  if (reflect_depth != prev_reflect_depth) {
    ray_hit_point.resize(reflect_depth, {{.size=4, .color=fan::colors::red}});
    rays.resize(reflect_depth, { { .src={0, 0, 0xfff}, .color = fan::colors::green}});
    prev_reflect_depth = reflect_depth;
  }
}