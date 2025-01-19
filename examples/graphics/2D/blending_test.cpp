#include <fan/pch.h>

int main() {
  fan::graphics::engine_t engine{{.window_size=800}};
  engine.clear_color = fan::colors::black;
  engine.blend_src_factor = fan::opengl::GL_ONE;
  engine.blend_dst_factor = fan::opengl::GL_ONE;

  fan::graphics::circle_t rr{{
    .position = fan::vec3(350, 300, 0),
    .radius = 128,
    .color = fan::colors::red.set_alpha(0.5),
    .blending = true
  }};
  fan::graphics::circle_t rg{{
    .position = fan::vec3(450, 300, 1),
    .radius = 128,
    .color = fan::colors::green.set_alpha(0.5),
    .blending = true
  }};
  fan::graphics::circle_t rb{{
    .position = fan::vec3(400, 400, 2),
    .radius = 128,
    .color = fan::colors::blue.set_alpha(0.5),
    .blending = true
  }};

  engine.loop([&] {

  });

  return 0;
}