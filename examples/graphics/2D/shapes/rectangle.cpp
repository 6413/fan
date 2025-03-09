#include <fan/pch.h>

//fan_track_allocations();

int main() {
  using namespace fan::graphics;
  engine_t engine{{
    .renderer=engine_t::renderer_t::opengl
  }};

  rectangle_t rect{ {
    .position = fan::vec3(fan::vec2(400, 400), 1),
    .size = 100,
    .color = fan::colors::blue,
  } };
  fan_window_loop{
  };
}