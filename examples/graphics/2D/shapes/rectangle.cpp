#include <fan/pch.h>

//fan_track_allocations();

int main() {
  using namespace fan::graphics;
  engine_t engine{{
    .renderer=engine_t::renderer_t::vulkan
  }};

  std::vector<rectangle_t>rects(1);
  for (int i = 0; i < 1; ++i) {
    rects[i] = { {
    .position = fan::vec3(fan::vec2(i, i), 1),
    .size = 100,
    .color = fan::colors::blue,
  } };
  }
  //fan::time::clock c;
  //c.start(1e+9);
  //int f = 0;
  int f = 0;
  fan_window_loop{

  };
}