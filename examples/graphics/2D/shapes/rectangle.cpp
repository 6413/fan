#include <fan/pch.h>

//fan_track_allocations();

int main() {
  using namespace fan::graphics;
  engine_t engine{{
    .renderer=engine_t::renderer_t::vulkan
  }};

  std::vector<rectangle_t> rects;
  for (int i = 0; i < 0; ++i) {
    rects.push_back({ {
      .position = fan::vec3(fan::random::vec2(100, 600), 0),
      .size = fan::random::vec2(10, 100),
      .color = fan::random::color(),
    }});
  }
 /* rectangle_t rect{ {
    .position = fan::vec3(fan::vec2(400, 400), 1),
    .size = 100,
    .color = fan::colors::blue,
  } };*/


  auto img = engine.image_load("images/tire.png");

  sprite_t s{ {
    .position = fan::vec3(fan::vec2(400, 400), 1),
    .size=100,
    .image = img
  }};

  sprite_t s2{ {
    .position = fan::vec3(fan::vec2(500, 400), 1),
    .size=300
  }};

 /* rectangle_t rect;

  rectangle_t rect2{ {
    .position = fan::vec3(fan::vec2(600, 600), 1),
    .size = 100,
    .color = fan::colors::blue,
  } };
  rect2.erase();*/

  fan_window_loop{
    //rect.set_position(get_mouse_position());
    //s.set_position(get_mouse_position());
  };
}