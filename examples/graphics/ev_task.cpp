#include <fan/pch.h>

#include <fan/ev/ev.h>

using namespace std::chrono_literals;


std::vector<loco_t::shape_t> shapes;


fan::event::task_t spawn_rectangles() {
  fan::print("start");
  while (true) {
    shapes.clear();
    shapes.push_back(fan::graphics::rectangle_t{ {
        .position = fan::vec3(fan::random::vec2(0, 400), 0),
        .size = fan::random::vec2(100, 400),
        .color = fan::random::color()
    } });
    co_await fan::co_sleep(1);
  }
}

int main() {
  loco_t loco;
  auto t = spawn_rectangles();
  loco.loop([&] {
    
  });
}