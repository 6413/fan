#include <fan/pch.h>

#include <coroutine>
#include <fan/ev/ev.h>

using namespace std::chrono_literals;


std::vector<loco_t::shape_t> shapes;

#if !defined(loco_noev)
task<void> spawn_rectangles(task<void>* t) {
  fan::print("start");
  while (true) {
    shapes.clear();
    shapes.push_back(fan::graphics::rectangle_t{ {
        .position = fan::vec3(fan::random::vec2(0, 400), 0),
        .size = fan::random::vec2(100, 400),
        .color = fan::random::color()
    } });
    co_await co_sleep_for(t, 1ms);
  }
}
#endif

int main() {
  loco_t loco;
#if !defined(loco_noev)
  task<void> t = spawn_rectangles(&t);
  t.coro.resume();
#endif
  loco.loop([&] {
    
  });
}