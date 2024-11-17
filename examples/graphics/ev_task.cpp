#include <fan/pch.h>

#include <coroutine>
#include <fan/ev/ev.h>

using namespace std::chrono_literals;


std::vector<loco_t::shape_t> shapes;

task<void> do_something(task<void>* t) {
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

int main() {
  loco_t loco;
  task<void> t = do_something(&t);
  t.coro.resume();
  loco.loop([&] {
    
  });
}