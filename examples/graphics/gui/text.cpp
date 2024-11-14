#include <fan/pch.h>

int main() {
  loco_t loco;

  loco.loop([&] {
    fan::graphics::text("top left", fan::vec2(0, 0), fan::colors::red);
    fan::graphics::text_bottom_right("bottom right", fan::colors::green);
  });

  return 0;
}