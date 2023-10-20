#include <global_pch.h>

int main() {
  loco_t loco;


  fan::graphics::rectangle_t r{{
    .position = fan::vec2{0.2, 0.2},
    .size = 0.3
  }};

  loco.loop([] {});
}