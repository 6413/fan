#include fan_pch

int main() {
  loco_t loco;
  fan::graphics::rectangle_t r{{
      .position = 0.3,
      .size = 0.2,
      .color = fan::colors::red
    }};

  loco.loop([] {
  });
}