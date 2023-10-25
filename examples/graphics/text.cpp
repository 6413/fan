#include fan_pch

int main() {
  loco_t loco;

  fan::graphics::text_t t{{
    .text = "hello",
    .color = fan::colors::red,
    .position = fan::vec2(0, 0)
  }};

  loco.loop([&] {
    loco.get_fps();
  });

  return 0;
}