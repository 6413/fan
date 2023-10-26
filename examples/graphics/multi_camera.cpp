#include fan_pch

int main() {

  loco_t loco;

  auto camera0 = fan::graphics::add_camera(fan::graphics::direction_e::down);
  auto camera1 = fan::graphics::add_camera(fan::graphics::direction_e::right);

  fan::graphics::rectangle_t r{{
      .color = fan::colors::red
  }};
  fan::graphics::rectangle_t r2{{
      .camera = camera0,
      .color = fan::colors::green
  }};
  fan::graphics::rectangle_t r3{{
      .camera = camera1,
      .color = fan::colors::blue
    }};

  loco.loop([&] {

  });
}