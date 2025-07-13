import fan;

int main() {
  loco_t loco;

  fan::vec2 window_size = loco.window.get_size();

  fan::graphics::rectangle_t rect{{
    .position = fan::vec3(400, 400, 0),
    .size = 50,
    .color = fan::colors::red
  }};

  loco.loop([&] {

  });

  return 0;
}