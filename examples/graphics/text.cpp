#include fan_pch

int main() {
  loco_t loco;
  loco.default_camera->camera.set_ortho(
    fan::vec2(-1, 1),
    fan::vec2(-1, 1)
  );
  fan::graphics::text_t t{{
    .text = "hello",
    .color = fan::colors::white,
  }};

  loco.loop([&] {
    if (int fps = loco.get_fps()) {
      auto text = std::to_string(fps);
      t.set_size(0.025 * text.size());
      t.set_text(text);
    }
  });

  return 0;
}