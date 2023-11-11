#include fan_pch

int main() {
  loco_t loco;
  loco.default_camera->camera.set_ortho(
    fan::vec2(0, loco.get_window()->get_size().x),
    fan::vec2(0, loco.get_window()->get_size().y)
  );
  fan::graphics::rectangle_t r{{
      .position = 400,
      .size = 200,
      .color = fan::colors::red
  }};
  
  loco.loop([&] {
    r.set_position(loco.get_mouse_position());
    fan::print(r.get_position());
  });
}