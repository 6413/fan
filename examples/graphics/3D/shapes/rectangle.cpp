import fan;

using namespace fan::graphics;

int main() {
  engine_t engine;

  fan::vec3 cube_size = 10.f;
  constexpr auto n = 10000;
  std::vector<fan::graphics::rectangle3d_t> r3(n);
  for (int i = 0; i < n; ++i) {
    r3[i] = {fan::random::vec3(-10000, 10000), cube_size, fan::colors::purple};
  }

  auto motion_cb_id = engine.window.add_mouse_motion_callback([&](const auto& d) {
    if (!engine.is_mouse_down(fan::mouse_right)) return;
    engine.camera_rotate(d.motion);
  });

  engine.loop([&] (f32_t dt) {
    engine.camera_move(10000.f);
  });
}