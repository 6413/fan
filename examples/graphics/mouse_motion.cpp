#include fan_pch

int main() {
  loco_t loco;

  loco.get_window()->set_flag_value<fan::window_t::flags::no_mouse>(true);

  loco.get_window()->add_mouse_motion([&](const auto& x) {
    fan::print(x.motion);
  });
  loco.set_vsync(false);
  loco.get_window()->set_max_fps(10);
  loco.loop([&] {

  });

  return 0;
}