#include <fan/pch.h>

int main() {
  fan::graphics::engine_t engine{{
    .renderer=fan::graphics::engine_t::renderer_t::vulkan
  }};

  int f = 0;
  engine.clear_color.r = 0;
  engine.loop([&] {
    fan_ev_timer_loop(1000, fan::print(1.0f / engine.delta_time););
    if (f == 1) {
      engine.set_target_fps(0);
    }
    f++;
    ImGui::Begin("test");
    ImGui::Button("press me");
    ImGui::End();
  });
  return 0;
}