#include <fan/pch.h>
#include <fan/graphics/vulkan/ssbo.h>

int main() {
  fan::graphics::engine_t engine{{
    .renderer=fan::graphics::engine_t::renderer_t::vulkan
  }};
  //fan::vulkan::core::shader_t shader;
  //fan::vulkan::core::memory_write_queue_t wq;
  //shader.open(engine, &wq);
  //fan::vulkan::core::ssbo_t<loco_t::rectangle_t::vi_t, loco_t::rectangle_t::ri_t, 128, 3> ssbo;
  //ssbo.allocate(engine, 1024);
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