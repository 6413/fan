#include <cstdint>
import fan;
import fan.graphics.vulkan.ray_tracing.hardware_renderer;
import fan.graphics.vulkan.ray_tracing.gpu_terrain_streamer;

namespace rt = fan::graphics::vulkan::ray_tracing;

int main() {
  fan::graphics::engine_t engine{{
    .renderer = fan::graphics::renderer_t::vulkan,
  }};

  rt::context_t renderer(engine);
  renderer.set_light(fan::vec3(0.f, 220.f, -180.f), fan::vec3(1.f, 0.96f, 0.88f), 0.6f);

  rt::gpu_terrain_streamer_t terrain;
  auto cam = engine.perspective_render_view.camera;
  engine.camera_set_position(cam, fan::vec3(0, 90, -520));

  auto cb = engine.window.add_mouse_motion_callback([&](const auto& d) {
    if (engine.window.is_cursor_enabled()) { return; }
    engine.camera_get(cam).rotate_camera(d.motion);
  });

  engine.loop([&] {
    terrain.update(renderer, engine.camera_get(cam).position);

    if (engine.is_key_clicked(fan::key_r)) { renderer.reload_pipeline(); }
    if (engine.is_mouse_clicked(fan::mouse_right)) { engine.window.toggle_cursor(); }
    if (engine.is_mouse_released(fan::mouse_right)) { engine.window.toggle_cursor(); }

    if (auto h = fan::graphics::gui::hud_interactive{"##rt"}; h && engine.is_toggled(fan::key_t)) {
      fan::graphics::gui::camera_controls();
      terrain.render_gui(renderer);
    }
    else {
      engine.camera_move();
    }
  });

  terrain.destroy();
}