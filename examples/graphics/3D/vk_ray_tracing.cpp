#include <cstdint>
#include <vector>
import fan;
import fan.graphics.vulkan.ray_tracing.hardware_renderer;

namespace rt = fan::graphics::vulkan::ray_tracing;

int main() {
  fan::graphics::engine_t engine{{
    .renderer = fan::graphics::renderer_t::vulkan,
  }};

  rt::context_t renderer(engine);
  renderer.animation_sample_rate = 30.f;

  constexpr std::uint32_t actor_count = 100;
  std::vector<rt::context_t::object_handle_t> actors;
  actors.reserve(actor_count);

  for (std::uint32_t i = 0; i < actor_count; ++i) {
    fan::vec3 p = fan::random::vec3(-100.f, 100.f);
    actors.push_back(renderer.add_animated_model(
      "models/oldman_idle.fbx",
      fan::translate(p).scale(1.f),
      "",
      fan::random::value(0.f, 10.f),
      fan::random::value(0.85f, 1.15f)
    ));
  }

  auto cam = engine.perspective_render_view.camera;
  engine.camera_set_position(cam, fan::vec3(0, 0, -100));

  auto cb = engine.window.add_mouse_motion_callback([&](const auto& d) {
    if (engine.window.is_cursor_enabled()) {
      return;
    }
    engine.camera_get(cam).rotate_camera(d.motion);
  });

  engine.loop([&] {
    if (engine.is_key_clicked(fan::key_r)) {
      renderer.reload_pipeline();
    }
    if (engine.is_mouse_clicked(fan::mouse_right)) {
      engine.window.toggle_cursor();
    }
    if (engine.is_mouse_released(fan::mouse_right)) {
      engine.window.toggle_cursor();
    }
    if (auto h = fan::graphics::gui::hud_interactive{"##rt"}) {
      fan::graphics::gui::camera_controls();
    }
    renderer.render_gui();
  });
}
