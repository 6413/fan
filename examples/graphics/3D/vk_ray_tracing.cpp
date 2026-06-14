import fan;
using namespace fan::graphics;
import fan.graphics.vulkan.ray_tracing.hardware_renderer;

namespace rt = fan::graphics::vulkan::ray_tracing;

int main() {
  engine_t engine{{
    .renderer = fan::graphics::renderer_t::vulkan,
  }};

  fan::graphics::vulkan::ray_tracing::context_t rt(engine);
  int i = 0;
  struct moving_fox_t {
    rt::context_t::object_handle_t handle;
    fan::vec3 base_position;
    f32_t phase;
  };
  std::vector<moving_fox_t> foxes;
  foxes.reserve(5000);
  for (; i < 4499; ++i) {
    fan::vec3 base_position(
      fan::random::value(-120.f, 120.f),
      fan::random::value(-120.f, 120.f),
      4.0f
    );
    foxes.push_back({
      rt.add_model("models/Fox.glb", fan::mat4(1).translate(base_position)),
      base_position,
      fan::random::value(0.f, fan::math::two_pi)
    });
  }

  auto camera_handle = engine.perspective_render_view.camera;
  auto& camera = engine.camera_get(camera_handle);

  auto cb = engine.window.add_mouse_motion_callback([&](const auto& d) {
    if (engine.window.is_cursor_enabled()) return;
    camera.rotate_camera(d.motion);
  });

  engine.loop([&] {
    if (engine.is_key_clicked(fan::key_r)) {
      rt.reload_pipeline();
      fan::print("Ray tracing shaders reloaded.");
    }
    if (engine.is_mouse_clicked(fan::mouse_right))  engine.window.toggle_cursor();
    if (engine.is_mouse_released(fan::mouse_right)) engine.window.toggle_cursor();
    if (engine.is_mouse_clicked()) {
      fan::vec3 base_position(0.0f, 0.0f, 4.0f);
      foxes.push_back({
        rt.add_model("models/Fox.glb", fan::mat4(1).translate(base_position).scale(1.f)),
        base_position,
        fan::random::value(0.f, fan::math::two_pi)
      });
      ++i;
    }
    f32_t t = engine.start_time.seconds();
    for (auto& moving_fox : foxes) {
      fan::vec3 offset(
        std::sin(t + moving_fox.phase) * 2.0f,
        std::cos(t * 0.7f + moving_fox.phase) * 2.0f,
        std::sin(t * 1.3f + moving_fox.phase) * 0.35f
      );
      rt.set_transform(
        moving_fox.handle,
        fan::mat4(1)
          .translate(moving_fox.base_position * 10.f + offset * 100.f)
          .scale(1.f)
      );
    }
    if (auto h = gui::hud_interactive{"##2"}) {
      gui::camera_controls();
    }
    rt.render_gui();
  });
}
