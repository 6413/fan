import fan;
using namespace fan::graphics;
import fan.graphics.vulkan.ray_tracing.hardware_renderer;

namespace rt = fan::graphics::vulkan::ray_tracing;

int main() {
  engine_t engine{{
    .renderer = fan::graphics::renderer_t::vulkan,
  }};

  fan::graphics::vulkan::ray_tracing::context_t rt(engine);
  struct moving_fox_t {
    rt::context_t::object_handle_t handle;
    fan::vec3 base_position;
    f32_t phase;
  };
  std::vector<moving_fox_t> models;
  constexpr int n = 100;
  models.reserve(n);
  constexpr auto model_path = "models/cube_with_colors.fbx";
  for (int i = 0; i < n; ++i) {
    fan::vec3 base_position = fan::random::vec3(-1000, 1000);
    models.push_back({
      rt.add_model(model_path, translate(base_position)),
      base_position,
      fan::random::value(0.f, fan::math::two_pi)
    });
  }

  auto cam = engine.perspective_render_view.camera;
  auto& camera = engine.camera_get(cam);
  engine.camera_set_position(cam, fan::vec3(0, 0, -100));

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
    f32_t t = engine.start_time.seconds();
    for (auto& model : models) {
      fan::vec3 offset(
        std::sin(t + model.phase) * 2.0f,
        std::cos(t * 0.7f + model.phase) * 2.0f,
        std::sin(t * 1.3f + model.phase) * 0.35f
      );
      rt.set_transform(
        model.handle,
        fan::translate(model.base_position).scale(0.1f).rotate(offset)
      );
    }
    if (auto h = gui::hud_interactive{"##2"}) {
      gui::camera_controls();
    }
    rt.render_gui();
  });
}
