import fan;
using namespace fan::graphics;
import fan.graphics.vulkan.ray_tracing.hardware_renderer;

int main() {
  engine_t engine{{
    .renderer = fan::graphics::renderer_t::vulkan,
  }};

  fan::graphics::vulkan::ray_tracing::context_t rt(engine);
  int i = 0;
  for (; i < 4500; ++i) {
    rt.add_model("models/Fox.glb", fan::mat4(1).translate(fan::random::vec3(-1000,1000)));
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
      rt.add_model("models/Fox.glb", fan::mat4(1).translate(fan::vec3(0.0f, i * 5.f, 4.0f)).scale(1.f));
      ++i;
    }
    gui::camera_controls();
    rt.render_gui();
  });
}