import std;
import fan;
import fan.graphics.svdag;

using namespace fan::graphics;

struct game_t {
  game_t() {
    loader.start({
      {"models/oldman_idle.fbx", fan::mat4(1).translate({0.f, 0.f, 0.f})},
      {"models/Fox.glb", fan::mat4(1).translate({300.f, 0.f, 0.f})},
      {"models/xbot_idle.fbx", fan::mat4(1).translate({-300.f, 0.f, 0.f})}
    }, voxel_res, "");

    engine.camera_set_position(engine.perspective_render_view, {100.8, -6.3, 75.7});
    auto& cam = engine.camera_get(engine.perspective_render_view);
    cam.yaw = -132;
    cam.pitch = 2.8;
    cam.update_view();

    static auto mouse_motion_handle = engine.window.add_mouse_motion_callback([&](const auto& d) {
      if (gui::is_any_item_active() || !engine.is_mouse_down(fan::mouse_right)) { return; }
      engine.camera_rotate(engine.perspective_render_view, d.motion);
      engine.camera_get(engine.perspective_render_view).update_view();
    });

    gloco()->add_custom_draw([&] {
      if (!renderer) { return; }
      auto& c = engine.camera_get(engine.perspective_render_view);
      renderer->render(c.position, (c.projection * c.view).inverse(), sun_direction);
    });

    engine.loop([&](f32_t dt) {
      if (!renderer && loader.ready()) {
        renderer = loader.finish(voxel_res);
      }

      if (auto h = gui::hud_interactive("##ctrl", 0.f)) {
        if (!renderer) {
          gui::text("Loading...");
        }
        else {
          gui::text(engine.perspective_render_view.get_camera());
          gui::drag("sun", &sun_direction);
          gui::drag("render_scale", &renderer->render_scale, 0.01f, 0.f);
          gui::drag("lod_bias", &renderer->lod_bias);
          gui::drag("ao_quality", &renderer->ao_quality);
          gui::drag("debug_heatmap", &renderer->debug_heatmap);
          gui::text("nodes:", renderer->scene.nodes.size());
          gui::text("leaves:", renderer->scene.leaf_data.size());
          gui::text("assets:", renderer->scene.assets.size());
          gui::text("instances:", renderer->scene.instances.size());
        }
        gui::new_line();
        gui::text("----------------------------------------------------------------------------");
        gui::new_line();
        gui::camera_controls();
      }
    });
  }

  engine_t engine;
  int voxel_res = 512;
  svdag_loader_t loader;
  std::unique_ptr<svdag_renderer_t> renderer;
  fan::vec3 sun_direction{0.6f, 0.9f, 0.4f};
};

int main() {
  game_t game;
}