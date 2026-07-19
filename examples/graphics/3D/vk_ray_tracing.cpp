import std;
import fan;

using namespace fan::graphics;
namespace rt = fan::graphics::vulkan::ray_tracing;

int main() {
  fan::graphics::engine_t engine;

  rt::context_t renderer(engine);
  renderer.set_light(fan::vec3(0.f, 220.f, -180.f), fan::vec3(1.f, 0.96f, 0.88f), 0.1f);

  rt::gpu_terrain_streamer_t terrain;
  auto cam = engine.perspective_render_view.camera;
  bool noclip = false;
  bool ui_open = false;

  {
    auto& camera = engine.camera_get(cam);
    camera.position = fan::vec3(0.f, 90.f, -520.f);
    camera.update_view();
    camera.view = camera.get_view_matrix();
  }

  auto highlight = renderer.add_model(
    "models/cube_with_colors.fbx",
    fan::translate(fan::vec3(0.f, -1000000.f, 0.f)).scale(fan::vec3(rt::gpu_terrain_streamer_t::voxel_size * 1.03f))
  );
  renderer.set_ray_mask(highlight, 0x02);

  auto hide_highlight = [&] {
    renderer.set_transform_deferred(
      highlight,
      fan::translate(fan::vec3(0.f, -1000000.f, 0.f)).scale(fan::vec3(rt::gpu_terrain_streamer_t::voxel_size * 1.03f))
    );
  };

  auto cb = engine.window.add_mouse_motion_callback([&](const auto& d) {
    if (ui_open || engine.window.is_cursor_enabled()) { return; }
    auto& camera = engine.camera_get(cam);
    camera.rotate_camera(d.motion);
    camera.view = camera.get_view_matrix();
  });

  engine.window.set_cursor(0);
  sprite_t crosshair(fan::vec3(engine.window.get_size() / 2.f, 0xffa), 4.f, "images/circle.png");

  engine.loop([&] {
    if (engine.is_key_clicked(fan::key_r)) { renderer.reload_pipeline(); }
    if (engine.is_key_clicked(fan::key_q)) { noclip = !noclip; }
    if (engine.is_key_clicked(fan::key_t)) {
      ui_open = !ui_open;
      engine.window.set_cursor(ui_open ? 1 : 0);
      if (ui_open) { hide_highlight(); }
    }

    if (engine.window.is_cursor_enabled() != ui_open) {
      engine.window.set_cursor(ui_open ? 1 : 0);
    }

    if (ui_open) {
      if (auto h = fan::graphics::gui::hud_interactive {"##rt"}; h) {
        fan::graphics::gui::camera_controls();
        terrain.render_gui(renderer);
      }
    }
    else {
      engine.camera_move(noclip, [&](f32_t x, f32_t z, f32_t r) {
        return terrain.ground_y(x, z, r);
      });
    }

    terrain.update(renderer, engine.camera_get(cam).position);

    rt::context_t::pick_result_t pick;
    if (renderer.get_pick_result(pick)) {
      fan::vec3 hit_position = fan::vec3(pick.position);
      fan::vec3 hit_normal = fan::vec3(pick.normal).normalize();
      fan::vec3 place_p = hit_position + hit_normal * 0.01f;
      auto place_block = terrain.world_to_block(place_p);
      fan::vec3 highlight_position = terrain.block_to_world(place_block) + fan::vec3(0.5f) * rt::gpu_terrain_streamer_t::voxel_size;

      //renderer.set_transform_deferred(
      //  highlight,
      //  fan::translate(highlight_position).scale(fan::vec3(rt::gpu_terrain_streamer_t::voxel_size * 0.005f))
      //);

      if (!ui_open && !engine.window.is_cursor_enabled() && engine.is_mouse_clicked(fan::mouse_right)) {
        terrain.set_block(renderer, terrain.world_to_block(place_p), 1);
      }

      if (!ui_open && !engine.window.is_cursor_enabled() && engine.is_mouse_clicked(fan::mouse_left)) {
        fan::vec3 break_p = hit_position - hit_normal * 0.01f;
        terrain.remove_block(renderer, terrain.world_to_block(break_p));
      }
    }
    else {
      hide_highlight();
    }
  });

  terrain.destroy();
}