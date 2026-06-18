import std;
import fan;
import fan.graphics.vulkan.ray_tracing.hardware_renderer;
import fan.graphics.vulkan.ray_tracing.gpu_terrain_streamer;

using namespace fan::graphics;

namespace rt = fan::graphics::vulkan::ray_tracing;

int main() {
  fan::graphics::engine_t engine {{
    .renderer = fan::graphics::renderer_t::vulkan,
  }};

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

  auto update_ground_camera = [&] {
    auto& camera = engine.camera_get(cam);
    f32_t dt = engine.get_delta_time();

    fan::vec3 forward(camera.front.x, 0.f, camera.front.z);
    if (forward.length_squared() > 0.f) {
      forward = forward.normalize();
    }

    fan::vec3 right(camera.right.x, 0.f, camera.right.z);
    if (right.length_squared() > 0.f) {
      right = right.normalize();
    }

    fan::vec3 wish {};
    if (engine.is_key_down(fan::key_w)) { wish += forward; }
    if (engine.is_key_down(fan::key_s)) { wish -= forward; }
    if (engine.is_key_down(fan::key_d)) { wish += right; }
    if (engine.is_key_down(fan::key_a)) { wish -= right; }

    if (wish.length_squared() > 0.f) {
      wish = wish.normalize();
    }

    constexpr f32_t speed = 5.f;
    constexpr f32_t gravity = 60.f;
    constexpr f32_t jump_speed = 20.f;
    constexpr f32_t eye_height = 2.4f;
    constexpr f32_t player_radius = 0.35f;

    f32_t ground_y = terrain.ground_y(camera.position.x, camera.position.z, player_radius) + eye_height;
    bool grounded = camera.position.y <= ground_y + 0.05f;

    camera.velocity.x = wish.x * speed;
    camera.velocity.z = wish.z * speed;

    if (grounded && camera.velocity.y < 0.f) {
      camera.velocity.y = 0.f;
    }

    if (grounded && engine.is_key_clicked(fan::key_space)) {
      camera.velocity.y = jump_speed;
      grounded = false;
    }

    camera.velocity.y -= gravity * dt;
    camera.position += camera.velocity * dt;

    ground_y = terrain.ground_y(camera.position.x, camera.position.z, player_radius) + eye_height;
    if (camera.position.y < ground_y) {
      camera.position.y = ground_y;
      camera.velocity.y = 0.f;
    }

    camera.update_view();
    camera.view = camera.get_view_matrix();
  };

  auto update_noclip_camera = [&] {
    auto& camera = engine.camera_get(cam);
    f32_t dt = engine.get_delta_time();

    fan::vec3 wish {};
    if (engine.is_key_down(fan::key_w)) { wish += camera.front; }
    if (engine.is_key_down(fan::key_s)) { wish -= camera.front; }
    if (engine.is_key_down(fan::key_d)) { wish += camera.right; }
    if (engine.is_key_down(fan::key_a)) { wish -= camera.right; }
    if (engine.is_key_down(fan::key_space)) { wish += fan::vec3(0.f, 1.f, 0.f); }
    if (engine.is_key_down(fan::key_left_shift)) { wish -= fan::vec3(0.f, 1.f, 0.f); }

    if (wish.length_squared() > 0.f) {
      wish = wish.normalize();
    }

    constexpr f32_t speed = 40.f;
    camera.velocity = wish * speed;
    camera.position += camera.velocity * dt;
    camera.update_view();
    camera.view = camera.get_view_matrix();
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
    if (engine.is_key_clicked(fan::key_q)) { noclip = !noclip; engine.camera_get(cam).velocity = fan::vec3(0.f); }
    if (engine.is_key_clicked(fan::key_t)) {
      ui_open = !ui_open;
      engine.window.set_cursor(ui_open ? 1 : 0);
      engine.camera_get(cam).velocity = fan::vec3(0.f);
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
      if (noclip) {
        update_noclip_camera();
      }
      else {
        update_ground_camera();
      }
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