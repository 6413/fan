#include <cmath>
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
  renderer.set_light(fan::vec3(0.f, 220.f, -180.f), fan::vec3(1.f, 0.96f, 0.88f), 0.1f);

  rt::gpu_terrain_streamer_t terrain;
  auto cam = engine.perspective_render_view.camera;

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

  auto world_to_block = [](const fan::vec3& p) {
    return fan::vec3i(
      (std::int32_t)std::floor(p.x / rt::gpu_terrain_streamer_t::voxel_size),
      (std::int32_t)std::floor(p.y / rt::gpu_terrain_streamer_t::voxel_size + rt::gpu_terrain_streamer_t::sea_level),
      (std::int32_t)std::floor(p.z / rt::gpu_terrain_streamer_t::voxel_size)
    );
  };

  auto block_to_world = [](const fan::vec3i& b) {
    return fan::vec3(
      (f32_t)b.x * rt::gpu_terrain_streamer_t::voxel_size,
      ((f32_t)b.y - (f32_t)rt::gpu_terrain_streamer_t::sea_level) * rt::gpu_terrain_streamer_t::voxel_size,
      (f32_t)b.z * rt::gpu_terrain_streamer_t::voxel_size
    );
  };

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

    fan::vec3 wish{};
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

    f32_t ground_y = terrain.ground_y(camera.position.x, camera.position.z) + eye_height;
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

    ground_y = terrain.ground_y(camera.position.x, camera.position.z) + eye_height;
    if (camera.position.y < ground_y) {
      camera.position.y = ground_y;
      camera.velocity.y = 0.f;
    }

    camera.update_view();
    camera.view = camera.get_view_matrix();
  };

  auto cb = engine.window.add_mouse_motion_callback([&](const auto& d) {
    if (engine.is_toggled(fan::key_t)) return;
    //if (engine.window.is_cursor_enabled()) { return; }
    auto& camera = engine.camera_get(cam);
    camera.rotate_camera(d.motion);
    camera.view = camera.get_view_matrix();
  });
engine.window.toggle_cursor();
  engine.loop([&] {
    if (engine.is_key_clicked(fan::key_r)) { renderer.reload_pipeline(); }
    if (engine.is_mouse_clicked(fan::key_t)) { engine.window.toggle_cursor(); }

    if (auto h = fan::graphics::gui::hud_interactive{"##rt"}; h && engine.is_toggled(fan::key_t)) {
      fan::graphics::gui::camera_controls();
      terrain.render_gui(renderer);
    }
    else {
      update_ground_camera();
    }

    terrain.update(renderer, engine.camera_get(cam).position);

    rt::context_t::pick_result_t pick;
    if (renderer.get_pick_result(pick)) {
      fan::vec3 hit_position = fan::vec3(pick.position);
      fan::vec3 hit_normal = fan::vec3(pick.normal).normalize();
      fan::vec3 place_p = hit_position + hit_normal * 0.01f;
      fan::vec3i block = world_to_block(place_p);
      fan::vec3 position = block_to_world(block) + fan::vec3(0.5f) * rt::gpu_terrain_streamer_t::voxel_size;

      renderer.set_transform_deferred(
        highlight,
        fan::translate(position).scale(fan::vec3(rt::gpu_terrain_streamer_t::voxel_size * 0.005f))
      );
    }
    else {
      hide_highlight();
    }
  });

  terrain.destroy();
}