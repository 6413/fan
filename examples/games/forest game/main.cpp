#include <fan/pch.h>
#include <fan/graphics/gui/tilemap_editor/renderer0.h>

fan_track_allocations();

struct player_t {
  player_t() {
    gloco->input_action.edit(fan::key_w, "move_up");
  }
  fan::vec2 velocity = 0;
  fan::graphics::character2d_t player{ fan::graphics::physics_shapes::circle_t{{
    .position = fan::vec3(1019.59076, 934.117065, 10),
    .radius = 16.f,
    .color = fan::color::hex(0x715a5eff),
    .blending = true,
    .body_type = fan::physics::body_type_e::dynamic_body,
    .mass_data{.mass = 0.01f},
    .shape_properties{
      .friction = 0.6f, 
      .density = 0.1f, 
      .fixed_rotation = true,
      .linear_damping = 15,
      .collision_multiplier = fan::vec2(1, 1)
    },
  }}};
};
  

int main() {
  loco_t loco;
  loco_t::image_load_properties_t lp;
  lp.visual_output = loco_t::image_sampler_address_mode::clamp_to_border;
  lp.min_filter = GL_NEAREST;
  lp.mag_filter = GL_NEAREST;

  loco_t::texturepack_t tp;
  tp.open_compiled("examples/games/forest game/forest_tileset.ftp", lp);

  fte_renderer_t renderer;
  renderer.open(&tp);
  auto compiled_map = renderer.compile("examples/games/forest game/forest.json");
  fan::vec2i render_size(16, 9);
  render_size /= 2;
  player_t player;
  fte_loader_t::properties_t p;
  p.size = render_size;
  p.position = player.player.get_position();
  auto map_id0 = renderer.add(&compiled_map, p);


  fan::graphics::interactive_camera_t ic(
    gloco->orthographic_camera.camera, 
    gloco->orthographic_camera.viewport
  );

  int x = 0;
  auto shape = loco.grid.push_back(loco_t::grid_t::properties_t{.position= fan::vec3(fan::vec2(32*32+32-32*6), 50000),.size = 32 * 32, .grid_size = 32});

  loco.loop([&] {
    if (x == 2) {
      loco.console.commands.call("set_target_fps 0");
      loco.console.commands.call("set_vsync 1");
    }
    ++x;
    fan::vec2 s = ImGui::GetContentRegionAvail();
    fan::vec2 dst = player.player.get_position();
    fan::vec2 src = loco.camera_get_position(gloco->orthographic_camera.camera);
    fan_ev_timer_loop(500, {
     // fan::print(player.player.get_position());
    });

    loco.camera_set_position(
      gloco->orthographic_camera.camera,
      src + (dst - src) * loco.delta_time * 10
    );
    fan::vec2 position = player.player.get_position();
    static f32_t z = 17;
    ImGui::Begin("A");
    fan_imgui_dragfloat1(z, 1);
    ImGui::End();
    player.player.set_position(fan::vec3(position, floor((position.y) / 64) + (0xFAAA - 2) / 2) + z);
    player.player.process_movement(fan::graphics::character2d_t::movement_e::top_view);
    renderer.update(map_id0, dst);
    loco.set_imgui_viewport(gloco->orthographic_camera.viewport);
    loco.physics_context.step(loco.delta_time);
  });
}