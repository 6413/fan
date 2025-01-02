#include <fan/pch.h>

#include <fan/graphics/gui/tilemap_editor/editor.h>
#include <fan/graphics/gui/tilemap_editor/renderer0.h>

// editor
fan::graphics::camera_t camera0;
// program
fan::graphics::camera_t camera1;

struct player_t {
   player_t() {
    b2World_SetPreSolveCallback(gloco->physics_context.world_id, presolve_static, this);
  }
  static bool presolve_static(b2ShapeId shapeIdA, b2ShapeId shapeIdB, b2Manifold* manifold, void* context) {
    player_t* pl = static_cast<player_t*>(context);
    return pl->presolve(shapeIdA, shapeIdB, manifold);
  }
  bool presolve(b2ShapeId shapeIdA, b2ShapeId shapeIdB, b2Manifold* manifold) const {
    return fan::physics::presolve_oneway_collision(shapeIdA, shapeIdB, manifold, player.character);
  }

  fan::graphics::character2d_t player{ fan::graphics::physics_shapes::capsule_t{{
    .camera = &camera1,
    .position = fan::vec3(400, 400, 10),
    .center0 = {0.f, -128.f},
    .center1 = {0.f, 128.0f},
    .radius = 16.f,
    .color = fan::color::hex(0x715a5eff),
    .outline_color = fan::color::hex(0x715a5eff) * 2,
    .blending = true,
    .body_type = fan::physics::body_type_e::dynamic_body,
    .mass_data{.mass = 0.01f},
    .shape_properties{.friction = 0.6f, .density = 0.1f, .fixed_rotation = true},
  }} };
};
int main(int argc, char** argv) {
  loco_t loco;
  
  loco.window.set_windowed_fullscreen();


  camera0.camera = loco.camera_create();
  camera1.camera = loco.camera_create();
  fan::vec2 window_size = loco.window.get_size();
  loco.camera_set_ortho(
    camera0.camera,
    fan::vec2(-window_size.x, window_size.x),
    fan::vec2(-window_size.y, window_size.y)
  );
  loco.camera_set_ortho(
    camera1.camera,
    fan::vec2(-window_size.x, window_size.x),
    fan::vec2(-window_size.y, window_size.y)
  );

  fan::graphics::interactive_camera_t ic(camera1.camera);

  camera0.viewport = loco.open_viewport(
    0,
    { 1, 1 }
  );
  camera1.viewport = loco.open_viewport(
    0,
    { 1, 1 }
  );

  fte_t fte;//
  fte_t::properties_t p;
  p.camera = &camera0;
  fte.open(p);
  fte.open_texturepack("platformer.ftp");

  std::unique_ptr<player_t> player;
  std::unique_ptr<fte_renderer_t> renderer;
  bool render_scene = false;
  std::unique_ptr<fte_renderer_t::id_t> map_id0_t;

  auto reload_scene = [&] {
    {
      renderer = std::make_unique<fte_renderer_t>();

      loco_t::image_load_properties_t lp;
      lp.visual_output = loco_t::image_sampler_address_mode::clamp_to_border;
      lp.min_filter = fan::opengl::GL_NEAREST;
      lp.mag_filter = fan::opengl::GL_NEAREST;
      renderer->open(&fte.texturepack);

      // STATIC POINTER
      static fte_loader_t::compiled_map_t compiled_map;
      compiled_map = renderer->compile(fte.file_name + "temp");
      fan::vec2i render_size(16, 9);
      //render_size *= 2;
      //render_size += 3;

      fte_loader_t::properties_t p;

      p.position = fan::vec3(0, 0, 0);
      p.size = (render_size);
      p.camera = &camera1;
      map_id0_t = std::make_unique<fte_renderer_t::id_t>(renderer->add(&compiled_map, p));

      loco.set_vsync(0);
      //loco.window.set_max_fps(3);
      f32_t total_delta = 0;
    }
  };

  loco.window.add_keys_callback([&](const auto& d) {
    if (d.state != fan::keyboard_state::press) {
      return;
    }
    switch (d.key) {
      case fan::key_f5: {
        render_scene = !render_scene;
        if (render_scene) {
          player = std::make_unique<player_t>();
          fte.fout(fte.file_name + "temp");
          reload_scene();
        }
        else {
          if (map_id0_t) {
            renderer.get()->clear(renderer.get()->map_list[*map_id0_t.get()]);
          }
          map_id0_t.reset();
          renderer.reset();
          player.reset();
        }
        break;
      }
    }
  });

  fte.modify_cb = [&](int mode) {
    if (map_id0_t) {
      renderer.get()->clear(renderer.get()->map_list[*map_id0_t.get()]);
    }
    map_id0_t.reset();
    renderer.reset();
    fte.fout(fte.file_name + "temp");
    reload_scene();
  };

  loco.set_vsync(0);

  fte.fin("example.json");

  loco.loop([&] {
    if (render_scene) {
      if (ImGui::Begin("Program", 0, ImGuiWindowFlags_NoBackground)) {
        fan::vec2 s = ImGui::GetContentRegionAvail();
        fan::vec2 dst = player->player.character.get_position();
        fan::vec2 src = loco.camera_get_position(camera1.camera);

        loco.camera_set_position(
          camera1.camera,
          dst
        );
        player->player.process_movement();
        renderer->update(*map_id0_t, dst);
        loco.set_imgui_viewport(camera1.viewport);
        loco.physics_context.step(loco.delta_time);
      }
      else {
        loco.viewport_zero(
          camera1.viewport
        );
      }
      ImGui::End();

    }
  });
  //
  return 0;
}
