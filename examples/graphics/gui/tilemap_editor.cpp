import fan;
import fan.graphics.gui.tilemap_editor.renderer;
import fan.graphics.gui.tilemap_editor.editor;
import fan.event;
import fan.io.file;

#include <box2d/box2d.h>

// editor
fan::graphics::camera_t camera0;
// program
fan::graphics::camera_t camera1;

struct player_t {
   player_t() {
    fan::physics::set_pre_solve_callback(gloco->physics_context.world_id, presolve_static, this);
    gloco->input_action.edit(fan::key_w, "move_up");
  }
  static bool presolve_static(b2ShapeId shapeIdA, b2ShapeId shapeIdB, b2Manifold* manifold, void* context) {
    player_t* pl = static_cast<player_t*>(context);
    return pl->presolve(shapeIdA, shapeIdB, manifold);
  }
  bool presolve(b2ShapeId shapeIdA, b2ShapeId shapeIdB, b2Manifold* manifold) const {
    return fan::physics::presolve_oneway_collision(shapeIdA, shapeIdB, manifold, player);
  }
  fan::graphics::physics::character2d_t player{ fan::graphics::physics::circle_t{{
    .camera = &camera1,
    .position = fan::vec3(400, 400, 10),
    .radius = 16.f,
    .color = fan::color::hex(0x715a5eff),
    .blending = true,
    .body_type = fan::physics::body_type_e::dynamic_body,
    .mass_data{.mass = 0.01f},
    .shape_properties{
      .friction = 0.6f, 
      .density = 0.1f, 
      .fixed_rotation = true,
      .linear_damping = 30,
      .collision_multiplier = fan::vec2(1, 1)
    },
  }}};
  loco_t::shape_t player_light = fan::graphics::light_t{ {
    .camera = &camera1,
    .position = player.get_position(),
    .size = player.get_size()*8,
    .color = fan::color::hex(0xe8c170ff)
  }};
};
int main(int argc, char** argv) {
  loco_t loco;
  
 // loco.window.set_windowed_fullscreen();

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

  camera0.viewport = loco.open_viewport(
    0,
    { 1, 1 }
  );
  camera1.viewport = loco.open_viewport(
    0,
    { 1, 1 }
  );

  fan::graphics::interactive_camera_t ic(camera1.camera, camera1.viewport);

  fte_t fte;//
  fte.original_image_width = 2048;
  fte_t::properties_t p;
  p.camera = &camera0;
  fte.open(p);
  fte.open_texturepack("texture_packs/TexturePack");
  fte.fin("map_game0_1.json");

  std::unique_ptr<player_t> player;
  std::unique_ptr<fte_renderer_t> renderer;
  bool render_scene = false;
  std::unique_ptr<fte_renderer_t::id_t> map_id0_t;

  auto reload_scene = [&] {
    {
      renderer = std::make_unique<fte_renderer_t>();

      loco_t::image_load_properties_t lp;
      lp.visual_output = loco_t::image_sampler_address_mode::clamp_to_border;
      lp.min_filter = fan::graphics::image_filter::nearest;
      lp.mag_filter = fan::graphics::image_filter::nearest;
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

  //fte.fin("examples/games/forest game/forest.json");
  f32_t z = 17;

  //fan::graphics::physics_shapes::physics_update_cb = [&](loco_t::shape_t& shape, const fan::vec3& p, const fan::vec2& size, f32_t radians) {
  //  fan::graphics::physics_shapes::hitbox_visualize[&shape] = fan::graphics::rectangle_t{ {
  //    .camera = &camera1,
  //    .position = fan::vec3(fan::vec2(p), 60000),
  //    .size=size,
  //    .color = fan::color(0, 1, 0, 0.2),
  //    .outline_color=fan::color(0, 1, 0, 0.2)*2,
  //    .angle = fan::vec3(0, 0, radians),
  //    .blending=true
  //  }};
  //};

 /* fan::event::fs_watcher_t fs_watcher("examples/games/boss/");

  fs_watcher.start([&](const std::string& filename, int events) {
    if (!(events & fan::fs_change)) {
      return;
    }
    if (filename.contains("tileset.png")) {
      std::string image_path = fs_watcher.watch_path + "tileset.png";
      loco_t::image_t img = loco.image_load(image_path);
      auto& img_data = loco.image_get_data(img);
      fan::vec2 size = img_data.size;
      fan::vec2 img_size = size;
      size = img_size / 128;
      loco.image_unload(img);
      std::string str = (std::string("image2texturepack.exe ") + 
        std::to_string(size.x) + " " + std::to_string(size.y) + 
         " \"" + image_path + "\"" +
         " \"" + fte.texturepack.file_path + "\""
      );
      system(str.c_str());
      fte.open_texturepack(fte.texturepack.file_path);
      if (fte.previous_file_name.size()) {
        fte.fin(fte.previous_file_name);
      }
    }
  });*/

  loco.loop([&] {

    fte.render();
    
    if (render_scene) {/*
      for (auto& i : fan::graphics::physics_shapes::hitbox_visualize) {
        i.second.set_camera(camera1.camera);
        i.second.set_viewport(camera1.viewport);
      }*/
      if (fan::graphics::gui::begin("Program", 0, fan::graphics::gui::window_flags_no_background)) {
        fan::vec2 s = fan::graphics::gui::get_content_region_avail();
        fan::vec2 dst = player->player.get_position();
        fan::vec2 src = loco.camera_get_position(camera1.camera);

        loco.camera_set_position(
          camera1.camera,
          dst
        );
        fan::vec2 position = player->player.get_position();
        player->player.set_position(fan::vec3(position, floor(position.y / 64) + (0xFAAA - 2) / 2) +z);
        fan::color c = player->player_light.get_color();
        if (fan::graphics::gui::color_edit4("color", &c)) {
          player->player_light.set_color(c);
        }
        fan::vec2 size = player->player_light.get_size();
        if (fan::graphics::gui::drag_float("size", & size, 0.1)) {
          player->player_light.set_size(size);
        }
        player->player_light.set_position(player->player.get_position()-player->player.get_size());
        player->player.process_movement(fan::graphics::physics::character2d_t::movement_e::top_view);
        renderer->update(*map_id0_t, dst);
        fan::graphics::gui::set_viewport(camera1.viewport);
        loco.physics_context.step(loco.delta_time);

        loco.viewport_zero(
          camera0.viewport
        );
      }
      else {
        loco.viewport_zero(
          camera1.viewport
        );
      }
      fan::graphics::gui::end();
    }
  });
  //
  return 0;
}
