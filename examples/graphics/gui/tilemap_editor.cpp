// Update your main file to use the new module split for the tilemap editor:

#include <string>
#include <memory>

import fan;
import fan.graphics.gui.tilemap_editor.core;
import fan.graphics.gui.tilemap_editor.ui;
import fan.graphics.event;

using namespace fan::graphics;

struct render_context_t {
  render_view_t editor;
  render_view_t program;
};

std::string add_temp_before_ext(const std::string& filename) {
  size_t pos = filename.find_last_of('.');
  if (pos == std::string::npos || pos == 0) return filename + "temp";
  return filename.substr(0, pos) + "temp" + filename.substr(pos);
}

struct player_t {
  player_t(const fan::vec2& spawn_position, render_view_t* view) : character(spawn_position, view) {
    character.player.enable_default_movement();
  }

  void update_light() {
    character.light.set_position(character.player.get_position() - character.player.get_size());
  }

  void set_position(const fan::vec3& pos) {
    character.player.set_position(pos);
    update_light();
  }

  struct character_wrapper_t {
    character_wrapper_t(const fan::vec2& spawn_position, render_view_t* view) :
      player(physics::character_capsule({
        .render_view = view,
        .position = fan::vec3(spawn_position, 0xfffA),
        }, {.fixed_rotation=true})),
      light(light_t{{
        .render_view = view,
        .position = player.get_position(),
        .size = player.get_size() * 8,
        .color = fan::color::from_rgba(0xe8c170ff),
      }})
    {}
    physics::character2d_t player;
    shape_t light;
  };
  character_wrapper_t character;
};

struct scene_manager_t {
  void setup_camera(engine_t& engine, render_view_t& view) {
    fan::vec2 window_size = engine.window.get_size();
    view.camera = engine.camera_create();
    engine.camera_set_ortho(
      view.camera,
      fan::vec2(-window_size.x / 2.f, window_size.x / 2.f),
      fan::vec2(-window_size.y / 2.f, window_size.y / 2.f)
    );
    view.viewport = engine.open_viewport(0, { 1, 1 });
  }

  void reload_scene(fte_t& fte, render_view_t* view) {
    renderer = std::make_unique<tilemap_renderer_t>();
    renderer->open();

    tilemap_loader_t::properties_t p;
    p.position = fan::vec3(0, 0, 0);
    p.size = fan::vec2i(16, 9);
    p.render_view = view;
    map_id = renderer->open_map(add_temp_before_ext(fte.file_name), p);
  }

  void clear_scene() {
    if (map_id) renderer->close_map(map_id);
    map_id = 0;
    renderer.reset();
    player.reset();
  }

  void toggle_scene(fte_t& fte, engine_t& engine, render_view_t* view) {
    render_scene = !render_scene;
    engine.update_physics(render_scene);
    if (render_scene) {
      fte.fout(add_temp_before_ext(fte.file_name));
      reload_scene(fte, view);
      fan::vec3 pos = 0;
      player = std::make_unique<player_t>(pos, view);
      engine.set_vsync(0);
    }
    else {
      clear_scene();
    }
  }

  std::unique_ptr<player_t> player;
  std::unique_ptr<tilemap_renderer_t> renderer;
  tilemap_renderer_t::id_t map_id{};
  bool render_scene = false;
};

int main() {
  engine_t engine;
  engine.set_culling_enabled(false);
  render_context_t views;
  scene_manager_t scene;

  scene.setup_camera(engine, views.editor);
  scene.setup_camera(engine, views.program);

  interactive_camera_t ic(views.program.camera, views.program.viewport);
  ic.set_zoom(1);

  fte_t fte;
  fte.texture_packs.push_back(&engine.texture_pack);
  fte_t::properties_t p;
  p.camera = &views.editor;
  fte.open(p);

  auto keys_handle = engine.window.add_keys_callback([&](const auto& d) {
    if (d.state != fan::keyboard_state::press) return;
    if (d.key == fan::key_f5) scene.toggle_scene(fte, engine, &views.program);
  });

  fte.modify_cb = [&](int mode) {
    scene.clear_scene();
    fte.fout(add_temp_before_ext(fte.file_name));
    scene.reload_scene(fte, &views.program);
  };

  const f32_t z = 17;

  engine.loop([&] {
    fte.render();

    if (scene.render_scene && scene.player) {
      if (gui::begin("Program", 0, gui::window_flags_no_background)) {
        gui::set_viewport(views.program.viewport);
        engine.viewport_zero(views.editor.viewport);

        fan::vec2 position = scene.player->character.player.get_position();
        scene.renderer->update(scene.map_id, position);
        scene.player->set_position(
          fan::vec3(position, floor(position.y / (fte.tile_size.y * 2.f)) + (0xFAAA - 2) / 2) + z + 1
        );
        ic.set_position(position);
      }
      else {
        engine.viewport_zero(views.program.viewport);
      }
      gui::end();
    }
  });

  return 0;
}