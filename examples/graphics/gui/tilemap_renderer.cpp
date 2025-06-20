#include <fan/pch.h>

#include <fan/graphics/gui/tilemap_editor/renderer0.h>

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

  fan::graphics::physics::character2d_t player{ fan::graphics::physics_shapes::capsule_t{{
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

int main() {
  loco_t loco;
  fan::graphics::interactive_camera_t ic;
  loco_t::image_load_properties_t lp;
  lp.visual_output = loco_t::image_sampler_address_mode::clamp_to_border;
  lp.min_filter = GL_NEAREST;
  lp.mag_filter = GL_NEAREST;
  loco_t::texturepack_t tp;
  //tp.open_compiled("texture_packs/tilemap.ftp", lp);
  tp.open_compiled("platformer.ftp", lp);

  fte_renderer_t renderer;

  fan::physics::body_id_t sensor_id = b2_nullBodyId;
  // must be set before calling open
  renderer.sensor_id_callbacks["sensor0"] = [&](
    fte_renderer_t::physics_entities_t& pe,
    fte_renderer_t::physics_data_t& pd
    ) {
    std::visit([&](const auto& entity) { sensor_id = entity.body_id; }, pe.visual);
  };


  renderer.open(&tp);

  //auto compiled_map = renderer.compile("tilemaps/map_game0_0.fte");
  auto compiled_map = renderer.compile("sensor_test.json");
  fan::vec2i render_size(16, 9);

  fte_loader_t::properties_t p;
  
  p.position = fan::vec3(0, 0, 0);
  p.size = render_size;
  // add custom stuff when importing files
  //p.object_add_cb = [&](fte_loader_t::fte_t::tile_t& tile) {
  //  if (tile.id == "1") {
  //    fan::print("a");
  //  }
  //};
  auto map_id0_t = renderer.add(&compiled_map, p);

  player_t player;

  loco.set_vsync(0);
  //loco.window.set_max_fps(3);
  f32_t total_delta = 0;

  loco.lighting.ambient = 0.7;

  loco.input_action.add(fan::key_e, "open_door");

  loco.loop([&] {
    fan::graphics::text(player.player.character.get_position());
    player.player.process_movement();
    loco.physics_context.step(loco.delta_time);
    if (loco.input_action.is_active("open_door") &&
    loco.physics_context.is_on_sensor(player.player.character, sensor_id)) {
      renderer.erase(map_id0_t);
      map_id0_t = renderer.add(&compiled_map, p);
    }
    fan::vec2 dst = player.player.character.get_position();
    fan::vec2 src = gloco->camera_get_position(gloco->orthographic_render_view.camera);
    gloco->camera_set_position(gloco->orthographic_render_view.camera, dst);
    renderer.update(map_id0_t, dst);
  });
}