// All stages are included here

struct pile_t;

pile_t* pile = 0;

#define stage_loader_path .
#include <fan/graphics/gui/stage_maker/loader.h>

struct pile_t {
  #include "player/player.h"
  #include "entity.h"

  pile_t();

  void step() {
    player.step();
    engine.physics_context.step(engine.delta_time);

    //player updates
    engine.camera_set_target(engine.orthographic_render_view.camera, player.body.get_position(), 0);
    fan::graphics::gui::set_viewport(engine.orthographic_render_view.viewport);

    // physics step
  }
  fan::graphics::engine_t engine;
  fte_renderer_t renderer;

  stage_loader_t stage_loader;
  stage_loader_t::nr_t current_stage;

  fan::graphics::interactive_camera_t ic{
    engine.orthographic_render_view.camera,
    engine.orthographic_render_view.viewport
  };

  player_t player;
  entity_t entity;
};

lstd_defstruct(level_t)
  #include <fan/graphics/gui/stage_maker/preset.h>
  static constexpr auto stage_name = "";
  #include "level.h"
};

pile_t::pile_t() {
//  fan::graphics::physics::debug_draw(true);
  engine.clear_color = fan::color::from_rgb(0x1A2A2E);

  engine.lighting.ambient = 1;
  engine.texture_pack.open_compiled("texture_pack.ftp", fan::graphics::image_presets::pixel_art());

  renderer.open();
  
  player.body.set_physics_position(player.body.get_position());
  engine.camera_set_target(engine.orthographic_render_view.camera, player.body.get_position(), 0);

  current_stage = pile->stage_loader.open_stage<level_t>();

  entity = entity_t(player.body.get_position() + fan::vec2(150, 0));
}