// All stages are included here

#define stage_loader_path .
#include <fan/graphics/gui/stage_maker/loader.h>

struct pile_t {
  fan::graphics::engine_t engine;

  #include "player.h"

  pile_t();

  void step() {
    //player updates
    engine.camera_set_target(engine.orthographic_render_view, player.body.get_position());
    player.step();

    fan::graphics::gui::set_viewport(engine.orthographic_render_view);
  }

  player_t player;
  tilemap_renderer_t renderer;

  stage_loader_t stage_loader;
  stage_loader_t::nr_t level_stage;

  fan::graphics::interactive_camera_t ic{engine.orthographic_render_view};
}pile;

lstd_defstruct(example_stage_t)
  #include <fan/graphics/gui/stage_maker/preset.h>
  static constexpr auto stage_name = "";
  #include "example_stage.h"
};

pile_t::pile_t() {
  engine.texture_pack.open_compiled("sample_texture_pack.ftp");
  engine.camera_set_target(engine.orthographic_render_view, player.body.get_position(), 0);/*0 for insta snap*/
  engine.update_physics(true);

  level_stage = pile.stage_loader.open_stage<example_stage_t>();
}