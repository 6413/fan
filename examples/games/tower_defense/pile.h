// All stages are included here

#define stage_loader_path .
#include <fan/graphics/gui/stage_maker/loader.h>

struct pile_t {
  fan::graphics::engine_t engine;

  pile_t();

  void update() {

  }

  tilemap_renderer_t renderer;

  stage_loader_t stage_loader;
  stage_loader_t::nr_t level_stage;

//  fan::graphics::interactive_camera_t ic{engine.orthographic_render_view};
}pile;

lstd_defstruct(example_stage_t)
  #include <fan/graphics/gui/stage_maker/preset.h>
  static constexpr auto stage_name = "map";
  #include "map.h"
};

pile_t::pile_t() {
  //engine.texture_pack.open_compiled("sample_texture_pack.ftp");
  engine.camera_follow(fan::vec2{0.f, 0.f}, 0);/*0 for insta snap*/
  level_stage = pile.stage_loader.open_stage<example_stage_t>();
}