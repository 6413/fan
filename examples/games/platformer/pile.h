// All stages are included here

struct pile_t;

pile_t* pile = 0;

#define stage_loader_path .
#include <fan/graphics/gui/stage_maker/loader.h>

struct pile_t {
  lstd_defstruct(level_t)
    #include <fan/graphics/gui/stage_maker/preset.h>
    static constexpr auto stage_name = "";
    #include "level.h"
  };

  lstd_defstruct(gui_t)
    #include <fan/graphics/gui/stage_maker/preset.h>
    static constexpr auto stage_name = "";
    #include "gui.h"
  };

  #include "player/player.h"
  #include "entity.h"

  pile_t();

  void step() {
    engine.physics_context.step(engine.delta_time);
    player.step();
    for (auto& enemy : pile->entity) {
      if (enemy.update()) {
        break;
      }
    }
    
    engine.camera_set_target(engine.orthographic_render_view.camera, player.get_physics_pos(), 0);
    fan::graphics::gui::set_viewport(engine.orthographic_render_view.viewport);
  }

  level_t& get_level() {
    return stage_loader.get_stage_data<level_t>(level_stage);
  }
  gui_t& get_gui() {
    return stage_loader.get_stage_data<gui_t>(gui_stage);
  }

  fan::graphics::engine_t engine;
  fte_renderer_t renderer;

  stage_loader_t stage_loader;
  stage_loader_t::nr_t level_stage, gui_stage;

  fan::graphics::interactive_camera_t ic{
    engine.orthographic_render_view.camera,
    engine.orthographic_render_view.viewport
  };

  player_t player;
  std::vector<entity_t> entity;
};


pile_t::pile_t() {
//  fan::graphics::physics::debug_draw(true);
  ic.zoom = 2.f;
  //engine.clear_color = fan::color::from_rgb(0x1A2A2E);
  engine.clear_color = 0;

  engine.lighting.ambient = 1;
  engine.texture_pack.open_compiled("texture_pack.ftp", fan::graphics::image_presets::pixel_art());

  renderer.open();
  
  player.body.set_physics_position(player.body.get_position());
  engine.camera_set_target(engine.orthographic_render_view.camera, player.body.get_position(), 0);

  level_stage = pile->stage_loader.open_stage<level_t>();
  gui_stage = pile->stage_loader.open_stage<gui_t>();
  // init map after setting current_stage
  get_level().load_map();
}