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
  #include "enemy/enemy.h"
  #include "enemy/skeleton/skeleton.h"

  pile_t();


  bool pause = false;

  void update_camera_zoom() {
    fan::vec2 r = engine.window.get_current_monitor_resolution() / fan::vec2(2560, 1440);
    ic.zoom = 2.2f * r.max();
  }

  void update() {
    //update_camera_zoom();

    if (!pause) {
      engine.physics_context.step(engine.delta_time);
      player.update();
      for (skeleton_t& enemy : pile->enemy_skeleton) {
        if (enemy.update()) {
          break;
        }
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

  fan::graphics::engine_t engine /*{{.window_open_mode = fan::window_t::mode::windowed_fullscreen}}*/;
  fte_renderer_t renderer;

  stage_loader_t stage_loader;
  stage_loader_t::nr_t level_stage, gui_stage;

  fan::graphics::interactive_camera_t ic{
    engine.orthographic_render_view.camera,
    engine.orthographic_render_view.viewport
  };

  fan::audio::piece_t audio_background;

  player_t player;

  #define bcontainer_set_StoreFormat 1
  #define BLL_set_Usage 1
  #define BLL_set_SafeNext 1
  #define BLL_set_prefix enemy_skeleton
  #include <fan/fan_bll_preset.h>
  #define BLL_set_Link 1
  #define BLL_set_type_node uint16_t
  #define BLL_set_NodeDataType skeleton_t
  #define BLL_set_AreWeInsideStruct 1
  #include <BLL/BLL.h>

  enemy_skeleton_t enemy_skeleton;
};


pile_t::pile_t() {
  update_camera_zoom();
  //fan::graphics::physics::debug_draw(true);
 // ic.ignore_input = true;
  //engine.clear_color = fan::color::from_rgb(0x1A2A2E);
  engine.clear_color = 0;

  //engine.lighting.ambient = 1;
  engine.texture_pack.open_compiled("texture_pack.ftp", fan::graphics::image_presets::pixel_art());

  renderer.open();
  
  player.body.set_physics_position(player.body.get_position());
  engine.camera_set_target(engine.orthographic_render_view.camera, player.body.get_position(), 0);

  level_stage = stage_loader.open_stage<level_t>();
  gui_stage = stage_loader.open_stage<gui_t>();

  audio_background = fan::audio::piece_t("audio/background.sac");

  fan::audio::set_volume(0.0f);
  fan::audio::play(audio_background, 0, true);
}