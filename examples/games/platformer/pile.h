// All stages are included here
struct pile_t;
pile_t* pile = 0;
#define stage_loader_path .
#include <fan/graphics/gui/stage_maker/loader.h>

struct pile_t {
  lstd_defstruct(level0_t)
    #include <fan/graphics/gui/stage_maker/preset.h>
      static constexpr auto stage_name = "level0";
      #include "level0/map.h"
    };
  lstd_defstruct(gui_t)
    #include <fan/graphics/gui/stage_maker/preset.h>
    static constexpr auto stage_name = "gui";
    #include "gui.h"
  };
  #include "player/player.h"
  #include "enemy/enemy.h"
  #include "enemy/skeleton/skeleton.h"
  #include "enemy/fly/fly.h"
  #include "enemy/boss.h"
  #include "enemy/boss_skeleton/boss_skeleton.h"
  pile_t();
  bool pause = false;
  void update_camera_zoom() {
    fan::vec2 r = engine.window.get_current_monitor_resolution() / fan::vec2(2560, 1440);
    ic.set_zoom(1.6f * r.max());
  }
  void update() {
    static bool force_zoom = true;
     if (force_zoom) {
     // update_camera_zoom();
    }
    if (fan::window::is_key_down(fan::key_left_control) && engine.is_key_pressed(fan::key_q)) {
      force_zoom = !force_zoom;
    }
    if (!pause) {
      player.update();
    }
    
    engine.update_physics(!pause);
    if (!engine.is_key_down(fan::mouse_middle)) {
      fan::vec2 target_pos = player.get_physics_pos() - fan::vec2(0, 50);
      engine.camera_set_target(engine.orthographic_render_view.camera, target_pos, 0);
      ic.camera_offset = target_pos;
    }
    if (!pause) {
      enemy_list.update();
    }

    //fan::graphics::gui::text(fan::graphics::screen_to_world(fan::window::get_mouse_position()));
    fan::graphics::gui::set_viewport(engine.orthographic_render_view.viewport);
  }
  level0_t& get_level() {
    return stage_loader.get_stage_data<level0_t>(level_stage);
  }
  gui_t& get_gui() {
    return stage_loader.get_stage_data<gui_t>(gui_stage);
  }
  fan::graphics::engine_t engine {{.window_size = {1920,1080}}};
  fan::graphics::gameplay::checkpoint_system_t checkpoint_system;
  tilemap_renderer_t renderer;
  stage_loader_t stage_loader;
  stage_loader_t::nr_t level_stage, gui_stage;
  fan::graphics::interactive_camera_t ic{
    engine.orthographic_render_view.camera,
    engine.orthographic_render_view.viewport
  };
  fan::audio::piece_t audio_background;
  fan::audio::sound_play_id_t audio_background_play_id;
  enum attack_result_e {
    miss,
    hit,
    blocked
  };
  player_t player;
  fan::graphics::rectangle_t stage_transition;

  std::unordered_map<std::string, tilemap_loader_t::compiled_map_t> tilemaps_compiled;

  using enemy_list_t = fan::graphics::entity::enemy_container_t<skeleton_t, fly_t, boss_skeleton_t>;
  enemy_list_t enemy_list;
  enemy_list_t& enemies() { return enemy_list; }
  const enemy_list_t& enemies() const = delete;

  template <typename T, typename... Args>
  enemy_list_t::nr_t spawn_enemy(Args&&... args) {
    enemy_list_t::nr_t nr = pile->enemy_list.add();
    pile->enemy_list[nr] = T(pile->enemy_list, nr, std::forward<Args>(args)...);
    return nr;
  }

  fan::graphics::update_callback_nr_t frame_update_handle;
};

pile_t::pile_t() {
  //pile->engine.set_culling_enabled(false);
  //ic.ignore_input = true;
  update_camera_zoom();
  //engine.window.set_size(engine.window.get_current_monitor_resolution());
  engine.clear_color = 0;
  engine.texture_pack.open_compiled("texture_pack.ftp", fan::graphics::image_presets::pixel_art());
  renderer.open();

  player.body.set_physics_position(player.body.get_position());
  engine.camera_set_target(engine.orthographic_render_view.camera, player.body.get_position(), 0);

  items::init();

  gui_stage = stage_loader.open_stage<gui_t>();
  tilemaps_compiled[level0_t::stage_name] = pile->renderer.compile("sample_level.fte");
  level_stage = stage_loader.open_stage<level0_t>();
  audio_background = fan::audio::piece_t("audio/background.sac");
  //audio_background_play_id = fan::audio::play(audio_background, 0, true);

  frame_update_handle = engine.add_update_callback_front([this] (void* engine) {
    update();
  });
}