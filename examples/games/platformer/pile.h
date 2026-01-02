// All stages are included here
struct pile_t;
pile_t* pile = 0;
#define stage_loader_path .
#include <fan/graphics/gui/stage_maker/loader.h>
struct pile_t {
  lstd_defstruct(level_t)
    #include <fan/graphics/gui/stage_maker/preset.h>
      static constexpr auto stage_name = "level";
      #include "level.h"
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
    static bool force_zoom = false;
     if (force_zoom) {
      update_camera_zoom();
    }
    if (engine.is_key_pressed(fan::key_q)) {
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
      for (auto enemy : enemies()) {
        if (enemy.update()) {
          break;
        }
      }
    }

    //fan::graphics::gui::text(fan::graphics::screen_to_world(fan::window::get_mouse_position()));
    fan::graphics::gui::set_viewport(engine.orthographic_render_view.viewport);
  }
  level_t& get_level() {
    return stage_loader.get_stage_data<level_t>(level_stage);
  }
  gui_t& get_gui() {
    return stage_loader.get_stage_data<gui_t>(gui_stage);
  }
  fan::graphics::engine_t engine {{.window_size = {1920,1080}}};
  tilemap_renderer_t renderer;
  stage_loader_t stage_loader;
  stage_loader_t::nr_t level_stage, gui_stage;
  fan::graphics::interactive_camera_t ic{
    engine.orthographic_render_view.camera,
    engine.orthographic_render_view.viewport
  };
  fan::audio::piece_t audio_background;
  player_t player;

  using enemy_list_t = std::variant<skeleton_t, fly_t, boss_skeleton_t>;

  #define bcontainer_set_StoreFormat 1
  #define BLL_set_Usage 1
  #define BLL_set_SafeNext 1
  #define BLL_set_prefix enemies
  #include <fan/fan_bll_preset.h>
  #define BLL_set_Link 1
  #define BLL_set_type_node uint16_t
  #define BLL_set_NodeDataType enemy_list_t
  #define BLL_set_AreWeInsideStruct 1
  #include <BLL/BLL.h>
  enemies_t enemy_list;

  struct enemy_range_t {
    enemies_t* container;
    struct iterator_t {
      using bll_iter = fan::bll_iterator_t<enemies_t>;
      bll_iter it;
      struct wrapper_t {
        enemy_list_t* variant_ref;
        bool update() {
          return std::visit([](auto& e) { return e.update(); }, *variant_ref); 
        }
        void destroy() {
          std::visit([](auto& e) { e.destroy(); }, *variant_ref);
        }
        bool on_hit(fan::graphics::physics::character2d_t* source, const fan::vec2& hit_direction) {
          return std::visit([source, &hit_direction](auto& e) { return e.on_hit(source, hit_direction); }, *variant_ref);
        }
        fan::graphics::physics::character2d_t& get_body() { 
          return std::visit([](auto& e) -> auto& { return e.get_body(); }, *variant_ref); 
        }
      };
      wrapper_t operator*() {
        return wrapper_t{&(*it)};
      }
      iterator_t& operator++() { 
        ++it;
        return *this; 
      }
      bool operator!=(const iterator_t& o) const { 
        return it != o.it; 
      }
    };
    iterator_t begin() { 
      return {fan::bll_iterator_t<enemies_t>(container, container->GetNodeFirst())};
    }
    iterator_t end() { 
      return {fan::bll_iterator_t<enemies_t>(container, container->dst)};
    }
  };
  enemy_range_t enemies() {
    return enemy_range_t{&enemy_list};
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
  level_stage = stage_loader.open_stage<level_t>();
  gui_stage = stage_loader.open_stage<gui_t>();
  audio_background = fan::audio::piece_t("audio/background.sac");
  fan::audio::set_volume(0.0f);
  fan::audio::play(audio_background, 0, true);

  frame_update_handle = engine.add_update_callback_front([this] (void* engine) {
    update();
  });
}