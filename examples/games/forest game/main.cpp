// --- ./main.cpp ---
#include <fan/utility.h>
#include <coroutine>
#include <string>
#include <fstream>
#include <unordered_map>

import fan;
import fan.graphics.tilemap_editor.renderer;

using namespace fan::graphics;
using namespace fan::window;
using namespace fan::color_literals;

struct weather_t {
  weather_t() {
    load_rain(rain_particles);
  }
  void lightning();
  void load_rain(shape_t& rain_particles);

  bool on = false;
  f32_t sin_var = 0;
  uint16_t repeat_count = 0;
  shape_t rain_particles;
  f32_t lightning_duration = 0;
};

struct equipable_t {
  fan::physics::entity_t sensor;
};

#define stage_loader_path .
#include <fan/graphics/gui/stage_maker/loader.h>

struct pile_t;
extern pile_t pile;

struct pile_t {
  lstd_defstruct(stage_shop_t)
    #include <fan/graphics/gui/stage_maker/preset.h>
    static constexpr auto stage_name = "stage_shop";
    #include "stage_shop.h"
  };

  lstd_defstruct(stage_forest_t)
    #include <fan/graphics/gui/stage_maker/preset.h>
    static constexpr auto stage_name = "stage_forest";
    #include "stage_forest.h"
  };

  #include "player.h"

  pile_t() {
    gloco()->texture_pack.open_compiled("examples/games/forest game/tileset.ftp", image_presets::pixel_art());

    renderer.open();
    engine.camera_set_position(engine.orthographic_render_view.camera, player.body.get_position());
  
    renderer.compile(stage_forest_t::stage_name, "examples/games/forest game/forest.fte");
    renderer.compile(stage_shop_t::stage_name,   "examples/games/forest game/shop/shop.fte");

    current_stage = stage_loader.open_stage<stage_forest_t>();
    engine.update_physics(true);
  }

  void step() {
    player.step();
    gui::set_viewport(engine.orthographic_render_view.viewport);
  }

  engine_t engine;
  fan::pathfind::generator pathfinder;
  f32_t path_grid_size = 32.f; 
  pet_t pet;
  player_t player;
  tilemap_renderer_t renderer;
  weather_t weather;
  stage_loader_t stage_loader;
  
  tilemap_loader_t::id_t active_map_id;
  stage_loader_t::nr_t current_stage;

  bool is_map_changing = true;
  fan::vec3 fadeout_target_color = -1.0f;
  fan::event::task_t map_transition_task;
};

pile_t pile;

void weather_t::lightning() {
  if (fan::time::every(4000)) {
    sin_var += pile.engine.get_delta_time() * 10;
    lightning_duration += pile.engine.get_delta_time();
    if (lightning_duration >= 1.0f) {
      lightning_duration = 0;
      on = false;
    }
  }
}

void weather_t::load_rain(shape_t& rain_particles) {}

int main() {
  interactive_camera_t ic(
    pile.engine.orthographic_render_view, 
    5.5
  );

  pile.player.body.enable_default_movement(physics::movement_e::top_view);
  pile.engine.loop();
}