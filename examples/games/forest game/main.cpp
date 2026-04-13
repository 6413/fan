#include <fan/utility.h>
#include <fan/event/types.h>

#include <coroutine>
#include <string>
#include <fstream>

import fan;
import fan.graphics.gui.tilemap_editor.renderer;

using namespace fan::graphics;

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

struct pile_t {
#include "player.h"

  pile_t();

  void step() {
    using namespace fan::graphics;
    player.step();
    
    gui::set_viewport(loco.orthographic_render_view.viewport);

    //pile.weather.rain_particles.set_position(fan::vec3(1200, -900, 50000));
  }

  loco_t loco;
  player_t player;
  tilemap_renderer_t renderer;

  //fan::pathfind::path_solver_t path_solver;

  weather_t weather;

  stage_loader_t stage_loader;
  uint16_t current_stage = 0;

  bool is_map_changing = true;
  std::unordered_map<std::string, tilemap_renderer_t::compiled_map_t> maps_compiled;

  fan::vec3 fadeout_target_color = -1.0f;
}pile;

struct stage_shop_t;
struct stage_forest_t;

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

pile_t::pile_t() {

  image_load_properties_t lp;
  lp.visual_output = image_sampler_address_mode_e::clamp_to_border;
  lp.min_filter = image_filter_e::nearest;
  lp.mag_filter = image_filter_e::nearest;

  gloco()->texture_pack.open_compiled("examples/games/forest game/tileset.ftp", lp);

  renderer.open();
  
  fan::vec2 dst = player.body.get_position();
  loco.camera_set_position(
    loco.orthographic_render_view.camera,
    dst
  );
  maps_compiled[stage_forest_t::stage_name] = renderer.compile("examples/games/forest game/forest.fte");
  maps_compiled[stage_shop_t::stage_name] = renderer.compile("examples/games/forest game/shop/shop.fte");

  current_stage = stage_loader.open_stage<stage_forest_t>().NRI;

  loco.update_physics(true);
}

void weather_t::lightning() {
  fan_ev_timer_loop(4000, { on = !on; });
  if (on) {
  //  pile.loco.lighting.ambient = fan::color::hsv(224.0, std::max(sin(sin_var * 2), 0.f) * 100.f, std::max(sin(sin_var), 0.f) * 100.f);
    sin_var += pile.loco.get_delta_time() * 10;
    lightning_duration += pile.loco.get_delta_time();
    
    if (lightning_duration >= 1.0f) {
      lightning_duration = 0;
      on = false;
    }
  }
  else {
    //pile.loco.lighting.ambient = fan::color::hsv(0, 0, 0);
  }
}

void weather_t::load_rain(shape_t& rain_particles) {
  //std::string data;
  //fan::io::file::read("raindrops.json", &data);
  //fan::json in = fan::json::parse(data);
  //shape_deserialize_t it;
  //while (it.iterate(in, &rain_particles)) {
  //}
  //auto image_star = pile.loco.image_load("images/waterdrop.webp");
  //rain_particles.set_image(image_star);
}

int main() {
  pile.loco.get_clear_color() = 0;
  pile.loco.get_lighting().set_target(0.1f);

  //physics::debug_draw(true);

  interactive_camera_t ic(
    pile.loco.orthographic_render_view.camera, 
    pile.loco.orthographic_render_view.viewport,
    5.5 /*zoom*/
  );

 // auto shape = pile.loco.grid.push_back(loco_t::grid_t::properties_t{.position= fan::vec3(fan::vec2(32*32+32-32*6), 50000),.size = 32 * 32, .grid_size = 32});
  pile.player.body.enable_default_movement(physics::movement_e::top_view);
  pile.loco.loop([&] {
    //pile.player.body.move_to_direction(pile.path_solver.step(pile.player.body.get_position()));
  });
}