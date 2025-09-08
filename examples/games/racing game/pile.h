struct pile_t;

#define stage_loader_path .
#include <fan/graphics/gui/stage_maker/loader.h>

struct pile_t {
  #define engine (*gloco)
#include "car.h"
#undef engine

  pile_t();

  void step() {
    car.step();
    pile.network_client.step();
    engine.camera_move_to_smooth(car.body);

    engine.physics_context.step(engine.delta_time);
  }
  fan::graphics::engine_t engine;

  // wasd keybinds
  car_t car;

  // arrow keys
  //car_t car2;

  fte_renderer_t renderer;

  stage_loader_t stage_loader;
  stage_loader_t::nr_t current_stage;

  fan::graphics::interactive_camera_t ic{
    engine.orthographic_render_view.camera,
    engine.orthographic_render_view.viewport
  };

  lstd_defstruct(racing_track_t)
    #include <fan/graphics/gui/stage_maker/preset.h>
    static constexpr auto stage_name = "";
    #include "racing_track.h"
  };

  #include "network_client.h"

}pile;


pile_t::pile_t() {
 // fan::graphics::physics::debug_draw(true);
  engine.set_target_fps(0);
  engine.set_vsync(false);
  engine.clear_color = 0;
  engine.lighting.ambient = 1;
  fan::graphics::image_load_properties_t lp;
  lp.visual_output = fan::graphics::image_sampler_address_mode::clamp_to_border;
  lp.min_filter = fan::graphics::image_filter::nearest;
  lp.mag_filter = fan::graphics::image_filter::nearest;

  engine.texture_pack.open_compiled("racetrack.ftp", lp);

  renderer.open();


  current_stage = stage_loader_t::open_stage<racing_track_t>();

  car.open();
  //car2.open(
  //  fan::vec3(1019.7828, 1580.1302, car_t::car_draw_depth), 
  //  fan::colors::green,
  //  "arrow_keys_",
  //  fan::key_up,
  //  fan::key_down,
  //  fan::key_left,
  //  fan::key_right
  //);

  engine.camera_move_to(car.body);
}