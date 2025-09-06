struct pile_t;

#define stage_loader_path .
#include <fan/graphics/gui/stage_maker/loader.h>

struct pile_t {
  #define engine (OFFSETLESS(this, pile_t, car)->engine)
#include "car.h"
#undef engine

  pile_t();

  void step() {
    car.step();
    engine.physics_context.step(engine.delta_time);
  }
  fan::graphics::engine_t engine;
  car_t car;
  fte_renderer_t renderer;

  stage_loader_t stage_loader;
  uint16_t current_stage = 0;

    fan::graphics::interactive_camera_t ic{
    engine.orthographic_render_view.camera,
    engine.orthographic_render_view.viewport
  };

}pile;

lstd_defstruct(racing_track_t)
  #include <fan/graphics/gui/stage_maker/preset.h>
  static constexpr auto stage_name = "";
  #include "racing_track.h"
};

pile_t::pile_t() {
  engine.clear_color = 0;
  engine.lighting.ambient = 1;
  fan::graphics::image_load_properties_t lp;
  lp.visual_output = fan::graphics::image_sampler_address_mode::clamp_to_border;
  lp.min_filter = fan::graphics::image_filter::nearest;
  lp.mag_filter = fan::graphics::image_filter::nearest;

  engine.texture_pack.open_compiled("racetrack.ftp", lp);

  renderer.open();

  car.car.set_physics_position(car.car.get_position());

  current_stage = stage_loader_t::open_stage<racing_track_t>().NRI;
}