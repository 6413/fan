struct pile_t;

#define stage_loader_path .
#include <fan/graphics/gui/stage_maker/loader.h>

struct pile_t {
  #define engine (*gloco)
#include "car.h"
#include "car_ai.h"
#undef engine


  pile_t();

  std::vector<fan::graphics::light_t> lights;
  std::vector<fan::graphics::shadow_t> shadows;


  void step() {
    car.step();
    car_ai.update();
    pile.network_client.step();
    engine.camera_move_to_smooth(car.body);

    engine.physics_context.step(engine.delta_time);

    shadows.resize(1);
    for (auto& light : lights) {
      shadows.back() = fan::graphics::shadow_t{ {
        .position = car.body.get_position(),
        .size = car.body.get_size() / 1.5f,
        .color = light.get_color(),
        .angle = car.body.get_angle(),
        .light_position = light.get_position(),
        .light_radius = light.get_radius()
      } };
    }
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

  car_ai_t car_ai;

}pile;


pile_t::pile_t() {
 // fan::graphics::physics::debug_draw(true);
  
  fan::vec2 viewport_size = fan::window::get_size();

  int light_n = 1;
  for (int i = 0; i < light_n; ++i) {
  //  fan::color c = fan::color::hsv((i / f32_t(light_n)) * 360.f, 100.f, 100.f);
    //fan::color c = fan::random::bright_color() / 3.f;
    fan::color c = fan::color(0.5, 0.4, 0.4, 1);
    lights.push_back({ {
      .position = fan::vec2(500, 500),
      .size = 3000,
      .color = c
    } });
  }

  engine.set_target_fps(0);
  engine.set_vsync(true);
  engine.clear_color = 0;
  fan::graphics::image_load_properties_t lp;
  lp.visual_output = fan::graphics::image_sampler_address_mode::clamp_to_border;
  lp.min_filter = fan::graphics::image_filter::nearest;
  lp.mag_filter = fan::graphics::image_filter::nearest;

  engine.texture_pack.open_compiled("racetrack.ftp", lp);

  renderer.open();
  engine.lighting.ambient = 1;


current_stage = stage_loader.open_stage<racing_track_t>();

  car.open();
  car_ai.open();
  car_ai.is_local = false;
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