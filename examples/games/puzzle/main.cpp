#include <fan/pch.h>

//fan_track_allocations();

std::string asset_path = "examples/games/puzzle/";

struct weather_t {
  weather_t() {
    load_rain(rain_particles);
  }
  void lightning();
  void load_rain(loco_t::shape_t& rain_particles);


  bool on = false;
  f32_t sin_var = 0;
  uint16_t repeat_count = 0;
  loco_t::shape_t rain_particles;

  f32_t lightning_duration = 0;
};

#define stage_loader_path .
#include <fan/graphics/gui/stage_maker/loader.h>

struct pile_t {
  pile_t();

  void step() {
    loco.set_imgui_viewport(loco.orthographic_camera.viewport);
  }
  loco_t loco;
  loco_t::texturepack_t tp;

  weather_t weather;

  stage_loader_t stage_loader;
  uint16_t current_stage = 0;
}pile;

struct stage_shop_t;

lstd_defstruct(stage_living_room_t)
  #include <fan/graphics/gui/stage_maker/preset.h>
  static constexpr auto stage_name = "";
  #include "stage_living_room.h"
};

pile_t::pile_t() {

  fan::graphics::image_load_properties_t lp;
  lp.visual_output = fan::graphics::image_sampler_address_mode::clamp_to_border;
  lp.min_filter = fan::graphics::image_filter::nearest;
  lp.mag_filter = fan::graphics::image_filter::nearest;
  

  current_stage = stage_loader_t::open_stage<stage_living_room_t>().NRI;
}

void weather_t::lightning() {
  fan_ev_timer_loop(4000, { on = !on; });
  if (on) {
    pile.loco.lighting.ambient = fan::color::hsv(224.0, std::max(sin(sin_var * 2), 0.f) * 100.f, std::max(sin(sin_var), 0.f) * 100.f);
    sin_var += pile.loco.delta_time * 10;
    lightning_duration += pile.loco.delta_time;
    
    if (lightning_duration >= 1.0f) {
      lightning_duration = 0;
      on = false;
    }
  }
  else {
    pile.loco.lighting.ambient = fan::color::hsv(0, 0, 0);
  }
}

void weather_t::load_rain(loco_t::shape_t& rain_particles) {
  std::string data;
  fan::io::file::read("rain.json", &data);
  fan::json in = fan::json::parse(data);
  fan::graphics::shape_deserialize_t it;
  while (it.iterate(in, &rain_particles)) {
  }
  auto image_star = pile.loco.image_load("images/waterdrop.webp");
  rain_particles.set_image(image_star);
}

int main() {
  pile.loco.clear_color = 0;

  fan::graphics::interactive_camera_t ic(
    pile.loco.orthographic_camera.camera, 
    pile.loco.orthographic_camera.viewport
  );

  pile.loco.loop([&] {

  });
}