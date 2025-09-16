// All stages are included here

#include "player.h"

#define stage_loader_path .
#include <fan/graphics/gui/stage_maker/loader.h>

struct pile_t {
  pile_t();

  void step() {

    //player updates
    engine.camera_set_target(engine.orthographic_render_view.camera, player.body.get_position());
    player.step();
    
    fan::graphics::gui::set_viewport(engine.orthographic_render_view.viewport);

    // physics step
    engine.physics_context.step(engine.delta_time);
  }
  fan::graphics::engine_t engine;
  player_t player;
  fte_renderer_t renderer;

  stage_loader_t stage_loader;
    stage_loader_t::nr_t  current_stage;

  fan::graphics::interactive_camera_t ic{
    engine.orthographic_render_view.camera,
    engine.orthographic_render_view.viewport
  };

  lstd_defstruct(example_stage_t)
    #include <fan/graphics/gui/stage_maker/preset.h>
    static constexpr auto stage_name = "";
    #include "main_map.h"
  };

  example_stage_t& get_current_stage() {
    return stage_loader.get_stage_data<example_stage_t>(current_stage);
  }

}pile;


pile_t::pile_t() {
  engine.clear_color = 0;
  engine.lighting.ambient = 1;
  fan::graphics::image_load_properties_t lp;
  lp.visual_output = fan::graphics::image_sampler_address_mode::clamp_to_border;
  lp.min_filter = fan::graphics::image_filter::nearest;
  lp.mag_filter = fan::graphics::image_filter::nearest;

  engine.texture_pack.open_compiled("sample_texture_pack.ftp", lp);

  renderer.open();
  
  player.body.set_physics_position(player.body.get_position());

  fan::vec2 dst = player.body.get_position();
  fan::vec2 camera_offset = fan::vec2(0, -engine.window.get_size().y / 4);
  engine.camera_set_position(
    engine.orthographic_render_view.camera, 
    dst + camera_offset // move camera higher to display more area upwards
  );

  current_stage = stage_loader_t::open_stage<example_stage_t>();
}