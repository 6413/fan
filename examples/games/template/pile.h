// All stages are included here

#include "player.h"

#define stage_loader_path .
#include <fan/graphics/gui/stage_maker/loader.h>

struct pile_t {
  pile_t();

  void step() {
    //player updates
    player.step();
    // map renderer & camera update
    f32_t screen_height = loco.window.get_size().y;
    f32_t pixels_from_bottom = 400.0f;

    fan::vec2 player_pos = player.player.get_position();
    fan::vec2 camera_target;
    camera_target.x = player_pos.x;
    camera_target.y = player_pos.y/* - (screen_height / 2 - pixels_from_bottom) / (ic.zoom * 1.5))*/;

    fan::vec2 src = loco.camera_get_position(loco.orthographic_render_view.camera);
    loco.camera_set_position(
      loco.orthographic_render_view.camera,
      src + (camera_target - src) * loco.delta_time * 10
    );
    fan::vec2 position = player.player.get_position();
    static f32_t z = 18;
    player.player.set_position(fan::vec3(position, floor((position.y) / 64) + (0xFAAA - 2) / 2) + z);
    player.player.process_movement(fan::graphics::physics::character2d_t::movement_e::side_view);
    
    fan::graphics::gui::set_viewport(loco.orthographic_render_view.viewport);

    // physics step
    loco.physics_context.step(loco.delta_time);
  }
  loco_t loco;
  player_t player;
  fte_renderer_t renderer;

  stage_loader_t stage_loader;
  uint16_t current_stage = 0;

  fan::graphics::interactive_camera_t ic{
    loco.orthographic_render_view.camera,
    loco.orthographic_render_view.viewport
  };
}pile;

lstd_defstruct(example_stage_t)
  #include <fan/graphics/gui/stage_maker/preset.h>
  static constexpr auto stage_name = "";
  #include "example_stage.h"
};

pile_t::pile_t() {
  fan::graphics::image_load_properties_t lp;
  lp.visual_output = fan::graphics::image_sampler_address_mode::clamp_to_border;
  lp.min_filter = fan::graphics::image_filter::nearest;
  lp.mag_filter = fan::graphics::image_filter::nearest;

  loco.texture_pack.open_compiled(current_path + "sample_texture_pack.ftp", lp);

  renderer.open();
  
  player.player.set_physics_position(player.player.get_position());

  fan::vec2 dst = player.player.get_position();
  fan::vec2 camera_offset = fan::vec2(0, -loco.window.get_size().y / 4);
  loco.camera_set_position(
    loco.orthographic_render_view.camera, 
    dst + camera_offset // move camera higher to display more area upwards
  );

  current_stage = stage_loader_t::open_stage<example_stage_t>().NRI;
}