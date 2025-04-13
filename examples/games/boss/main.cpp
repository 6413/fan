#include <fan/pch.h>
#include <fan/graphics/gui/tilemap_editor/renderer0.h>

//fan_track_allocations();

std::string asset_path = "examples/games/boss";

struct player_t {
  fan::vec2 velocity = 0;
  std::array<loco_t::image_t, 4> img_idle;
  std::array<loco_t::image_t, std::size(fan::movement_e::_strings)> img_movement;

  player_t();

  void step() {
    light.set_position(player.get_position());
    fan::vec2 dir = animator.prev_dir;
    uint32_t flag = 0;
    light.set_flags(flag);
  }

  fan::graphics::physics::character2d_t player{ fan::graphics::physics::circle_sprite_t{{
    .position = fan::vec3(0, 0, 10),
    // collision radius
    .radius = 8,
    // image size
    .size = fan::vec2(8, 16),
    /*.color = fan::color::hex(0x715a5eff),*/
    .blending = true,
    .body_type = fan::physics::body_type_e::dynamic_body,
    .mass_data{.mass = 0.01f},
    .shape_properties{
      .friction = 0.6f, 
      .density = 0.1f, 
      .fixed_rotation = true,
      .linear_damping = 30,
      .collision_multiplier = fan::vec2(1, 1)
    },
  }}};
  loco_t::shape_t light;
  fan::graphics::animator_t animator;
};

struct weather_t {
  weather_t() {
  }


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
    //player updates
    player.step();

    // map renderer & camera update
    fan::vec2 s = ImGui::GetContentRegionAvail();
    fan::vec2 dst = player.player.get_position();
    fan::vec2 src = loco.camera_get_position(loco.orthographic_camera.camera);
    loco.camera_set_position(
      loco.orthographic_camera.camera,
      src + (dst - src) * loco.delta_time * 10
    );
    fan::vec2 position = player.player.get_position();
    //ImGui::Begin("A");
    static f32_t z = 18;
    //ImGui::DragFloat("z", &z, 1);
    ///ImGui::End();
    player.player.set_position(fan::vec3(position, floor((position.y) / 64) + (0xFAAA - 2) / 2) + z);
    player.player.process_movement(fan::graphics::physics::character2d_t::movement_e::top_view);
    
    loco.set_imgui_viewport(loco.orthographic_camera.viewport);

    // physics step
    loco.physics_context.step(loco.delta_time);
  }
  loco_t loco;
  player_t player;
  loco_t::texturepack_t tp;
  fte_renderer_t renderer;

  weather_t weather;

  stage_loader_t stage_loader;
  uint16_t current_stage = 0;
}pile;


struct stage_shop_t;

lstd_defstruct(stage_spawn_t)
  #include <fan/graphics/gui/stage_maker/preset.h>
  static constexpr auto stage_name = "";
  #include "stage_spawn.h"
};

pile_t::pile_t() {

  fan::graphics::image_load_properties_t lp;
  lp.visual_output = fan::graphics::image_sampler_address_mode::clamp_to_border;
  lp.min_filter = fan::graphics::image_filter::nearest;
  lp.mag_filter = fan::graphics::image_filter::nearest;

  tp.open_compiled("examples/games/boss/tileset.ftp", lp);

  renderer.open(&tp);
  
  fan::vec2 dst = player.player.get_position();
  loco.camera_set_position(
    loco.orthographic_camera.camera,
    dst
  );

  current_stage = stage_loader_t::open_stage<stage_spawn_t>().NRI;
}

player_t::player_t() {
  img_movement.fill(gloco->default_texture);
  player.set_image(gloco->default_texture);

  pile.loco.input_action.edit(fan::key_w, "move_up");
  light = fan::graphics::light_t{ {
    .position = player.get_position(),
    .size = 200,
    .color = fan::colors::white,
    .flags = 3
  } };
}
int main() {
  pile.loco.physics_context.set_gravity(0);
  pile.loco.clear_color = 0;
  pile.player.player.force = 50;
  pile.player.player.max_speed = 1000;
  gloco->lighting.ambient =1;

  fan::graphics::interactive_camera_t ic(
    pile.loco.orthographic_camera.camera, 
    pile.loco.orthographic_camera.viewport
  );

  pile.loco.loop([&] {

  });
}