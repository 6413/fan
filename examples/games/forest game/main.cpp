#include <fan/utility.h>
#include <fan/event/types.h>

#include <string>
#include <fan/graphics/algorithm/astar.h>

import fan;
import fan.graphics.gui.tilemap_editor.renderer;


//fan_track_allocations();

std::string asset_path = "examples/games/forest game/";

struct player_t {
  fan::vec2 velocity = 0;
  std::array<loco_t::image_t, 4> img_idle;
  std::array<std::array<loco_t::image_t, 4>, std::size(fan::movement_e::_strings)> img_movement;

  player_t();

  void step() {
    animator.process_walk(
      player,
      player.get_linear_velocity(),
      img_idle, img_movement[fan::movement_e::left], img_movement[fan::movement_e::right],
      img_movement[fan::movement_e::up], img_movement[fan::movement_e::down]
    );
    light.set_position(player.get_position());
    fan::vec2 dir = animator.prev_dir;
    uint32_t flag = 0;
    /*auto player_img = player.get_image();
    for (auto [i, d] : std::views::enumerate(img_idle)) {
      if (player_img == d) {
        flag = i + 3;
        break;
      }
    }
    for (auto [j, d] : std::views::enumerate(img_movement)) {
      for (auto [i, d2] : std::views::enumerate(d)) {
        if (player_img == d2) {
          flag = j + 3;
          break;
        }
      }
    }*/
   // light.set_flags(flag);
  }

  fan::graphics::physics::character2d_t player{ fan::graphics::physics::circle_sprite_t{{
    .position = fan::vec3(1019.59076, 934.117065, 10),
    // collision radius
    .radius = 8,
    // image size
    .size = fan::vec2(8, 16),
    /*.color = fan::color::from_rgba(0x715a5eff),*/
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
    using namespace fan::graphics;
    //player updates
    player.step();

    // map renderer & camera update
    fan::vec2 s = gui::get_content_region_avail();
    fan::vec2 dst = player.player.get_position();
    fan::vec2 src = loco.camera_get_position(loco.orthographic_render_view.camera);
    loco.camera_set_position(
      loco.orthographic_render_view.camera,
      src + (dst - src) * loco.delta_time * 10
    );
    fan::vec2 position = player.player.get_position();
    //ImGui::Begin("A");
    static f32_t z = 18;
    //ImGui::DragFloat("z", &z, 1);
    ///ImGui::End();
    player.player.set_position(fan::vec3(position, floor((position.y) / 64) + (0xFAAA - 2) / 2) + z);
    player.player.process_movement(fan::graphics::physics::character2d_t::movement_e::top_view);
    
    fan::graphics::gui::set_viewport(loco.orthographic_render_view.viewport);

    // physics step
    loco.physics_context.step(loco.delta_time);
  }
  loco_t loco;
  player_t player;
  fte_renderer_t renderer;

  fan::algorithm::path_solver_t path_solver;

  weather_t weather;

  stage_loader_t stage_loader;
  uint16_t current_stage = 0;
}pile;

struct stage_shop_t;

lstd_defstruct(stage_forest_t)
  #include <fan/graphics/gui/stage_maker/preset.h>
  static constexpr auto stage_name = "";
  #include "stage_forest.h"
};

lstd_defstruct(stage_shop_t)
  #include <fan/graphics/gui/stage_maker/preset.h>
  static constexpr auto stage_name = "";
  #include "stage_shop.h"
};

pile_t::pile_t() {

  fan::graphics::image_load_properties_t lp;
  lp.visual_output = fan::graphics::image_sampler_address_mode::clamp_to_border;
  lp.min_filter = fan::graphics::image_filter::nearest;
  lp.mag_filter = fan::graphics::image_filter::nearest;

  gloco->texture_pack.open_compiled("examples/games/forest game/forest_tileset.ftp", lp);

  renderer.open();
  
  fan::vec2 dst = player.player.get_position();
  loco.camera_set_position(
    loco.orthographic_render_view.camera,
    dst
  );

  current_stage = stage_loader_t::open_stage<stage_forest_t>().NRI;
}

player_t::player_t() {
  fan::graphics::image_load_properties_t lp;
  for (std::size_t i = 0; i < std::size(img_idle); ++i) {
    img_idle[i] = pile.loco.image_load(asset_path + "npc/static_" + fan::movement_e::_strings[i] + ".png", lp);
  }
  static auto load_movement_images = [](std::array<loco_t::image_t, 4>& images, const std::string& direction) {
    const std::array<std::string, 4> pose_variants = {
        direction + "_left_hand_forward.png",
        "static_" + direction + ".png",
        direction + "_right_hand_forward.png",
        "static_" + direction + ".png"
    };

    fan::graphics::image_load_properties_t lp;
    lp.min_filter = fan::graphics::image_filter::nearest;
    lp.mag_filter = fan::graphics::image_filter::nearest;
    for (const auto& [i, pose] : pose_variants | fan::enumerate) {
      images[i] = (pile.loco.image_load(asset_path + "npc/" + pose, lp));
    }
  };

  load_movement_images(img_movement[fan::movement_e::left], "left");
  load_movement_images(img_movement[fan::movement_e::right], "right");
  load_movement_images(img_movement[fan::movement_e::up], "up");
  load_movement_images(img_movement[fan::movement_e::down], "down");

  player.set_image(img_idle[fan::movement_e::down]);

  pile.loco.input_action.edit(fan::key_w, "move_up");
  light = fan::graphics::light_t{ {
    .position = player.get_position(),
    .size = 200,
    .color = fan::colors::white/2
  } };
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
  fan::io::file::read("effects/rain.json", &data);
  fan::json in = fan::json::parse(data);
  fan::graphics::shape_deserialize_t it;
  while (it.iterate(in, &rain_particles)) {
  }
  auto image_star = pile.loco.image_load("images/waterdrop.webp");
  rain_particles.set_image(image_star);
}

int main() {
  pile.loco.clear_color = 0;
  pile.player.player.force = 50;
  pile.player.player.max_speed = 1000;

  fan::graphics::interactive_camera_t ic(
    pile.loco.orthographic_render_view.camera, 
    pile.loco.orthographic_render_view.viewport
  );

 // auto shape = pile.loco.grid.push_back(loco_t::grid_t::properties_t{.position= fan::vec3(fan::vec2(32*32+32-32*6), 50000),.size = 32 * 32, .grid_size = 32});

  pile.loco.input_action.add(fan::mouse_left, "move_to_position");

  pile.loco.loop([&] {
    pile.player.player.move_to_direction(pile.path_solver.step(pile.player.player.get_position()));
  });
}