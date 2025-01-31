#include <fan/pch.h>
#include <fan/graphics/algorithm/astar.h>
#include <fan/graphics/gui/tilemap_editor/renderer0.h>

fan_track_allocations();

std::string asset_path = "examples/games/forest game/";

namespace fan {
  struct movement_e {
    fan_enum_string(
      ,
      left,
      right,
      up,
      down
    );
  };
}

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
    light.set_flags(flag);
  }

  fan::graphics::character2d_t player{ fan::graphics::physics_shapes::circle_sprite_t{{
    .position = fan::vec3(1019.59076, 934.117065, 10),
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
  void lightning();

  bool on = false;
  f32_t sin_var = 0;
  uint16_t repeat_count = 0;
};

struct pile_t {
  pile_t() {
    loco_t::image_load_properties_t lp;
    lp.visual_output = loco_t::image_sampler_address_mode::clamp_to_border;
    lp.min_filter = GL_NEAREST;
    lp.mag_filter = GL_NEAREST;

    tp.open_compiled("examples/games/forest game/forest_tileset.ftp", lp);
    // dont use
    //renderer.sensor_id_callbacks["npc0_door"] = 
    //  [&](fte_renderer_t::map_list_data_t::physics_entities_t& pe, fte_renderer_t::compiled_map_t::physics_data_t& pd) {
    //  std::visit([&]<typename T>(T& entity) { 
    //    if constexpr (std::is_same_v<T, fan::graphics::physics_shapes::rectangle_t>) {
    //      npc0_door_sensor = entity;
    //    }
    //  }, pe.visual);
    //};
    
    renderer.open(&tp);
    compiled_map0 = renderer.compile("examples/games/forest game/forest.json");
    fan::vec2i render_size(16, 9);
    render_size /= 1.5;
    fte_loader_t::properties_t p;
    p.size = render_size;
    p.position = player.player.get_position();
    map_id0 = renderer.add(&compiled_map0, p);
    fan::vec2 dst = player.player.get_position();
    loco.camera_set_position(
      loco.orthographic_camera.camera,
      dst
    );
  }

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
    player.player.process_movement(fan::graphics::character2d_t::movement_e::top_view);
    renderer.update(map_id0, dst);
    loco.set_imgui_viewport(loco.orthographic_camera.viewport);

    // physics step
    loco.physics_context.step(loco.delta_time);
  }

  loco_t loco;
  loco_t::texturepack_t tp;
  fte_renderer_t renderer;
  fte_loader_t::compiled_map_t compiled_map0;
  fte_loader_t::id_t map_id0;

  player_t player;
  fan::physics::body_id_t npc0_door_sensor;

  weather_t weather;
}*pile;

pile_t& getp() {
  return *pile;
}

player_t::player_t() {
  for (std::size_t i = 0; i < std::size(img_idle); ++i) {
    img_idle[i] = getp().loco.image_load(asset_path + "npc/static_" + fan::movement_e::_strings[i] + ".png");
  }
  static auto load_movement_images = [](std::array<loco_t::image_t, 4>& images, const std::string& direction) {
    const std::array<std::string, 4> pose_variants = {
        direction + "_left_hand_forward.png",
        "static_" + direction + ".png",
        direction + "_right_hand_forward.png",
        "static_" + direction + ".png"
    };

    for (const auto& [i, pose] : std::views::enumerate(pose_variants)) {
      images[i] = (getp().loco.image_load(asset_path + "npc/" + pose));
    }
    };

  load_movement_images(img_movement[fan::movement_e::left], "left");
  load_movement_images(img_movement[fan::movement_e::right], "right");
  load_movement_images(img_movement[fan::movement_e::up], "up");
  load_movement_images(img_movement[fan::movement_e::down], "down");

  player.set_image(img_idle[fan::movement_e::down]);

  getp().loco.input_action.edit(fan::key_w, "move_up");
  light = fan::graphics::light_t{ {
    .position = player.get_position(),
    .size = 200,
    .color = fan::colors::white,
    .flags = 3
  } };
}

void weather_t::lightning() {
  fan_ev_timer_loop(4000, { on = !on; });
  if (on) {
    getp().loco.lighting.ambient = fan::color::hsv(224.0, std::max(sin(sin_var * 2), 0.f) * 100.f, std::max(sin(sin_var), 0.f) * 100.f);
    sin_var += getp().loco.delta_time * 10;
    repeat_count++;
    if (repeat_count == 20) {
      repeat_count = 0;
      on = false;
    }
  }
  else {
    getp().loco.lighting.ambient = fan::color::hsv(0, 0, 0);
  }
}

void create_manual_collisions(std::vector<fan::physics::entity_t>& collisions, fan::algorithm::path_solver_t& path_solver) {
  for (auto& x : getp().compiled_map0.compiled_shapes) {
    for (auto& y : x) {
      for (auto& z : y) {
        if (z.image_name == "tile0" || z.image_name == "tile1" || z.image_name == "tile2") {
          collisions.push_back(getp().loco.physics_context.create_circle(
            fan::vec2(z.position) + fan::vec2(0, -z.size.y / 6),
            z.size.y / 2.f,
            fan::physics::body_type_e::static_body,
            fan::physics::shape_properties_t{ .friction = 0 }
          ));
          /* visual_collisions.push_back(fan::graphics::circle_t{ {
          .position = fan::vec3(fan::vec2(z.position)+ fan::vec2(0, -z.size.y / 6), 50000),
          .radius = z.size.y / 2.f,
          .color = fan::colors::red.set_alpha(0.5),
          .blending = true
          }});*/
          path_solver.add_collision(fan::vec3(fan::vec2(z.position) + fan::vec2(0, -z.size.y / 6), 50000));
        }
      }
    }
  }
}

void load_rain(loco_t::shape_t& rain_particles) {
  std::string data;
  fan::io::file::read("rain.json", &data);
  fan::json in = fan::json::parse(data);
  fan::graphics::shape_deserialize_t it;
  while (it.iterate(in, &rain_particles)) {
  }
  auto image_star = getp().loco.image_load("images/waterdrop.webp");
  rain_particles.set_image(image_star);;
  auto& ri = *(loco_t::particles_t::ri_t*)getp().loco.shaper.GetData(rain_particles);
  //fan::vec3 position = fan::vec3(;
  //sky_particles.set_position(position);
}

int main() {
  pile = (pile_t*)malloc(sizeof(pile_t));
  std::construct_at(pile);
  pile->loco.clear_color = 0;
  pile->player.player.force = 50;
  pile->player.player.max_speed = 1000;

  pile_t& pile_r = getp();

  fan::graphics::interactive_camera_t ic(
    pile_r.loco.orthographic_camera.camera, 
    pile_r.loco.orthographic_camera.viewport
  );

 // auto shape = pile_r.loco.grid.push_back(loco_t::grid_t::properties_t{.position= fan::vec3(fan::vec2(32*32+32-32*6), 50000),.size = 32 * 32, .grid_size = 32});

  fan::algorithm::path_solver_t path_solver(pile_r.compiled_map0.map_size*2, pile_r.compiled_map0.tile_size*2);
  pile_r.loco.input_action.add(fan::mouse_left, "move_to_position");
  fan::graphics::rectangle_t rect_dst{ {
    .position = 0,
    .size = pile_r.compiled_map0.tile_size/4,
    .color = fan::colors::red.set_alpha(0.3),
    .blending = true
  }};
  std::vector<fan::graphics::rectangle_t> rect_path;

  std::vector<fan::physics::entity_t> collisions;

  std::vector<fan::graphics::circle_t> visual_collisions;
  create_manual_collisions(collisions, path_solver);

  loco_t::shape_t rain_particles;
  load_rain(rain_particles);

  for (auto& i : pile_r.renderer.map_list[pile_r.map_id0].physics_entities) {
    if (i.id == "npc0_door") {
      std::visit([&]<typename T>(T& entity) { 
        if constexpr (std::is_same_v<T, fan::graphics::physics_shapes::rectangle_t>) {
          pile_r.npc0_door_sensor = entity;
        }
      }, i.visual);
      break;
    }
  }

  if (pile_r.npc0_door_sensor.is_valid() == false) {
    fan::throw_error("sensor not found");
  }

  pile_r.loco.loop([&] {
    ImGui::Begin("A");
    static bool v = 0;
    ImGui::ToggleButton("lightning", &v);
    ImGui::End();
    if (v)
    pile->weather.lightning();

    static int x = 0;
    if (pile_r.loco.physics_context.is_on_sensor(pile_r.player.player, pile_r.npc0_door_sensor) && x == 0) {
      if (pile_r.loco.lighting.ambient > -1) {
        pile_r.loco.lighting.ambient -= pile->loco.delta_time * 5;
      }
      else {
        if (pile_r.map_id0.iic() == false && x == 0) {
          pile_r.renderer.erase(pile_r.map_id0);
          pile_r.compiled_map0 = pile_r.renderer.compile("examples/games/forest game/shop/shop.json");
          fan::vec2i render_size(16, 9);
          render_size /= 1.5;
          fte_loader_t::properties_t p;
          p.size = render_size;
          pile_r.player.player.set_position(fan::vec2{320.384949, 382.723236 });
          pile_r.player.player.set_physics_position(pile_r.player.player.get_position());
          p.position = pile_r.player.player.get_position();
          pile_r.map_id0 = pile_r.renderer.add(&pile_r.compiled_map0, p);
          fan::vec2 dst = pile_r.player.player.get_position();
          pile_r.loco.camera_set_position(
            pile_r.loco.orthographic_camera.camera,
            dst
          );
          pile_r.loco.lighting.ambient = -1;
          x = 1;
        }
      }
    }
    if (x) {
      if (pile_r.loco.lighting.ambient < 1) {
        pile_r.loco.lighting.ambient += pile_r.loco.delta_time * 5;
      }
      else {
        pile_r.loco.lighting.ambient = 1;
      }
    }
    
    if (pile_r.loco.input_action.is_action_clicked("move_to_position") && !ImGui::IsAnyItemHovered()) {
      rect_path.clear();
      fan::vec2 dst = pile_r.loco.get_mouse_position(pile_r.loco.orthographic_camera.camera, pile_r.loco.orthographic_camera.viewport);
      path_solver.set_dst(dst);
      rect_dst.set_position(fan::vec3(dst, 50000));
      path_solver.init(pile_r.player.player.get_position());

      rect_path.reserve(path_solver.path.size());
      for (const auto& p : path_solver.path) {
        fan::vec2i pe = p;
        rect_path.push_back({ {
          .position = fan::vec3(pe * pile_r.compiled_map0.tile_size*2, 50000),
          .size = pile_r.compiled_map0.tile_size/4,
          .color = fan::colors::cyan.set_alpha(0.3),
          .blending = true
        }});
      }
    }
    if (rect_path.size() && path_solver.current_position < rect_path.size())
    rect_path[path_solver.current_position].set_color(fan::colors::green);

    pile_r.player.player.move_to_direction(path_solver.step(pile_r.player.player.get_position()));

    pile_r.step();
  });
}