#include <fan/pch.h>
#include <fan/graphics/algorithm/astar.h>
#include <fan/graphics/gui/tilemap_editor/renderer0.h>

fan_track_allocations();

std::string asset_path = "examples/games/forest game/";

struct player_t {
  fan::vec2 velocity = 0;
  std::array<loco_t::image_t, 4> img_idle;
  std::array<loco_t::image_t, 4> img_movement_up;
  std::array<loco_t::image_t, 4> img_movement_down;
  std::array<loco_t::image_t, 4> img_movement_left;
  std::array<loco_t::image_t, 4> img_movement_right;

  player_t() :
    img_idle({
      gloco->image_load(asset_path + "npc/static_back.png"),
      gloco->image_load(asset_path + "npc/static_down.png"),
      gloco->image_load(asset_path + "npc/static_left.png"),
      gloco->image_load(asset_path + "npc/static_right.png")
    }),
    img_movement_up({
      gloco->image_load(asset_path + "npc/back_left_hand_forward.png"),
      gloco->image_load(asset_path + "npc/static_back.png"),
      gloco->image_load(asset_path + "npc/back_right_hand_forward.png"),
      gloco->image_load(asset_path + "npc/static_back.png"),
    }),
    img_movement_down({
      gloco->image_load(asset_path + "npc/forward_left_hand_forward.png"),
      gloco->image_load(asset_path + "npc/static_down.png"),
      gloco->image_load(asset_path + "npc/forward_right_hand_forward.png"),
      gloco->image_load(asset_path + "npc/static_down.png"),
      }),
    img_movement_left({
      gloco->image_load(asset_path + "npc/left_left_hand_forward.png"),
      gloco->image_load(asset_path + "npc/static_left.png"),
      gloco->image_load(asset_path + "npc/left_right_hand_forward.png"),
      gloco->image_load(asset_path + "npc/static_left.png"),
    }),
    img_movement_right({
      gloco->image_load(asset_path + "npc/right_left_hand_forward.png"),
      gloco->image_load(asset_path + "npc/static_right.png"),
      gloco->image_load(asset_path + "npc/right_right_hand_forward.png"),
      gloco->image_load(asset_path + "npc/static_right.png")
    })
     {
    gloco->input_action.edit(fan::key_w, "move_up");
  }

  void animate_player_walk() {
    animator.process_walk(
      player,
      player.get_linear_velocity(),
      img_idle, img_movement_up, img_movement_down,
      img_movement_left, img_movement_right
    );
  }

  fan::graphics::character2d_t player{ fan::graphics::physics_shapes::circle_sprite_t{{
    .position = fan::vec3(1019.59076, 934.117065, 10),
    // collision radius
    .radius = 8,
    // image size
    .size = fan::vec2(16, 32),
    .image = img_idle[1],
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
  fan::graphics::animator_t animator;
};

struct pile_t {
  pile_t() {
    loco_t::image_load_properties_t lp;
    lp.visual_output = loco_t::image_sampler_address_mode::clamp_to_border;
    lp.min_filter = GL_NEAREST;
    lp.mag_filter = GL_NEAREST;

    tp.open_compiled("examples/games/forest game/forest_tileset.ftp", lp);

    renderer.open(&tp);
    compiled_map0 = renderer.compile("examples/games/forest game/forest.json");
    fan::vec2i render_size(16, 9);
    render_size /= 1.5;
    fte_loader_t::properties_t p;
    p.size = render_size;
    p.position = player.player.get_position();
    map_id0 = renderer.add(&compiled_map0, p);
  }

  void step() {

    //player updates
    player.animate_player_walk();

    // map renderer & camera update
    fan::vec2 s = ImGui::GetContentRegionAvail();
    fan::vec2 dst = player.player.get_position();
    fan::vec2 src = loco.camera_get_position(gloco->orthographic_camera.camera);
    loco.camera_set_position(
      gloco->orthographic_camera.camera,
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
    loco.set_imgui_viewport(gloco->orthographic_camera.viewport);

    // physics step
    loco.physics_context.step(loco.delta_time);
  }

  loco_t loco;
  loco_t::texturepack_t tp;
  fte_renderer_t renderer;
  fte_loader_t::compiled_map_t compiled_map0;
  fte_loader_t::id_t map_id0;

  player_t player;
};

int main() {
  pile_t pile;

  fan::graphics::interactive_camera_t ic(
    gloco->orthographic_camera.camera, 
    gloco->orthographic_camera.viewport
  );

  auto shape = pile.loco.grid.push_back(loco_t::grid_t::properties_t{.position= fan::vec3(fan::vec2(32*32+32-32*6), 50000),.size = 32 * 32, .grid_size = 32});

  fan::algorithm::path_solver_t path_solver(pile.compiled_map0.map_size*2, pile.compiled_map0.tile_size*2);
  pile.loco.input_action.add(fan::mouse_left, "move_to_position");
  fan::graphics::rectangle_t rect_dst{ {
    .position = 0,
    .size = pile.compiled_map0.tile_size/4,
    .color = fan::colors::red.set_alpha(0.3),
    .blending = true
  }};
  std::vector<fan::graphics::rectangle_t> rect_path;

  std::vector<fan::physics::entity_t> collisions;

  std::vector<fan::graphics::circle_t> visual_collisions;
  for (auto& x : pile.compiled_map0.compiled_shapes) {
    for (auto& y : x) {
      for (auto& z : y) {
        if (z.image_name == "tile0" || z.image_name == "tile1" || z.image_name == "tile2") {
          collisions.push_back(pile.loco.physics_context.create_circle(
            fan::vec2(z.position) + fan::vec2(0, -z.size.y / 6),
            z.size.y / 2.f,
            fan::physics::body_type_e::static_body,
            fan::physics::shape_properties_t{.friction=0}
          ));
          visual_collisions.push_back(fan::graphics::circle_t{ {
            .position = fan::vec3(fan::vec2(z.position)+ fan::vec2(0, -z.size.y / 6), 50000),
            .radius = z.size.y / 2.f,
            .color = fan::colors::red.set_alpha(0.5),
            .blending = true
          }});
          path_solver.add_collision(visual_collisions.back().get_position());
        }
      }
    }
  }

  pile.loco.loop([&] {
    if (pile.loco.input_action.is_action_clicked("move_to_position")) {
      rect_path.clear();
      fan::vec2 dst = pile.loco.get_mouse_position(gloco->orthographic_camera.camera, gloco->orthographic_camera.viewport);
      path_solver.set_dst(dst);
      rect_dst.set_position(fan::vec3(dst, 50000));
      path_solver.init(pile.player.player.get_position());

      rect_path.reserve(path_solver.path.size());
      for (const auto& p : path_solver.path) {
        fan::vec2i pe = p;
        rect_path.push_back({ {
          .position = fan::vec3(pe * pile.compiled_map0.tile_size*2, 50000),
          .size = pile.compiled_map0.tile_size/4,
          .color = fan::colors::cyan.set_alpha(0.3),
          .blending = true
        }});
      }
    }
    if (rect_path.size() && path_solver.current_position < rect_path.size())
    rect_path[path_solver.current_position].set_color(fan::colors::green);

    pile.player.player.move_to_direction(path_solver.step(pile.player.player.get_position()));

    pile.step();
  });
}