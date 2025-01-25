#include <fan/pch.h>
#include <fan/graphics/algorithm/AStar.hpp>
#include <fan/graphics/gui/tilemap_editor/renderer0.h>

fan_track_allocations();

std::string asset_path = "examples/games/forest game/";

struct animator_t {
  fan::vec2 prev_dir = 0;

  void process_walk(loco_t::shape_t& shape,
    const fan::vec2& vel,
    const std::array<loco_t::image_t, 4>& img_idle,
    const std::array<loco_t::image_t, 4>& img_movement_up,
    const std::array<loco_t::image_t, 4>& img_movement_down,
    const std::array<loco_t::image_t, 4>& img_movement_left,
    const std::array<loco_t::image_t, 4>& img_movement_right
  ) {
    f32_t animation_velocity_threshold = 10.f;
    fan_ev_timer_loop_init(150,
      0/*vel.y*/,
      {
      if (vel.y > animation_velocity_threshold) {
        static int i = 0;
        shape.set_image(img_movement_down[i % std::size(img_movement_left)]);
        prev_dir.y = 1;
        prev_dir.x = 0;
        ++i;
      }
      else if (vel.y < -animation_velocity_threshold) {
        static int i = 0;
        shape.set_image(img_movement_up[i % std::size(img_movement_left)]);
        prev_dir.y = -1;
        prev_dir.x = 0;
        ++i;
      }
      else {
        if (prev_dir.y < 0) {
          shape.set_image(img_idle[0]);
        }
        else if (prev_dir.y > 0) {
          shape.set_image(img_idle[1]);
        }
        prev_dir.y = 0;
      }
      });

    if (prev_dir.y == 0) {
      fan_ev_timer_loop_init(150,
        0/*vel.x*/,
        {
        if (vel.x > animation_velocity_threshold) {
          static int i = 0;
          shape.set_image(img_movement_right[i % std::size(img_movement_left)]);
          prev_dir.y = 0;
          prev_dir.x = 1;
          ++i;
        }
        else if (vel.x < -animation_velocity_threshold) {
          static int i = 0;
          shape.set_image(img_movement_left[i % std::size(img_movement_left)]);
          prev_dir.y = 0;
          prev_dir.x = -1;
          ++i;
        }
        else {
          if (prev_dir.x < 0) {
            shape.set_image(img_idle[2]);
          }
          else if (prev_dir.x > 0) {
            shape.set_image(img_idle[3]);
          }
          prev_dir.x = 0;
        }
        });
    }
  }
};

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
    walk_animation.process_walk(
      player,
      player.get_linear_velocity(),
      img_idle, img_movement_up, img_movement_down,
      img_movement_left, img_movement_right
    );
  }

  fan::graphics::character2d_t player{ fan::graphics::physics_shapes::sprite_t{{
    .position = fan::vec3(1019.59076, 934.117065, 10),
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
  animator_t walk_animation;
};

struct path_solver_t {
  using path_t = AStar::CoordinateList;
  path_solver_t(const fan::vec2i& map_size_, const fan::vec2& tile_size_) {
    this->map_size = map_size_;
    generator.setWorldSize(map_size);
    generator.setHeuristic(AStar::Heuristic::euclidean);
    generator.setDiagonalMovement(false);
    tile_size = tile_size_;
  }
  path_t get_path(const fan::vec2& src_ = -1) {
    if (src_ != -1) {
      src = get_grid(src_);
    }
    path_t v = generator.findPath(src, dst);
    v.pop_back();
    std::reverse(v.begin(), v.end());
    return v;
  }
  fan::vec2i get_grid(const fan::vec2& p) const {
    return (p / tile_size).floor();
  }
  // takes raw position and converts it to grid automatically
  void set_src(const fan::vec2& src_) {
    src = get_grid(src_);
  }
  void set_dst(const fan::vec2& dst_) {
    dst = get_grid(dst_);
  }
  void add_collision(const fan::vec2& p) {
    generator.addCollision(get_grid(p));
  }
  AStar::Generator generator;
  fan::vec2i src = 0;
  fan::vec2i dst = 0;
  fan::vec2i map_size;
  fan::vec2 tile_size;
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
    static f32_t z = 17;
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

  //auto shape = loco.grid.push_back(loco_t::grid_t::properties_t{.position= fan::vec3(fan::vec2(32*32+32-32*6), 50000),.size = 32 * 32, .grid_size = 32});

  path_solver_t path_solver(pile.compiled_map0.map_size, pile.compiled_map0.tile_size);
  bool move = false;
  bool init = true;

  path_solver_t::path_t path;
  int current_position = 0;

  pile.loco.input_action.add(fan::mouse_left, "move_to_position");
  fan::graphics::rectangle_t rect_dst{ {
    .position = 0,
    .size = pile.compiled_map0.tile_size/4,
    .color = fan::colors::red.set_alpha(0.3),
    .blending = true
  }};
  std::vector<fan::graphics::rectangle_t> rect_path;
  pile.loco.loop([&] {
    if (pile.loco.input_action.is_action_clicked("move_to_position")) {
      fan::vec2 dst = pile.loco.get_mouse_position(gloco->orthographic_camera.camera, gloco->orthographic_camera.viewport);
      path_solver.set_dst(dst);
      rect_dst.set_position(fan::vec3(dst, 50000));
      
      path = path_solver.get_path(pile.player.player.get_position());
      rect_path.reserve(path.size());
      for (const auto& p : path) {
        rect_path.push_back({ {
          .position = fan::vec3(fan::vec2i(p) * pile.compiled_map0.tile_size, 50000),
          .size = pile.compiled_map0.tile_size/4,
          .color = fan::colors::cyan.set_alpha(0.3),
          .blending = true
        }});
      }
      move = true;
      init = true;
    }
    
    if (move) {
      fan::vec2i src_pos = path_solver.get_grid(pile.player.player.get_position());
      if (init) {
        current_position = 0;
        init = false;
      }
      if (src_pos == fan::vec2i(path[current_position])) {
        ++current_position;
      }
      if (src_pos == path_solver.dst) {
        move = false;
      }
      else {
        fan::vec2i current = current_position >= path.size() ? path_solver.dst : fan::vec2i(path[current_position]);
        fan::vec2i direction = current - src_pos;
        pile.player.player.move_to_direction(direction);
      }
    }

    pile.step();
  });
}