#include <fan/pch.h>
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
  fan::vec2 prev_dir = 0;

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
};

int main() {
  loco_t loco;
  loco_t::image_load_properties_t lp;
  lp.visual_output = loco_t::image_sampler_address_mode::clamp_to_border;
  lp.min_filter = GL_NEAREST;
  lp.mag_filter = GL_NEAREST;

  loco_t::texturepack_t tp;
  tp.open_compiled("examples/games/forest game/forest_tileset.ftp", lp);

  fte_renderer_t renderer;
  renderer.open(&tp);
  auto compiled_map = renderer.compile("examples/games/forest game/forest.json");
  fan::vec2i render_size(16, 9);
  render_size /= 1.5;
  player_t player;
  fte_loader_t::properties_t p;
  p.size = render_size;
  p.position = player.player.get_position();
  auto map_id0 = renderer.add(&compiled_map, p);


  fan::graphics::interactive_camera_t ic(
    gloco->orthographic_camera.camera, 
    gloco->orthographic_camera.viewport
  );

  int x = 0;
  //auto shape = loco.grid.push_back(loco_t::grid_t::properties_t{.position= fan::vec3(fan::vec2(32*32+32-32*6), 50000),.size = 32 * 32, .grid_size = 32});

  loco.loop([&] {
    if (x == 2) {
      loco.console.commands.call("set_target_fps 0");
      loco.console.commands.call("set_vsync 1");
    }
    ++x;
    fan::vec2 s = ImGui::GetContentRegionAvail();
    fan::vec2 dst = player.player.get_position();
    fan::vec2 src = loco.camera_get_position(gloco->orthographic_camera.camera);

    fan_ev_timer_loop_init(150, 
      loco.input_action.is_action_clicked("move_up") || 
      loco.input_action.is_action_clicked("move_down"),
      {
      if (loco.input_action.is_action_down("move_down")) {
        static int i = 0;
        player.player.set_image(player.img_movement_down[i % std::size(player.img_movement_left)]);
        player.prev_dir.y = 1;
        player.prev_dir.x = 0;
        ++i;
      }
      else if (loco.input_action.is_action_down("move_up")) {
        static int i = 0;
        player.player.set_image(player.img_movement_up[i % std::size(player.img_movement_left)]);
        player.prev_dir.y = -1;
        player.prev_dir.x = 0;
        ++i;
      }
      else {
        if (player.prev_dir.y < 0) {
          player.player.set_image(player.img_idle[0]);
        }
        else if (player.prev_dir.y > 0) {
          player.player.set_image(player.img_idle[1]);
        }
        player.prev_dir.y = 0;
      }
    });

    if (player.prev_dir.y == 0) {
      fan_ev_timer_loop_init(150,
        loco.input_action.is_action_clicked("move_left") || 
        loco.input_action.is_action_clicked("move_right"),
        {
        if (loco.input_action.is_action_down("move_right")) {
          static int i = 0;
          player.player.set_image(player.img_movement_right[i % std::size(player.img_movement_left)]);
          player.prev_dir.y = 0;
          player.prev_dir.x = 1;
          ++i;
        }
        else if (loco.input_action.is_action_down("move_left")) {
          static int i = 0;
          player.player.set_image(player.img_movement_left[i % std::size(player.img_movement_left)]);
          player.prev_dir.y = 0;
          player.prev_dir.x = -1;
          ++i;
        }
        else {
          if (player.prev_dir.x < 0) {
            player.player.set_image(player.img_idle[2]);
          }
          else if (player.prev_dir.x > 0) {
            player.player.set_image(player.img_idle[3]);
          }
          player.prev_dir.x = 0;
        }
      });
    }

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
    loco.physics_context.step(loco.delta_time);
  });
}