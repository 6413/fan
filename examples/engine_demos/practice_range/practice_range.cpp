#include <string>

import fan;

using namespace fan::graphics;

struct app_t : engine_t {
  static inline constexpr fan::vec2 draw_offset {0.f, -42.5f};
  static inline constexpr f32_t aabb_scale = 0.17f;

  app_t() {
    texture_pack.open_compiled("textures.ftp");
    id = renderer.open_map("map.json", {.offset= -64*32});
    //player.set_position(renderer.get_spawn_position(id));
    //player.enable_default_movement();
    player.set_draw_offset(draw_offset);
    physics::character_movement_preset_t::setup_default_controls(player);

  }

  void loop() {
    update_physics(true);
    engine_t::loop([&]{
      camera_look_at(player.get_position(), 0);
      renderer.update(id, {64, 64});
      player.update_animations();
    });
  }
  interactive_camera_t ic;
  tilemap_renderer_t renderer;
  tilemap_renderer_t::id_t id;
  physics::character2d_t player{
    physics::character2d_t::from_json({
      .json_path = "examples/games/platformer/player/player.json",
      .aabb_scale = aabb_scale,
      .attack_cb = [this](fan::graphics::physics::character2d_t& c) -> bool {
        bool attack_pressed = is_clicked(fan::actions::light_attack);
        if (!attack_pressed || gui::want_io()) {
          return false;
        }
        return c.attack_state.try_attack(&c);
      },
    })
  };
};

int main() {
  app_t app;
  app.loop();
}