import std;
import fan;

using namespace fan::graphics;

struct pile_t : engine_t, fan::frame_task_t<pile_t> {

  struct level1_t : fan::stage_t<level1_t> {
    void open(void* sod) {
      map = tilemap_instance_t(pile.renderer, "sample_level.fte", {
        .position = pile.player.body.get_position(),
        .size = fan::vec2i(16, 9) * 5.f,
      });

      map.build_collisions();

      fan::vec2 ts = pile.renderer.get_tile_size(map.id);
      pile.renderer.setup_view(map.id, pile.player.body, pile.ic, 2.08f);

      pile.renderer.iterate_marks(map.id, {
        {"key", [&](auto& m) {
          key.open(pile.player.body, {{
            .position = m.position, .size = 32.f,
            .image = {"images/key.webp", image_presets::pixel_art()},
            .shape_properties{.is_sensor = true}
          }}, [](physics::sprite_t& s) { s.erase(); });
        }},
        {"door", [&](auto& m) {
          fan::vec2 size{32.f / 6.f, 64.f};
          door.open(pile.player.body, {{
            .position = m.position.offset_y(-size.y + ts.y), .size = size,
            .image = {"images/door_closed.png", image_presets::pixel_art()},
            .shape_properties{.is_sensor = true}
          }}, [](physics::sprite_t& s) {
            s.set_position(s.get_position().offset_x(-32.f));
            s.set_size({32.f, 64.f});
            s.set_image({"images/door.png", image_presets::pixel_art()});
          });
        }},
      });
    }

    void update() { map.update(pile.player.body.get_position()); }

    tilemap_instance_t map;
    trigger_t key, door;
    fan::vec2 door_pos = 0;
    bool door_open = false;
  };

  struct player_t {
    player_t() {
      body.enable_default_movement();
      body.set_jump_height(32.f);
      body.set_movement_speed(300.f);
      body.add_child(light);
    }

    physics::character2d_t body = physics::character_capsule({
      .center0 = {0.f, -24.f},
      .center1 = {0.f, 24.f},
      .radius = 12,
    });
    light_t light{body.get_position(), 200, fan::colors::white};
  };

  pile_t() {
    texture_pack.open_compiled("sample_texture_pack.ftp");
    update_physics(true);
    stage_loader.restart_stage<level1_t>(level_stage);
  }

  void update() {
    if (is_key_clicked(fan::key_r)) {
      stage_loader.restart_stage<level1_t>(level_stage);
    }
  }

  player_t player;
  tilemap_renderer_t renderer;
  fan::stage_loader_t stage_loader;
  fan::stage_loader_t::nr_t level_stage;
  interactive_camera_t ic{orthographic_render_view};
} pile;

int main() {
  pile.loop();
}