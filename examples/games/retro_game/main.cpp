import std;
import fan;

using namespace fan::graphics;

struct pile_t : engine_t, fan::frame_task_t<pile_t> {

  struct level1_t : fan::stage_t<level1_t> {
    void open(void* sod) {
      id = pile.renderer.open_map("sample_level.fte", {
        .position = pile.player.body.get_position(),
        .size = fan::vec2i(16, 9) * 5.f,
      });

      pile.renderer.iterate_tiles(id, [&](const auto& tile) {
        collisions.emplace_back(physics->create_box(tile.position, tile.size, 0, fan::physics::body_type_e::static_body, {}));
      });

      fan::vec2 ts = pile.renderer.get_tile_size(id);
      pile.renderer.setup_view(id, pile.player.body, pile.ic, 2.08f);

      pile.renderer.iterate_marks(id, {
        {"key", [&](auto& m) {
          key = {{
            .position = m.position, .size = 32.f,
            .image = {"images/key.webp", image_presets::pixel_art()},
            .shape_properties{.is_sensor = true}
          }};
          key.on_sensor_enter(pile.player.body, [&] { key.erase(); });
        }},
        {"door", [&](auto& m) {
          fan::vec2 size{32.f / 6.f, 64.f};
          door = {{
            .position = m.position.offset_y(-size.y + ts.y), .size = size,
            .image = {"images/door_closed.png", image_presets::pixel_art()},
            .shape_properties{.is_sensor = true}
          }};
          door_pos = door.get_position();

          door.on_sensor_enter(pile.player.body, [&] {
            if (!door_open) {
              door.set_position(door_pos.offset_x(-32.f));
              door.set_size({32.f, 64.f});
              door.set_image({"images/door.png", image_presets::pixel_art()});
              door_open = true;
            }
          });
        }},
      });
    }
    void close() {
      for (auto& c : collisions) { c.destroy(); }
      pile.renderer.close_map(id); 
    }

    void update() {
      pile.renderer.update(id, pile.player.body.get_position());
    }

    std::vector<fan::physics::entity_t> collisions;
    tilemap_renderer_t::id_t id;
    physics::sprite_t key, door;
    fan::vec2 door_pos = 0;
    bool door_open = false;
  };

  struct player_t : fan::frame_task_t<player_t> {
    player_t() {
      body.enable_default_movement();
      body.set_jump_height(32.f);
      body.set_movement_speed(300.f);
      body.add_child(light);
    }

    void update() {
      if (auto h = gui::hud_interactive{"test"}) {
      }
    }

    physics::character2d_t body = physics::character_capsule({
      .center0 = {0.f, -24.f},
      .center1 = {0.f, 24.f},
      .radius = 12,
    }, { .fixed_rotation = true });
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