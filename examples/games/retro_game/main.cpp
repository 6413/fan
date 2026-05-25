import std;
import fan;

using namespace fan::graphics;

struct pile_t;
extern pile_t pile;

struct pile_t : engine_t, fan::frame_task_t<pile_t> {

  struct level1_t : fan::stage_t<level1_t> {
    void open(void*) {
      map = tilemap_instance_t(pile.renderer, "sample_level.fte", {
        .position = pile.player.body.get_position(),
        .size = fan::vec2i(16, 9) * 5.f,
        .collision_props{.presolve_events = true},
        .build_collisions = true
      });
      pile.renderer.iterate_physics_entities(map.id, [&](auto& a, auto& t) {
        t.set_friction(0.f);
        return false;
      });

      fan::vec2 ts = map.get_tile_size();
      map.setup_view(pile.player.body, pile.ic, 1.20370352);

      map.iterate_marks({
        {"key", [&](auto& m) {
          key.open(pile.player.body, {
            .position = m.position, .size = 32.f,
            .image = {"images/key.webp", image_presets::pixel_art()}
          }, [](physics::sprite_t& s) { s.erase(); });
        }},
        {"door", [&](auto& m) {
          fan::vec2 size{32.f / 6.f, 64.f};
          fan::vec2 pos = m.position.offset_y(-size.y + ts.y);
          door.open(pile.player.body, {
            .position = pos, .size = size,
            .image = {"images/door_closed.png", image_presets::pixel_art()}
          }, [&, door_pos = pos](physics::sprite_t& s) {
            if (key.shape.is_valid()) return;
            s.set_position(door_pos.offset_x(-32.f));
            s.set_size({32.f, 64.f});
            s.set_image({"images/door.png", image_presets::pixel_art()});
          });
        }},
        {"spikes_up", [&](auto& m) {
          spikes.add(m.position, m.size, "up");
        }},
      });

      collision_scope.on_enter(pile.player.body, [&](fan::physics::entity_t other) {
        auto* info = map.get_collision_info(other);
        if (info && info->id == "platform") {
          if (auto* shape = map.get_shape(other)) { shape->set_color(fan::colors::green); }
        }
      });
    }

    void update() { 
      map.update(pile.player.body.get_position());
      if (spikes.query(pile.player.body)) {
        pile.stage_loader.restart_stage<level1_t>(pile.level_stage);
      }
    }

    tilemap_instance_t map;
    trigger_t key, door;
    gameplay::spikes_t spikes;
    physics::collision_scope_t collision_scope;
  };

  struct player_t {
    static inline constexpr fan::vec2 draw_offset {0.f, -42.5f};
    static inline constexpr f32_t aabb_scale = 0.17f;

    player_t() { 
      body.enable_oneway_platforms();
      body.enable_default_movement(300.f, 52.f);
      body.set_draw_offset(draw_offset);
      body.set_flags(sprite_flags_e::use_hsl);
      body.set_color(fan::color::hsl(20.7f, 18.3f, -58.4f));
    }

    physics::character2d_t body {
      physics::character2d_t::from_json({
        .json_path = "examples/games/platformer/player/player.json",
        .aabb_scale = aabb_scale,
      })
    };
    light_t light{body, 200, fan::colors::white};
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
    player.body.update_animations();
  }

  player_t player;
  tilemap_renderer_t renderer;
  fan::stage_loader_t stage_loader;
  fan::stage_loader_t::nr_t level_stage;
  interactive_camera_t ic;
} pile;

int main() {
  pile.loop();
}