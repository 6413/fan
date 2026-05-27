import std;
import fan;

using namespace fan::graphics;

extern struct pile_t pile;

struct pile_t : engine_t, fan::frame_task_t<pile_t> {
  struct main_menu_t : fan::stage_t<main_menu_t> {
    void update() {
      if (auto h = gui::hud_interactive{"##mainmenu"}) {
        if (gui::button("Play")) {
          pile.stage_change<main_menu_t, ingame_t>(fan::stage_fade_mode_t::fade_in);
        }
      }
    }
  };

  struct ingame_t : fan::stage_t<ingame_t> {
    struct player_t {
      player_t() {
        body.enable_oneway_platforms();
        body.enable_default_movement(300.f, 52.f);
        body.set_draw_offset({0.f, -42.5f});
        body.set_flags(sprite_flags_e::use_hsl);
        body.set_color(fan::color::hsl(20.7f, 18.3f, -58.4f));
        body.set_max_health(3.f);
        body.reset_health();
        body.attack_state.cooldown_timer.start_seconds(1.f);
        body.attack_state.on_death = [&] {
          body.reset_health();
          pile.stage_restart<level1_t>();
        };
      }
      physics::character2d_t body{physics::character2d_t::from_json({
        .json_path = "examples/games/platformer/player/player.json",
        .aabb_scale = 0.17f,
      })};
      light_t light{body, 200, fan::colors::white};
    };

    struct level1_t : fan::stage_t<level1_t> {
      void open(void*) {
        auto& ig = pile.stage_get<ingame_t>();
        map = tilemap_instance_t(ig.renderer, "sample_level.fte", {
          .position = ig.player.body.get_position(),
          .size = fan::vec2i(16, 9) * 5.f,
          .collision_props{.presolve_events = true},
          .build_collisions = true,
          .default_friction = 0.f,
        });
        map.setup_view(ig.player.body, ig.ic, 1.20370352);
        fan::vec2 ts = map.get_tile_size();
        map.iterate_marks({
          {"door", [&](auto& m) {
            fan::vec2 size{64.f, 64.f};
            auto shape = shape_from_json("images/gate.json");
            shape.set_position(m.position.offset_y(-size.y + ts.y));
            shape.set_size(size);
            shape.set_sprite_sheet_start();
            door.open(ig.player.body, std::move(shape), [&](physics::sprite_t&) {
              if (platforms_activated != -1) return;
              pile.stage_change<level1_t, level1_t>();
            });
          }},
          {"spikes_up", [&](auto& m) { spikes.add(m.position, m.size, "up"); }},
        });
        collision_scope.on_enter(ig.player.body, [&](fan::physics::entity_t other) {
          auto* info = map.get_collision_info(other);
          if (info && info->id == "platform") {
            if (auto* shape = map.get_shape(other)) {
              if (shape->get_color() != fan::colors::green) { ++platforms_activated; }
              shape->set_color(fan::colors::green);
              if (platforms_activated == map.count("platform")) {
                door.shape.play_sprite_sheet_once("open");
                platforms_activated = -1;
              }
            }
          }
        });
        fan::vec2 enemy_pos = map.get_enemy_spawn("enemy_skeleton");
        enemy.open({
          .json_path = "examples/games/platformer/enemy/skeleton/skeleton.json",
          .aabb_scale = 0.14f * 1.5f,
          .draw_offset = {0.f, -06.f / 1.5f},
          .target = &ig.player.body,
          .physics_properties = {.fixed_rotation = true, .linear_damping = 2.0f},
        }, enemy_pos);
        enemy.body.attack_state.damage = 1.f;
        enemy.behavior.enable_ai_patrol({enemy_pos.offset_x(-300), enemy_pos.offset_x(0)});
      }
      void update() {
        auto& ig = pile.stage_get<ingame_t>();
        map.update(ig.player.body.get_position());
        if (spikes.query_and_kill(ig.player.body)) return;
        enemy.update(map.get_tile_size());
        if (enemy.body.test_overlap(ig.player.body)) { ig.player.body.take_hit(&enemy.body); }
      }

      tilemap_instance_t map;
      trigger_t door;
      gameplay::spikes_t spikes;
      physics::ai_character2d_t enemy;
      physics::collision_scope_t collision_scope;
      int platforms_activated = 0;
    };

    struct hud_t : fan::stage_t<hud_t> {
      void update() {
        if (auto h = gui::hud("##hud")) {
          gui::text("HP", (int)pile.stage_get<ingame_t>().player.body.get_health());
        }
      }
    };

    void open(void*) {
      pile.stage_open<level1_t>();
      pile.stage_open<hud_t>();
    }
    void close() {
      pile.stage_close<level1_t>();
      pile.stage_close<hud_t>();
    }
    void update() {
      player.body.update_animations();
      if (pile.is_key_clicked(fan::key_r)) { pile.stage_restart<level1_t>(); }
    }

    player_t player;
    tilemap_renderer_t renderer;
    interactive_camera_t ic;
  };

  pile_t() {
    texture_pack.open_compiled("sample_texture_pack.ftp");
    update_physics(true);
    stage_open<main_menu_t>();
  }

} pile;

int main() {
  pile.loop();
}