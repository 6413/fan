import std;
import fan;

using namespace fan::graphics;

extern struct pile_t pile;

struct pile_t : engine_t, fan::frame_task_t<pile_t> {
  struct main_menu_t : fan::stage_t<main_menu_t> {
    void update() {
      if (auto h = gui::hud_interactive{"##mainmenu"}) {
        gui::image(image_t("images/main_menu.webp", image_presets::pixel_art()), gui::get_window_size());
        if (gui::button_centered("Play", {.size={256, 128}})) { 
          pile.stage_change<main_menu_t, ingame_t>(fan::stage_fade_mode_t::fade_in); 
        }
      }
    }
  };
  struct ingame_t : fan::stage_t<ingame_t> {
    struct player_t {
      player_t() {
        body.enable_oneway_platforms();
        body.enable_default_movement();
        body.attack_state.on_death = [&] {
          body.reset_health();
          pile.stage_restart<level_t>(&pile.stage_get<ingame_t>().level_props);
        };
      }
      physics::character2d_t body{physics::from_json({"player/player.json"})};
      light_t light{body, 200, fan::colors::white};
    };
    struct level_t : fan::stage_t<level_t> {
      struct props_t { const char *fte, *enemy_spawn, *next_fte = nullptr; };
      void open(void* raw) {
        auto& p = *static_cast<props_t*>(raw);
        auto& ig = pile.stage_get<ingame_t>();
        map = tilemap_instance_t(ig.renderer, p.fte, {
          .position = ig.player.body.get_position(),
          .size = fan::vec2i(16, 9) * 5.f,
          .collision_props{.friction=0.f, .presolve_events = true},
          .build_collisions = true,
        });
        map.setup_view(ig.player.body, ig.ic, 1.20370352);
        map.iterate_marks({
          {"door", [&, next = p.next_fte](auto& m) {
            auto shape = shape_from_json("images/gate.json");
            shape.set_position(m.position.offset_y(-64.f + map.get_tile_size().y)).set_size({64.f, 64.f});
            door.open(ig.player.body, std::move(shape), [&, next](physics::sprite_t&) {
              if (platforms_activated != -1 || !next) { return; }
              pile.stage_get<ingame_t>().level_props = {next, "enemy_skeleton"};
              pile.stage_restart<level_t>(&pile.stage_get<ingame_t>().level_props);
            });
          }},
          {"spike_up", [&](auto& m) { spikes.add(m.position, m.size, "up"); }},
        });
        collision_scope.on_enter(ig.player.body, [&](fan::physics::entity_t other) {
          if (auto* info = map.get_collision_info(other); info && info->id == "platform") {
            if (auto* shape = map.get_shape(other)) {
              platforms_activated += (shape->get_color() != fan::colors::green);
              shape->set_color(fan::colors::green);
              if (platforms_activated == map.count("platform")) {
                door.shape.play_sprite_sheet_once("open");
                platforms_activated = -1;
              }
            }
          }
        });
        auto poss = map.get_enemy_spawns(p.enemy_spawn);
        enemies.resize(poss.size());
        for (auto [i, pos] : fan::enumerate(poss)) {
          auto& e = enemies[i];
          e.open({.json_path = "enemies/skeleton/skeleton.json", .target = &ig.player.body}, pos);
          e.body.enable_oneway_platforms();
          e.navigation.add_obstacle([&](fan::vec2 wp) { return spikes.is_at(wp); });
          e.behavior.enable_ai_patrol({pos.offset_x(-300), pos.offset_x(300)});
        }
      }
      void update() {
        auto& ig = pile.stage_get<ingame_t>();
        map.update(ig.player.body.get_position());
        if (spikes.query_and_kill(ig.player.body)) { return; }
        for (auto& e : enemies) {
          e.update(map.get_tile_size());
          if (e.body.test_overlap(ig.player.body)) { ig.player.body.take_hit(&e.body); }
        }
      }
      tilemap_instance_t map;
      trigger_t door;
      gameplay::spikes_t spikes;
      std::vector<physics::ai_character2d_t> enemies;
      physics::collision_scope_t collision_scope;
      std::int32_t platforms_activated = 0;
    };
    struct hud_t : fan::stage_t<hud_t> {
      void update() {
        if (auto h = gui::hud("##hud")) { 
          gui::text("HP", static_cast<std::int32_t>(pile.stage_get<ingame_t>().player.body.get_health())); 
        }
      }
    };
    void open(void*) {
      level_props = {"level2.fte", "enemy_skeleton", "level2.fte"};
      pile.stage_open<level_t>(&level_props);
      pile.stage_open<hud_t>();
    }
    void close() {
      pile.stage_close<level_t>();
      pile.stage_close<hud_t>();
    }
    void update() {
      player.body.update_animations();
      if (pile.is_key_clicked(fan::key_r)) { pile.stage_restart<level_t>(&level_props); }
    }
    player_t player;
    tilemap_renderer_t renderer;
    interactive_camera_t ic;
    level_t::props_t level_props;
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