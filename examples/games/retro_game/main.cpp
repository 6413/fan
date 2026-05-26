import std;
import fan;

using namespace fan::graphics;

struct pile_t;
extern pile_t pile;

struct pile_t : engine_t, fan::frame_task_t<pile_t> {

  void enter_portal() {
    static fan::auto_color_transition_t anim;
    pile.stage_transition = fan::graphics::rectangle_t(
      fan::vec3(camera_get_position() - 500, 0xfffB),
      100000,
      fan::colors::red.set_alpha(0)
    );
    anim.on_end = [this] {
      fan::print("end");
      pile.stage_transition.remove();
      stage_loader.close_stage(level_stage);
      level_stage.sic();
    };
    anim.start_once(
      fan::colors::red.set_alpha(0),
      fan::colors::red,
      1.f,
      [](fan::color c) { 
      if (c.a > 0.1) {
        fan::print(c.a);

      }
      pile.stage_transition.set_color(c);
      fan::print(pile.stage_transition.get_color(), c);
    }
    );
  }
  struct level1_t : fan::stage_t<level1_t> {
    void open(void*) {
      map = tilemap_instance_t(pile.renderer, "sample_level.fte", {
        .position = pile.player.body.get_position(),
        .size = fan::vec2i(16, 9) * 5.f,
        .collision_props{.presolve_events = true},
        .build_collisions = true,
        .default_friction = 0.f,
      });
      map.setup_view(pile.player.body, pile.ic, 1.20370352);
      fan::vec2 ts = map.get_tile_size();
      map.iterate_marks({
        {"door", [&](auto& m) {
          fan::vec2 size{64.f, 64.f};
          fan::vec2 pos = m.position.offset_y(-size.y + ts.y);
          auto shape = shape_from_json("images/gate.json");
          shape.set_position(pos);
          shape.set_size(size);
          shape.set_sprite_sheet_start();
          door.open(pile.player.body, std::move(shape), [&, door_pos = pos](physics::sprite_t& s) {
            pile.enter_portal();
            if (platforms_activated != -1) return;

          });
        }},
        {"spikes_up", [&](auto& m) { spikes.add(m.position, m.size, "up"); }},
      });

      collision_handle = pile.player.body.on_collision_enter([&](fan::physics::entity_t other) {
        auto* info = map.get_collision_info(other);
        if (info && info->id == "platform") {
          if (auto* shape = map.get_shape(other)) {
            if (shape->get_color() != fan::colors::green) {
              ++platforms_activated;
            }
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
        .target = &pile.player.body,
        .physics_properties = {.fixed_rotation = true, .linear_damping = 2.0f},
      }, enemy_pos);
      enemy.body.attack_state.damage = 1.f;
      enemy.behavior.enable_ai_patrol({enemy_pos.offset_x(-300), enemy_pos.offset_x(0)});
    }
    void close() {
      fan::physics::remove_collision_listener(collision_handle);
    }

    void update() {
      map.update(pile.player.body.get_position());

      if (spikes.query(pile.player.body)) {
        pile.stage_loader.restart_stage<level1_t>(pile.level_stage);
      }
      enemy.update(map.get_tile_size());
      if (enemy.body.test_overlap(pile.player.body)) {
        pile.player.body.take_hit(&enemy.body);
        if (pile.player.body.get_health() <= 0) {
          pile.reset_player();
          pile.stage_loader.restart_stage<level1_t>(pile.level_stage);
        }
      }
    }

    tilemap_instance_t map;
    trigger_t door;
    gameplay::spikes_t spikes;
    physics::ai_character2d_t enemy;
    fan::physics::collision_listener_handle_t collision_handle;
    bool finished = false;
    int platforms_activated = 0;

  };

  struct ui_t : fan::stage_t<ui_t> {
    void update() {
      if (auto h = gui::hud("##hud")) {
        gui::text("HP", (int)pile.player.body.get_health());
      }
    }
  };

  struct player_t {
    static inline constexpr fan::vec2 draw_offset{0.f, -42.5f};
    static inline constexpr f32_t aabb_scale = 0.17f;

    player_t() {
      body.enable_oneway_platforms();
      body.enable_default_movement(300.f, 52.f);
      body.set_draw_offset(draw_offset);
      body.set_flags(sprite_flags_e::use_hsl);
      body.set_color(fan::color::hsl(20.7f, 18.3f, -58.4f));
      body.set_health(3);
      body.attack_state.cooldown_timer.start_seconds(1.f);
    }

    void reset() { body.set_health(3.f); }

    physics::character2d_t body{
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
    stage_loader.restart_stage<ui_t>(ui_stage);
  }

  void reset_player() { player.reset(); }

  void update() {
    if (is_key_clicked(fan::key_r)) {
      stage_loader.restart_stage<level1_t>(level_stage);
    }
    player.body.update_animations();
  }

  player_t player;
  tilemap_renderer_t renderer;
  fan::stage_loader_t stage_loader;
  fan::stage_loader_t::nr_t level_stage, ui_stage;
  interactive_camera_t ic;
  fan::graphics::rectangle_t stage_transition;

} pile;

int main() {
  pile.loop();
}