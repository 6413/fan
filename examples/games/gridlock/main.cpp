#include <fan/utility.h>
#include <cstdint>
#include <vector>
#include <string>
#include <cmath>
#include <span>
#include <array>

import fan;

using namespace fan::graphics;
using namespace fan::ecs;
using fan::vec2;
using fan::vec2i;
using fan::color;
namespace colors = fan::colors;

static constexpr f32_t cfg_windows_bg_alpha = 0.7f;
static constexpr f32_t cfg_grid_size = 64.f;
static constexpr f32_t cfg_build_time = 10.f;
static constexpr f32_t cfg_player_speed = 400.f;
static constexpr f32_t cfg_player_fire_cd = 0.1f;
static constexpr f32_t cfg_bullet_speed = 1200.f;
static constexpr f32_t cfg_bullet_life = 2.f;
static constexpr f32_t cfg_turret_range = 2000.f;
static constexpr f32_t cfg_turret_fire_cd = 0.2f;
static constexpr f32_t cfg_turret_bspeed = 1000.f;
static constexpr f32_t cfg_enemy_sep_r = 24.f;
static constexpr f32_t cfg_enemy_base_speed = 8.f;
static constexpr f32_t cfg_path_speed_bonus = 0.04f;
static constexpr int cfg_path_len_thresh = 10;
static constexpr int cfg_path_bash_thresh = 20;
static constexpr int cfg_start_money = 50;
static constexpr int cfg_core_health = 100;
static constexpr int cfg_obstacle_count = 30;
static constexpr int cfg_wall_hp = 60;
static constexpr int cfg_wall_dmg = 1;
static constexpr int cfg_wall_dmg_bash = 3;

static f32_t hud_min_height = 0.f;
static f32_t hud_max_height = 0.f;

struct spawn_config_t {
  f32_t base_interval = 0.8f;
  f32_t interval_variance = 0.2f;
  int base_hp = 10;
  int hp_per_wave = 5;
  f32_t base_fly_chance = 0.f;
  f32_t fly_chance_per_wave = 10.f;
  f32_t max_fly_chance = 60.f;
};
static constexpr spawn_config_t cfg_spawn;

struct build_def_t {
  fan::str_view_t name;
  int cost;
  std::string label;
};

inline static const auto build_defs = std::to_array<build_def_t>({
  {"Wall", 10, "Wall\n[$10]"},
  {"Turret", 100, "Turret\n[$100]"}
});

struct tag_wall     { color outer_col = color(0, 0.3f, 1.f, 1.f); };
struct tag_obstacle { vec2i cell; };
struct tag_enemy    { bool flying; std::vector<vec2i> path; int initial_path_size = 0; };
struct tag_turret   { fan::cooldown_t cd; };
struct tag_bullet   {};
struct tag_particle {};

struct game_t {
  void recalc_paths() {
    vec2i goal = (engine.viewport_get_size() / 2.f).grid_cell(cfg_grid_size);
    registry.each([&](c_pos& pos, tag_enemy& enemy) {
      if (!enemy.flying)
        enemy.path = pathfinder.find_path(pos.v.grid_cell(cfg_grid_size), goal);
    });
  }

  bool has_los(vec2 p1, vec2 p2) {
    bool hit = false;
    world.raycast(p1, p2, [&](uint32_t id) {
      if (registry.has<tag_wall>(id) || registry.has<tag_obstacle>(id))
        hit = true;
    });
    return !hit;
  }

  void open() {
    vec2 vs = engine.viewport_get_size();
    vec2 center = vs / 2.f;
    engine.get_clear_color() = color(0.02f, 0.02f, 0.05f, 1.f);
    registry.clear();
    fan::spatial::auto_clean(world, registry);

    money = cfg_start_money;
    score = 0;
    wave = 1;
    core_health = cfg_core_health;
    is_action_phase = false;

    phase_timer.max = cfg_build_time;
    phase_timer.reset();
    spawn_timer.max = 0.f;
    spawn_timer.reset();
    player_shoot_cd.max = cfg_player_fire_cd;
    player_shoot_cd.reset();

    player_pos = center + vec2(0, 150);
    player_trail.color = colors::yellow.set_alpha(0.5f);
    player_trail.thickness = 6.f;

    player_shape  = rectangle_t{fan::vec3(player_pos, 4), vec2(15), colors::yellow};
    player_gun    = line_t{fan::vec3(player_pos, 4), player_pos + vec2(25, 0), colors::yellow, 4.f};
    build_preview = rectangle_t{fan::vec3(0, 0, 2), vec2(cfg_grid_size / 2.f - 2.f), colors::transparent};
    core_outer    = circle_t{fan::vec3(center, 3), 40.f, colors::cyan};
    core_inner    = circle_t{fan::vec3(center, 4), 30.f, colors::black};

    world.init(vec2(-1000), vec2(cfg_grid_size), vec2i(100));
    world.reset();

    vec2i grid_cells  = vs.grid_cell(cfg_grid_size);
    vec2i center_cell = grid_cells / 2;
    pathfinder.init(grid_cells, false);

    for (int i = 0; i < cfg_obstacle_count; ++i) {
      vec2i c(fan::random::value(0, grid_cells.x - 1), fan::random::value(0, grid_cells.y - 1));
      if (c == vec2i(0) || (vec2(c) - vec2(center_cell)).length_squared() <= 25.f)
        continue;
      vec2 opos = vec2(c) * cfg_grid_size + cfg_grid_size / 2.f;
      pathfinder.add_collision(c);
      uint32_t e = registry.create_with(c_pos{opos}, tag_obstacle{c});
      world.upsert(e, fan::physics::aabb_t::from_center(opos, vec2(cfg_grid_size / 2.f)), fan::spatial::movement_static);
    }

    bg_grid = grid_t{{
      .position = fan::vec3(-fan::window::get_size() / 2.f, 0),
      .size     = vs.max() * 2.f,
      .grid_size = vs.max() * 2.f / cfg_grid_size,
      .color    = color(0.1f, 0.2f, 0.4f, 0.8f)
    }};
  }

  void draw_gui() {
    if (core_health <= 0) {
      gui::push_style_color(gui::col_window_bg, color(0, 0, 0, 0.95f));
      if (auto w = gui::fullscreen_window("##end")) {
        vec2 vs = gui::get_display_size();
        gui::set_cursor_pos({0, vs.y / 2.f - 60.f});
        gui::text_centered_outlined_big("CORE DESTROYED", 48.f, colors::red, colors::black);
        gui::set_cursor_pos({
          (vs.x - (gui::calc_text_size("Restart").x + gui::get_style().FramePadding.x * 2.f)) / 2.f,
          vs.y / 2.f + 20.f
        });
        if (gui::button("Restart"))
          open();
      }
      gui::pop_style_color();
      return;
    }

    vec2 vs = gui::get_viewport_rect().size;
    hud_min_height = vs.y / 15.f;
    gui::set_next_window_size({vs.x, hud_min_height});
    gui::set_next_window_bg_alpha(cfg_windows_bg_alpha);
    gui::window_anchor_top_left(0.f);

    if (auto w = gui::window("##hud_phase", nullptr, gui::window_flags_overlay)) {
      std::string phase_text;
      if (is_action_phase && phase_timer.is_ready())
        phase_text = fan::format_args("ACTION PHASE  WAVE:", wave, "  (CLEARING...)");
      else
        phase_text = fan::format_args(is_action_phase ? "ACTION" : "BUILD", " PHASE  WAVE:", wave, "  (", (int)phase_timer.current, "s)");

      gui::text_centered(phase_text, colors::white, {-0.1f, 0.5f}, {-1.f, 0.f});

      if (!is_action_phase) {
        gui::same_line();
        if (gui::button_centered("Skip >>", 0.f, {0, 1}, {10.f, 0.f}))
          phase_timer.current = 0.f;
      }
      gui::text_centered(fan::format_args(money, " gold"), colors::yellow, {1.2f, 0.5f}, {0.f, 0.f});
      gui::same_line();
      gui::text("  SCORE: ", score);
      gui::anchor_center_right(vec2(-vs.x / 5.f, -10.f));
      gui::healthbar_labeled(
        "CORE HEALTH:", core_health, cfg_core_health, {200.f, 20.f},
        colors::cyan, colors::cyan, color(0.2f, 0.2f, 0.2f, 1.f)
      );
    }

    if (!is_action_phase) {
      vec2 bs{120, 70};
      vec2 sz{vs.x, vs.y / 9.f};
      hud_max_height = vs.y - sz.y;
      gui::set_next_window_size(sz);
      gui::window_anchor_bottom_center(vec2(-sz.x / 2.f, -sz.y));
      gui::set_next_window_bg_alpha(cfg_windows_bg_alpha);
      if (auto box = gui::window("##build", nullptr, gui::window_flags_overlay)) {
        gui::anchor_center(sz, bs, build_defs.size());
        gui::button_row(
          std::span(build_defs),
          [&](const auto& d) { return fan::str_view_t(d.label); },
          [&](const auto& d) { return money >= d.cost; },
          bs,
          [&](int i) { selected_build = i; }
        );
      }
    }
  }

  void update() {
    if (core_health <= 0) {
      draw_gui();
      return;
    }
    f32_t dt  = engine.get_delta_time();
    vec2 mpos = get_mouse_world_pos();
    vec2 vs   = engine.viewport_get_size();

    if (is_action_phase) {
      phase_timer.tick(dt);

      bool enemies_alive = false;
      registry.each_breakable<tag_enemy>([&](tag_enemy&) {
        return !(enemies_alive = true);
      });

      if (!phase_timer.is_ready()) {
        if (spawn_timer.tick(dt); spawn_timer.is_ready()) {
          f32_t interval = cfg_spawn.base_interval / std::sqrt((f32_t)wave);
          spawn_timer.max = std::max(0.05f, interval + fan::random::value(-cfg_spawn.interval_variance, cfg_spawn.interval_variance));
          spawn_timer.reset();
          vec2 sp  = fan::random::border_pos(vs, 50.f);
          bool fly = fan::random::value(0.f, 100.f) < std::min(cfg_spawn.base_fly_chance + cfg_spawn.fly_chance_per_wave * (wave - 1), cfg_spawn.max_fly_chance);
          auto path = fly ? std::vector<vec2i>{} : pathfinder.find_path(sp.grid_cell(cfg_grid_size), (vs / 2.f).grid_cell(cfg_grid_size));
          int initial_size = path.size();
          registry.create_with(
            tag_enemy{fly, std::move(path), initial_size},
            c_pos{sp}, c_vel{vec2(0)},
            c_hp{cfg_spawn.base_hp + wave * cfg_spawn.hp_per_wave, 0},
            c_rectangle{vec2(6), fly ? colors::magenta : colors::red, 3.f}
          );
        }
      } else if (!enemies_alive) {
        is_action_phase = false;
        phase_timer.max = cfg_build_time;
        phase_timer.reset();
        wave++;
        wait_for_mouse_release = true;
      }
    } else {
      if (phase_timer.tick(dt); phase_timer.is_ready()) {
        is_action_phase = true;
        phase_timer.max = 10.f + (wave - 1) * 5.f;
        phase_timer.reset();
      }
    }

    player_pos   = (player_pos + fan::window::get_input_vector() * cfg_player_speed * dt).clamp(vec2(0), vs);
    player_angle = std::atan2(mpos.y - player_pos.y, mpos.x - player_pos.x);
    player_trail.set_point(player_pos, 0.f);
    player_shape.set_position(player_pos);
    player_gun.set_line(player_pos, player_pos + vec2::from_angle(player_angle, 25.f));

    vec2i cell = mpos.grid_cell(cfg_grid_size);
    vec2  gpos = mpos.grid_floor(cfg_grid_size, cfg_grid_size / 2.f);

    if (!is_action_phase && !gui::want_io() && !(mpos.y < hud_min_height || mpos.y > hud_max_height)) {
      build_preview.set_position(gpos);
      build_preview.set_color(colors::transparent);
      build_preview.set_outline_color(colors::white);

      if (wait_for_mouse_release && !fan::window::is_mouse_down(fan::mouse_left))
        wait_for_mouse_release = false;

      auto occ = [&] {
        bool o = false;
        registry.each_breakable<c_pos, tag_wall>([&](c_pos& p, tag_wall&) { return !(o = p.v == gpos); });
        if (!o) registry.each_breakable<c_pos, tag_turret>([&](c_pos& p, tag_turret&) { return !(o = p.v == gpos); });
        if (!o) registry.each_breakable<tag_obstacle>([&](tag_obstacle& ob) { return !(o = ob.cell == cell); });
        return o;
      };

      if (!wait_for_mouse_release && fan::window::is_mouse_down(fan::mouse_left) && !occ() && money >= build_defs[selected_build].cost) {
        pathfinder.add_collision(cell);
        if (!pathfinder.is_fully_connected((vs / 2.f).grid_cell(cfg_grid_size))) {
          pathfinder.remove_collision(cell);
        } else {
          money -= build_defs[selected_build].cost;
          if (selected_build == 0) {
            uint32_t e = registry.create_with(c_pos{gpos}, tag_wall{}, c_hp{cfg_wall_hp, cfg_wall_hp});
            world.upsert(e, fan::physics::aabb_t::from_center(gpos, vec2(cfg_grid_size / 2.f)), fan::spatial::movement_static);
          } else {
            registry.create_with(c_pos{gpos}, tag_turret{fan::cooldown_t{}});
          }
          recalc_paths();
        }
      }

      if (fan::window::is_mouse_down(fan::mouse_right)) {
        bool c = false;
        registry.destroy_if<c_pos, tag_wall>([&](c_pos& p, tag_wall&) { return (c |= p.v == gpos), p.v == gpos; });
        registry.destroy_if<c_pos, tag_turret>([&](c_pos& p, tag_turret&) { return (c |= p.v == gpos), p.v == gpos; });
        if (c) {
          pathfinder.remove_collision(cell);
          recalc_paths();
        }
      }
    } else {
      build_preview.set_outline_color(colors::transparent);
    }

    player_shoot_cd.tick(dt);
    if (is_action_phase && fan::window::is_mouse_down(fan::mouse_left) && !gui::want_io() && player_shoot_cd.is_ready()) {
      player_shoot_cd.max = cfg_player_fire_cd;
      player_shoot_cd.reset();
      vec2 bv = vec2::from_angle(player_angle, cfg_bullet_speed);
      registry.create_with(tag_bullet{}, c_pos{player_pos}, c_vel{bv}, c_life{cfg_bullet_life}, c_line{bv.normalize() * 15.f, colors::yellow, 3.f});
    }

    fan::physics::auto_aim<tag_turret>(registry, world, dt, cfg_turret_range, cfg_turret_bspeed, cfg_turret_fire_cd,
      [&](uint32_t id, vec2 src) {
        return registry.has<tag_enemy>(id) && has_los(src, registry.get<c_pos>(id).v);
      },
      [&](vec2 src, vec2 dir) {
        registry.create_with(
          tag_bullet{}, c_pos{src + dir * (cfg_grid_size * 0.5f)},
          c_vel{dir * cfg_turret_bspeed}, c_life{1.f},
          c_line{dir * 15.f, colors::green, 3.f}
        );
      }
    );

    bool w_dead = false;
    registry.each([&](uint32_t e, c_pos& p, c_vel& v, tag_enemy& en) {
      vec2 sep = world.separation_force(e, p.v, cfg_enemy_sep_r, [&](uint32_t id) {
        return registry.has<tag_enemy>(id) ? registry.get<c_pos>(id).v : p.v;
      });
      bool bash = false;
      vec2 tgt  = vs / 2.f;
      if (!en.flying) {
        if (!en.path.empty()) {
          tgt = vec2(en.path.back()) * cfg_grid_size + (cfg_grid_size / 2.f);
          if ((p.v - tgt).length_squared() < 1600.f)
            en.path.pop_back();
        }
        if ((int)en.path.size() > cfg_path_bash_thresh) {
          tgt  = vs / 2.f;
          bash = true;
        }
      }

      f32_t speed_mul = 1.f + std::max(0, en.initial_path_size - cfg_path_len_thresh) * cfg_path_speed_bonus;
      vec2 move_dir   = (tgt - p.v).normalize() + sep * 0.5f;
      if (move_dir.length_squared() > 0.0001f)
        move_dir = move_dir.normalize();
      v.v = v.v * 0.9f + move_dir * (cfg_enemy_base_speed * speed_mul);

      if (!en.flying) {
        world.query_radius(p.v, cfg_grid_size * 0.75f, [&](uint32_t id) {
          vec2 col_extents = vec2(cfg_grid_size / 2.f - 0.1f);
          if (registry.has<tag_wall>(id) && fan::physics::aabb_t::from_center(registry.get<c_pos>(id).v, col_extents).push_out(p.v, 200.f * dt)) {
            auto& hp = registry.get<c_hp>(id);
            if ((hp.current -= bash ? cfg_wall_dmg_bash : cfg_wall_dmg) <= 0)
              w_dead = true;
            else
              registry.get<tag_wall>(id).outer_col = colors::red.lerp(color(0, 0.3f, 1.f, 1.f), (f32_t)hp.current / hp.max);
          } else if (registry.has<tag_obstacle>(id)) {
            fan::physics::aabb_t::from_center(registry.get<c_pos>(id).v, col_extents).push_out(p.v, 200.f * dt);
          }
        });
      }
    });

    if (w_dead) {
      registry.destroy_if<c_pos, c_hp, tag_wall>([&](c_pos& p, c_hp& hp, tag_wall&) {
        if (hp.current <= 0) {
          pathfinder.remove_collision(p.v.grid_cell(cfg_grid_size));
          return true;
        }
        return false;
      });
      recalc_paths();
    }

    registry.destroy_if<c_pos, tag_bullet>([&](c_pos& p, tag_bullet&) {
      bool d = false;
      world.query_radius(p.v, 12.f, [&](uint32_t id) {
        if (!d) {
          if (registry.has<tag_enemy>(id)) { registry.get<c_hp>(id).current -= 10; d = true; }
          else if (registry.has<tag_wall>(id) || registry.has<tag_obstacle>(id)) d = true;
        }
      });
      return d;
    });

    fan::physics::proximity_trigger<c_pos, tag_enemy>(registry, vs / 2.f, 40.f, [&](uint32_t e, c_pos& p) {
      core_health--;
      registry.destroy(e);
      fan::graphics::emit_radial(registry, p.v, colors::red, 10, {50.f, 200.f}, {0.2f, 0.6f});
    });
    fan::physics::proximity_trigger<c_pos, tag_enemy>(registry, player_pos, 20.f, [&](uint32_t e, c_pos& p) {
      core_health -= 5;
      registry.destroy(e);
      fan::graphics::emit_radial(registry, p.v, colors::yellow, 20, {50.f, 200.f}, {0.2f, 0.6f});
    });

    registry.destroy_if<c_pos, c_hp, tag_enemy>([&](c_pos& p, c_hp& hp, tag_enemy&) {
      if (hp.current <= 0) {
        money += 5;
        score += 100;
        fan::graphics::emit_radial(registry, p.v, colors::red, 5, {50.f, 200.f}, {0.2f, 0.6f});
        return true;
      }
      return false;
    });

    fan::ecs::systems::apply_drag<c_vel, tag_particle>(registry, 0.95f);
    fan::ecs::systems::kinematics<c_pos, c_vel>(registry, dt);
    fan::ecs::systems::lifetimes<c_life>(registry, dt);
    fan::spatial::sync_grid<c_pos, tag_enemy>(registry, world, vec2(6));

    auto* rv = &get_orthographic_render_view();
    fan::graphics::systems::render2d(registry, rv);

    registry.each<c_pos, tag_wall>([rv](c_pos& p, tag_wall& w) {
      fan::graphics::rectangle(fan::vec3(p.v, 2), vec2(cfg_grid_size / 2.f - 2.f), w.outer_col, rv);
      fan::graphics::rectangle(fan::vec3(p.v, 3), vec2(cfg_grid_size / 2.f - 6.f), colors::black, rv);
    });
    registry.each<c_pos, tag_obstacle>([rv](c_pos& p, tag_obstacle&) {
      fan::graphics::rectangle(fan::vec3(p.v, 1.f), vec2(cfg_grid_size / 2.f - 2.f), color(0.2f, 0.2f, 0.2f, 1.f), rv);
    });
    registry.each<c_pos, tag_turret>([rv](c_pos& p, tag_turret&) {
      fan::graphics::rectangle(fan::vec3(p.v, 2), vec2(cfg_grid_size / 2.f - 4.f), colors::green, rv);
      fan::graphics::rectangle(fan::vec3(p.v, 3), vec2(cfg_grid_size / 2.f - 8.f), colors::black, rv);
    });

    draw_gui();
  }

  engine_t engine{{ .vsync = false }};
  using registry_t = fan::ecs_t<
    c_pos, c_vel, c_hp, c_life, c_rectangle, c_line,
    tag_wall, tag_obstacle, tag_enemy, tag_turret, tag_bullet, tag_particle
  >;
  registry_t registry;
  fan::spatial::world_t<uint32_t> world;
  fan::graphics::algorithm::pathfind::generator pathfinder;

  int money = cfg_start_money;
  int score = 0;
  int wave = 1;
  int core_health = cfg_core_health;
  int selected_build = 0;
  bool is_action_phase = false;
  bool wait_for_mouse_release = false;
  fan::cooldown_t phase_timer, spawn_timer, player_shoot_cd;
  vec2 player_pos;
  f32_t player_angle = 0.f;

  trail_t player_trail;
  grid_t bg_grid;
  circle_t core_outer, core_inner;
  rectangle_t player_shape, build_preview;
  line_t player_gun;
};

int main() {
  game_t game;
  game.open();
  game.engine.loop([&] { game.update(); });
}