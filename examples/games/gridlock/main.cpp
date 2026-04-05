#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <unordered_set>
#include <span>
#include <array>

import fan;

using namespace fan::graphics;
using namespace fan::ecs;
using fan::vec2;
using fan::vec2i;
using fan::color;
namespace colors = fan::colors;

static constexpr gui::text_style_t style_hud_left  = {.text_offset={0.5f, 0.5f}, .window_offset = {-0.9f,0.f}};
static constexpr gui::text_style_t style_hud_right = {.text_offset={1.2f,0.5f},  .window_offset={0.f,0.f}};

static constexpr f32_t cfg_grid         = 64.f;
static constexpr f32_t cfg_build_time   = 10.f;
static constexpr f32_t cfg_player_speed = 400.f;
static constexpr f32_t cfg_bullet_speed = 1200.f;
static constexpr f32_t cfg_bullet_life  = 2.f;
static constexpr f32_t cfg_turret_range = 2000.f;
static constexpr f32_t cfg_turret_bspd  = 1000.f;
static constexpr f32_t cfg_turret_cd    = 0.2f;
static constexpr f32_t cfg_enemy_sep    = 24.f;
static constexpr f32_t cfg_enemy_speed  = 8.f;
static constexpr f32_t cfg_path_bonus   = 0.04f;
static constexpr int   cfg_path_thresh  = 10;
static constexpr int   cfg_bash_thresh  = 20;
static constexpr int   cfg_start_money  = 50;
static constexpr int   cfg_core_hp      = 20;
static constexpr int   cfg_obstacles    = 30;
static constexpr int   cfg_wall_hp      = 60;
static constexpr f32_t cfg_action_time_base     = 10.f;
static constexpr f32_t cfg_action_time_per_wave =  5.f;

static constexpr struct {
  f32_t interval = 0.8f, variance = 0.2f;
  int   base_hp = 10, hp_per_wave = 5;
  f32_t fly_base = 0.f, fly_per_wave = 10.f, fly_max = 60.f;
} cfg_spawn;

struct build_data_t { std::string_view title; uint32_t price; };
static constexpr std::array<build_data_t, 2> builds = {{
  {"Wall\n[$10]",    10},
  {"Turret\n[$100]", 100}
}};

struct tag_wall     {};
struct tag_obstacle { vec2i cell; };
struct tag_enemy    { bool flying; std::vector<vec2i> path; int initial_path_size = 0; };
struct tag_turret   { fan::cooldown_t cd; };
struct tag_particle {};

struct game_t {
  game_t() { open(); engine.loop([&]{ update(); }); }

  void recalc_paths() {
    vec2i goal = (fan::window::get_size() / 2.f).grid_cell(cfg_grid);
    registry.each([&](c_pos& p, tag_enemy& e) {
      if (!e.flying) e.path = pathfinder.find_path(p.v.grid_cell(cfg_grid), goal);
    });
  }

  void place(vec2i cell, vec2 gpos) {
    pathfinder.add_collision(cell);
    if (!pathfinder.is_fully_connected((fan::window::get_size() / 2.f).grid_cell(cfg_grid))) {
      pathfinder.remove_collision(cell); return;
    }
    money -= builds[selected_build].price;
    occupied.insert(cell);
    if (selected_build == 0) {
      fan::spatial::create_static(registry, world, gpos, vec2(cfg_grid / 2.f),
        c_pos{gpos}, tag_wall{}, c_hp{cfg_wall_hp, cfg_wall_hp},
        c_rectangle_bordered{vec2(cfg_grid/2.f-2.f), color(0, 0.3f, 1.f, 1.f), vec2(cfg_grid/2.f-6.f), colors::black});
    } else {
      registry.create_with(c_pos{gpos}, tag_turret{},
        c_rectangle_bordered{vec2(cfg_grid/2.f-4.f), colors::green, vec2(cfg_grid/2.f-8.f), colors::black});
    }
    paths_dirty = true;
  }

  void remove(vec2i cell, vec2 gpos) {
    registry.destroy_at<tag_wall, tag_turret>(gpos, [&] {
      pathfinder.remove_collision(cell); occupied.erase(cell); paths_dirty = true;
    });
  }

  void open() {
    vec2 vs = fan::window::get_size(), center = vs / 2.f;
    engine.get_clear_color() = color(0.02f, 0.02f, 0.05f, 1.f);
    registry.clear(); fan::spatial::auto_clean(world, registry);
    money = cfg_start_money; score = 0; wave = 1; core_hp = cfg_core_hp;
    is_action = paths_dirty = any_spawned = false; occupied.clear();

    phase_cd = fan::cooldown_t::full(cfg_build_time);
    spawn_cd = fan::cooldown_t::full(1.f);
    shoot_cd = fan::cooldown_t::full(0.1f);

    player_pos = center + vec2(0, 150);
    player_trail.color = colors::yellow.set_alpha(0.5f); player_trail.thickness = 6.f;
    player_shape  = rectangle_t{fan::vec3(player_pos, 4), vec2(15), colors::yellow};
    player_gun    = line_t{fan::vec3(player_pos, 4), player_pos + vec2(25, 0), colors::yellow, 4.f};
    build_preview = rectangle_t{fan::vec3(0, 0, 2), vec2(cfg_grid / 2.f - 2.f), colors::transparent};
    core_outer    = circle_t{fan::vec3(center, 3), 40.f, colors::cyan};
    core_inner    = circle_t{fan::vec3(center, 4), 30.f, colors::black};

    world.init(vec2(-1000), vec2(cfg_grid), vec2i(100)); world.reset();
    vec2i gcells = vs.grid_cell(cfg_grid), gcenter = gcells / 2;
    pathfinder.init(gcells, false);
    for (int i = 0; i < cfg_obstacles; ++i) {
      vec2i c = fan::random::vec(vec2i(0), gcells - 1);
      if (c == vec2i(0) || (vec2(c) - vec2(gcenter)).length_squared() <= 25.f) continue;
      vec2 opos = vec2(c) * cfg_grid + cfg_grid / 2.f;
      pathfinder.add_collision(c); occupied.insert(c);
      fan::spatial::create_static(registry, world, opos, vec2(cfg_grid / 2.f), 
        c_pos{opos}, tag_obstacle{c}, c_rectangle{vec2(cfg_grid/2.f-2.f), color(0.2f,0.2f,0.2f,1.f)});
    }
    bg_grid = grid_t{{
      .position = fan::vec3(-vs / 2.f, 0),
      .size = vs.max() * 2.f, .grid_size = vs.max() * 2.f / cfg_grid,
      .color = color(0.1f, 0.2f, 0.4f, 0.8f)
    }};
  }

  void draw_gui() {
    vec2 vs = fan::window::get_size();
    if (core_hp <= 0) {
      auto _ = gui::style_scope_t{gui::col_window_bg, color(0,0,0,0.95f)};
      if (auto w = gui::fullscreen_window("##end")) {
        gui::text_outlined("CORE DESTROYED", {.color=colors::red, .font_size=48.f, .offset={0,-60.f}, .window_offset={0.f,0.f}});
        if (gui::button_centered("Restart", {.offset={0,10.f}})) open();
      }
      return;
    }
    gui::window_anchor_top_left(0.f);
    if (auto w = gui::overlay_window("##hud", {vs.x, vs.y / 15.f})) {
      std::string pt;
      if (is_action && phase_cd.is_ready()) pt = fan::format_args("ACTION PHASE  WAVE:", wave, "  (CLEARING...)");
      else pt = fan::format_args(is_action ? "ACTION" : "BUILD", " PHASE  WAVE:", wave, "  (", (int)phase_cd.current, "s)");
      gui::text(pt, style_hud_left);
      if (!is_action) { gui::same_line(); if (gui::button_centered("Skip >>", {.affects_axis={0,1}, .offset={10.f,0.f}})) phase_cd.expire(); }
      gui::text(fan::format_args(money, " gold"), style_hud_right);
      gui::same_line(); gui::text({.text_offset={-0.5f, 0.5f}, .window_offset{0.f}}, "  SCORE: ", score);
      gui::anchor_center_right(vec2(-vs.x / 5.f, -10.f));
      gui::healthbar_labeled("CORE HEALTH:", core_hp, cfg_core_hp, {200.f,20.f}, colors::cyan, colors::cyan, color(0.2f,0.2f,0.2f,1.f));
    }
    if (!is_action) {
      vec2 bs{120,70}, sz{vs.x, vs.y/9.f};
      gui::window_anchor_bottom_center(vec2(-sz.x/2.f,-sz.y));
      if (auto w = gui::overlay_window("##build", sz)) {
        gui::anchor_center(sz, bs, builds.size());
        gui::button_row(std::span(builds),
          [](const auto& d) { return fan::str_view_t(d.title); },
          [&](const auto& d) { return money >= d.price; },
          bs, [&](int i) { selected_build = i; });
      }
    }
  }

  void update() {
    if (core_hp <= 0) { draw_gui(); return; }
    if (paths_dirty)  { recalc_paths(); paths_dirty = false; }

    vec2 vs = fan::window::get_size();
    vec2 mpos = get_mouse_world_pos();
    f32_t dt = engine.get_delta_time();

    if (is_action) {
      phase_cd.tick(dt);
      bool alive = registry.any<tag_enemy>();
      if (!phase_cd.is_ready()) {
        if (spawn_cd.tick_ready(dt)) {
          f32_t iv = cfg_spawn.interval / std::sqrt((f32_t)wave);
          spawn_cd = fan::cooldown_t::full(std::max(0.05f, iv + fan::random::value(-cfg_spawn.variance, cfg_spawn.variance)));
          vec2 sp = fan::random::border_pos(vs, 50.f);
          bool fly = fan::random::value(0.f,100.f) < std::min(cfg_spawn.fly_base + cfg_spawn.fly_per_wave*(wave-1), cfg_spawn.fly_max);
          auto path = fly ? std::vector<vec2i>{} : pathfinder.find_path(sp.grid_cell(cfg_grid), (vs/2.f).grid_cell(cfg_grid));
          int psz = path.size();
          registry.create_with(tag_enemy{fly, std::move(path), psz}, c_pos{sp}, c_vel{vec2(0)},
            c_hp{cfg_spawn.base_hp + wave*cfg_spawn.hp_per_wave, 0}, c_rectangle{vec2(6), fly ? colors::magenta : colors::red, 3.f});
          any_spawned = true;
        }
      } else if (!alive && any_spawned) {
        is_action = false;
        phase_cd = fan::cooldown_t::full(cfg_build_time);
        wave++; no_click = true; any_spawned = false;
      }
    } else {
      if (phase_cd.tick_ready(dt)) {
        is_action = true;
        phase_cd = fan::cooldown_t::full(cfg_action_time_base + (wave-1)*cfg_action_time_per_wave);
      }
    }

    player_pos = (player_pos + fan::window::get_input_vector() * cfg_player_speed * dt).clamp(vec2(0), vs);
    player_angle = (mpos - player_pos).angle();
    player_trail.set_point(player_pos, 0.f);
    player_shape.set_position(player_pos);
    player_gun.set_line(player_pos, player_pos + vec2::from_angle(player_angle, 25.f));

    vec2i cell = mpos.grid_cell(cfg_grid);
    vec2  gpos = mpos.grid_floor(cfg_grid, cfg_grid / 2.f);
    if (!is_action && !gui::want_io()) {
      build_preview.set_position(gpos); build_preview.set_outline_color(colors::white);
      if (no_click && !fan::window::is_mouse_down(fan::mouse_left)) no_click = false;
      if (!no_click && fan::window::is_mouse_down(fan::mouse_left) && !occupied.contains(cell) && money >= builds[selected_build].price)
        place(cell, gpos);
      if (fan::window::is_mouse_down(fan::mouse_right)) remove(cell, gpos);
    } else {
      build_preview.set_outline_color(colors::transparent);
    }

    if (is_action && fan::window::is_mouse_down(fan::mouse_left) && !gui::want_io() && shoot_cd.tick_ready(dt)) {
      shoot_cd.expire();
      vec2 bv = vec2::from_angle(player_angle, cfg_bullet_speed);
      registry.create_with(tag_bullet{}, c_pos{player_pos}, c_vel{bv}, c_life{cfg_bullet_life}, c_line{bv.normalize()*15.f, colors::yellow, 3.f});
    }

    fan::physics::auto_aim<tag_turret>(registry, world, dt, cfg_turret_range, cfg_turret_bspd, cfg_turret_cd,
      [&](uint32_t id, vec2 src) { 
        return registry.has<tag_enemy>(id) && 
          fan::physics::has_los<tag_wall, tag_obstacle>(registry, world, src, registry.get<c_pos>(id).v); 
      },
      [&](vec2 src, vec2 dir) {
        registry.create_with(tag_bullet{}, c_pos{src + dir*(cfg_grid*0.5f)},
          c_vel{dir*cfg_turret_bspd}, c_life{1.f}, c_line{dir*15.f, colors::green, 3.f});
      }
    );

    registry.each([&](uint32_t e, c_pos& p, c_vel& v, tag_enemy& en) {
      vec2 sep = fan::physics::separation_force<tag_enemy>(registry, world, e, p.v, cfg_enemy_sep);
      auto [tgt, bash] = en.flying
        ? fan::pathfind::follow_result_t{vs/2.f, false}
        : fan::pathfind::follow(en.path, p.v, vs/2.f, cfg_grid, cfg_bash_thresh);
      f32_t spd = 1.f + std::max(0, en.initial_path_size - cfg_path_thresh) * cfg_path_bonus;
      v.v = fan::physics::steer_toward(p.v, v.v, tgt, sep, {cfg_enemy_speed}, spd);
      if (!en.flying)
        fan::physics::push_out_walls<tag_wall, tag_obstacle>(registry, world, p.v, cfg_grid, dt, bash,
          [&](uint32_t id, bool bash) {
            auto& hp = registry.get<c_hp>(id);
            if ((hp.current -= bash ? 3 : 1) > 0)
              registry.get<c_rectangle_bordered>(id).outer_col = colors::red.lerp(color(0,0.3f,1.f,1.f), (f32_t)hp.current/hp.max);
          });
    });

    if (registry.destroy_dead<c_pos, tag_wall>([&](c_pos& p, tag_wall&) {
      vec2i c = p.v.grid_cell(cfg_grid); pathfinder.remove_collision(c); occupied.erase(c);
    })) {
      paths_dirty = true;
    }

    fan::physics::tick_bullets<tag_enemy, tag_wall, tag_obstacle>(registry, world, 12.f, 10);

    fan::graphics::physics::proximity_damage<tag_enemy>(registry, vs/2.f,     40.f, core_hp, 1, colors::red,    10);
    fan::graphics::physics::proximity_damage<tag_enemy>(registry, player_pos, 20.f, core_hp, 5, colors::yellow, 20);

    registry.destroy_dead<c_pos, tag_enemy>([&](c_pos& p, tag_enemy&) {
      money += 5; score += 100;
      fan::graphics::emit_radial(registry, p.v, colors::red, 5, {50.f,200.f}, {0.2f,0.6f});
    });

    fan::ecs::systems::apply_drag<c_vel, tag_particle>(registry, 0.95f);
    fan::ecs::systems::kinematics<c_pos, c_vel>(registry, dt);
    fan::ecs::systems::lifetimes<c_life>(registry, dt);
    fan::spatial::sync_grid<c_pos, tag_enemy>(registry, world, vec2(6));

    fan::graphics::systems::render2d(registry);
    draw_gui();
  }

  engine_t engine{{.vsync = false }};
  using registry_t = fan::ecs_t<
    c_pos, c_vel, c_hp, c_life, c_rectangle, c_rectangle_bordered, c_line,
    tag_wall, tag_obstacle, tag_enemy, tag_turret, tag_bullet, tag_particle
  >;
  registry_t registry;
  fan::spatial::world_t<uint32_t> world;
  fan::pathfind::generator pathfinder;
  std::unordered_set<vec2i> occupied;

  int   money = cfg_start_money, score = 0, wave = 1, core_hp = cfg_core_hp, selected_build = 0;
  bool  is_action = false, no_click = false, paths_dirty = false, any_spawned = false;
  f32_t player_angle = 0;
  fan::cooldown_t phase_cd, spawn_cd, shoot_cd;
  vec2 player_pos;

  trail_t player_trail; grid_t bg_grid;
  circle_t core_outer, core_inner;
  rectangle_t player_shape, build_preview;
  line_t player_gun;
};

int main() {
  game_t{};
}