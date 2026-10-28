import std;
import fan;

static constexpr f32_t cfg_bullet_life  = 1.5f;

static constexpr auto player_size = fan::vec2(48.f);
static constexpr auto capsule_center0 = fan::vec2(0.f, -24.f);
static constexpr auto capsule_center1 = fan::vec2(0.f, 24.f);
static constexpr auto capsule_radius = player_size.x / 2.f;
static constexpr auto depth_background = 100;
static constexpr auto depth_tiles = 200;
static constexpr auto depth_player = 300;
static constexpr auto depth_gun = 310;
static constexpr auto depth_target = 320;
static constexpr auto depth_trail = 350;
static constexpr auto depth_bullet = 400;
static constexpr auto depth_muzzle = 450;
static constexpr auto depth_casing = 500;
static constexpr auto depth_feet = 0xfffa;
static constexpr auto depth_crosshair = 0xfffd;
static constexpr auto depth_flash = 0xfffe;
static constexpr auto feet_y_offset = capsule_center1.y + capsule_radius;
static constexpr auto feet_x_offset = capsule_radius;
static constexpr f32_t cfg_muzzle_flash_life = 0.01f;
static constexpr f32_t cfg_muzzle_sprite_size = 32.f;
static constexpr std::size_t cfg_max_casings = 128;
static constexpr f32_t cfg_casing_settle_time = 2.f;
static constexpr auto aim_origin_offset = fan::vec2(0.f, -4.f);
static constexpr f32_t cfg_gun_length = 40.f;
static constexpr f32_t cfg_spread_decay = 0.6f; // radians per second
static constexpr auto target_size = fan::vec2(20.f, 32.f);
static constexpr int cfg_target_hp = 12;
static constexpr f32_t cfg_target_respawn = 1.5f;
static constexpr f32_t cfg_target_flash = 0.06f;
static constexpr f32_t cfg_target_knockback = 260.f;
static constexpr f32_t cfg_target_spring = 120.f;
static constexpr f32_t cfg_target_damping = 10.f;
static constexpr f32_t cfg_hit_marker_life = 0.12f;
static constexpr f32_t cfg_popup_life = 0.7f;
static constexpr f32_t cfg_popup_merge_window = 0.1f;
static constexpr f32_t cfg_kill_hitstop = 0.07f;
static constexpr f32_t cfg_challenge_countdown = 3.f;

struct weapon_t {
  const char* name;
  bool automatic;
  f32_t cooldown;
  int damage;
  int pellets;
  f32_t speed;
  f32_t base_spread;     // half-angle, radians
  f32_t spread_per_shot; // bloom added each shot
  f32_t max_spread;
  f32_t recoil;
  f32_t shake;
  f32_t bump;
  f32_t hitstop;         // seconds, on hit
  fan::color tracer;
};

static const std::array weapons{
  weapon_t{.name = "Rifle", .automatic = true, .cooldown = 0.06f, .damage = 1, .pellets = 1, .speed = 8000.f,
    .base_spread = 0.01f, .spread_per_shot = 0.012f, .max_spread = 0.08f, .recoil = 300.f, .shake = 0.01f, .bump = 0.3f,
    .hitstop = 0.f, .tracer = fan::color(1.0, 0.7, 0.1)},
  weapon_t{.name = "Shotgun", .automatic = false, .cooldown = 0.55f, .damage = 1, .pellets = 8, .speed = 6500.f,
    .base_spread = 0.16f, .spread_per_shot = 0.f, .max_spread = 0.16f, .recoil = 900.f, .shake = 0.04f, .bump = 1.2f,
    .hitstop = 0.05f, .tracer = fan::color(1.0, 0.5, 0.1)},
  weapon_t{.name = "Sniper", .automatic = false, .cooldown = 0.8f, .damage = 6, .pellets = 1, .speed = 14000.f,
    .base_spread = 0.f, .spread_per_shot = 0.f, .max_spread = 0.f, .recoil = 700.f, .shake = 0.05f, .bump = 1.5f,
    .hitstop = 0.09f, .tracer = fan::color(0.6, 0.9, 1.0)},
};

struct player_t {
  void init(fan::vec2 spawn_point) {
    body = fan::graphics::physics::character_capsule_sprite(
      fan::vec3(spawn_point, depth_player),
      capsule_center0,
      capsule_center1,
      player_size,
      fan::graphics::get_default_texture(),
      {.friction = 0.6f, .fixed_rotation = true, .filter = {.categoryBits = 2, .maskBits = 0xFFFFFFFF}}
    );
    body.set_color(fan::color(0.2f, 0.6f, 1.0f));
    body.enable_default_movement();
    body.enable_double_jump();
    body.set_jump_height(48.f);
  }

  fan::graphics::physics::character2d_t body;
};

struct tag_muzzle_flash {};
struct c_damage { int v; };

using casing_body_t = fan::graphics::physics::rectangle_t;

struct casing_t {
  casing_body_t body;
  f32_t age = 0;
  bool settled = false;
};

// springs back to home after knockback, home moves along base +- axis for sliders
struct target_t {
  fan::vec2 base;
  fan::vec2 axis = 0;
  f32_t speed = 0;
  f32_t phase = 0;

  fan::vec2 home = base;
  fan::vec2 pos = base;
  fan::vec2 vel = 0;
  int hp = cfg_target_hp;
  f32_t flash = 0;
  f32_t respawn = 0;
  f32_t squish = 0;

  bool alive() const { return hp > 0; }
  bool hanging() const { return axis.x == 0.f && axis.y == 0.f; }
  target_t reset() const { return target_t{.base = base, .axis = axis, .speed = speed, .phase = phase}; }
};

struct damage_popup_t {
  fan::vec2 pos;
  fan::vec2 vel;
  int amount;
  f32_t age = 0;
  bool kill = false;
  std::size_t target;
};

enum class challenge_e { idle, countdown, running, done };

// returns entry t in [0, 1] of segment a->b against box, or -1
static f32_t segment_vs_aabb(fan::vec2 a, fan::vec2 b, fan::vec2 center, fan::vec2 half) {
  fan::vec2 d = b - a;
  f32_t tmin = 0.f, tmax = 1.f;
  for (int i = 0; i < 2; ++i) {
    f32_t lo = center[i] - half[i], hi = center[i] + half[i];
    if (std::abs(d[i]) < 1e-6f) {
      if (a[i] < lo || a[i] > hi) return -1.f;
      continue;
    }
    f32_t t0 = (lo - a[i]) / d[i], t1 = (hi - a[i]) / d[i];
    if (t0 > t1) std::swap(t0, t1);
    tmin = std::max(tmin, t0);
    tmax = std::min(tmax, t1);
    if (tmin > tmax) return -1.f;
  }
  return tmin;
}

static fan::vec2 from_angle(f32_t a) {
  return fan::vec2(std::cos(a), std::sin(a));
}

using registry_t = fan::ecs_t<
  fan::ecs::c_pos, fan::ecs::c_vel, fan::ecs::c_life, fan::ecs::c_line,
  fan::ecs::tag_bullet,
  tag_muzzle_flash,
  c_damage
>;

struct pile_t : fan::graphics::engine_t, fan::frame_task_t<pile_t> {

  struct example_stage_t : fan::stage_t<example_stage_t> {
    void open(void* sod) {
      map_id = renderer.open_map("map.json", {
        .position = fan::vec2(0),
        .size = fan::vec2i(64, 32),
        .depth_fn = [](auto&, auto&, auto&, f32_t) { return depth_tiles; },
      });

      tile_world.init(fan::vec2(-10000), fan::vec2(256), fan::vec2i(256));
      renderer.iterate_tiles(map_id, [&, i = 0u](const auto& tile) mutable {
        collisions.emplace_back(engine->get_physics_context().create_box(
          tile.position, tile.size, 0, fan::physics::body_type_e::static_body,
          fan::physics::shape_properties_t{.friction=1.f}
        ));
        tile_world.upsert(i++, fan::physics::aabb_t::from_center(tile.position, tile.size), fan::spatial::movement_static);
      });

      player.init(renderer.get_spawn(map_id));
      player.body.movement_state.jump_state.on_jump = [this](int jump_type) {
        auto squished = fan::vec2(player_size.x * 1.5f, player_size.y * 0.5f);
        player.body.squish(squished, player_size, 0.7f);
        auto pos = player.body.get_position();

        int count = jump_type == 1 ? 6 : 8;
        f32_t alive_min = jump_type == 1 ? 0.3f : 0.4f;
        f32_t alive_max = jump_type == 1 ? 0.6f : 0.8f;

        particles.spawn_footstep_dust(fan::vec3(pos.x, pos.y + feet_y_offset, depth_feet), feet_x_offset, count, smoke_image, alive_min, alive_max);
      };
      engine->camera_follow(player.body.get_position(), 0);
      ic.set_zoom(1.728f);

      // placed in open cells of map.json (64px cells, spawn is cell 32,17)
      auto cell = [](f32_t x, f32_t y) { return fan::vec2(x, y) * 64.f; };
      targets = {
        target_t{.base = cell(36, 15)},
        target_t{.base = cell(26, 15)},
        target_t{.base = cell(24, 17)},
        target_t{.base = cell(47, 13)},
        target_t{.base = cell(38, 14), .axis = fan::vec2(256.f, 0.f), .speed = 1.1f},
        target_t{.base = cell(38, 19), .axis = fan::vec2(320.f, 0.f), .speed = 1.7f, .phase = 1.f},
        target_t{.base = cell(49, 15.5f), .axis = fan::vec2(0.f, 96.f), .speed = 2.3f},
      };

engine->shadow_enable_tile_mode();
      engine->shadow_set_darkness(0.35f);

      std::vector<fan::vec4> tiles;
      renderer.iterate_tiles(map_id, [&](const auto& tile) mutable {
        tiles.push_back({tile.position.x - tile.size.x, tile.position.y - tile.size.y, tile.position.x + tile.size.x, tile.position.y + tile.size.y});
      });
      engine->shadow_set_tile_occluders(tiles);
    }

    void close() {
      tile_world.reset();
      for (auto& c : collisions) { c.destroy(); }
      collisions.clear();
      casings.clear();
      targets.clear();
      popups.clear();
      renderer.close_map(map_id);
    }

    const weapon_t& weapon() const { return weapons[weapon_index]; }

    fan::vec2 aim_origin() {
      return player.body.get_position() + aim_origin_offset;
    }

    void fire() {
      const weapon_t& w = weapon();
      shoot_task = start_shoot_cd(w.cooldown);
      count_shot();

      f32_t aim_angle = std::atan2(aim_dir.y, aim_dir.x);
      fan::vec2 muzzle_pos = aim_origin() + aim_dir * cfg_gun_length;

      // gun recoil movement via knockback system
      player.body.movement_state.knockback_initial_velocity.x = -aim_dir.x * w.recoil;
      player.body.movement_state.knockback_ticks_remaining = 3;
      player.body.movement_state.is_in_knockback = true;

      for (int i = 0; i < w.pellets; ++i) {
        f32_t a = aim_angle + fan::random::value(-spread, spread);
        f32_t speed = w.speed * fan::random::value(0.95f, 1.05f);
        fan::vec2 bullet_vel = from_angle(a) * speed;
        registry.create_with(fan::ecs::tag_bullet{}, fan::ecs::c_pos{muzzle_pos},
          fan::ecs::c_vel{bullet_vel},
          fan::ecs::c_life{cfg_bullet_life},
          c_damage{w.damage},
          fan::ecs::c_line{bullet_vel.normalize() * 15.f, w.tracer, 3.f, depth_bullet});
      }
      spread = std::min(spread + w.spread_per_shot, w.max_spread);
      gun_kick = 1.f;

      // muzzle flash burst
      auto flash_col = fan::color(1.0, 0.6, 0.05);
      for (auto [offset, weight] : {
        std::pair{aim_dir * 28.f, 2.5f},
        std::pair{from_angle(aim_angle - 0.02f) * 18.f, 2.f},
      }) {
        registry.create_with(tag_muzzle_flash{}, fan::ecs::c_pos{muzzle_pos},
          fan::ecs::c_life{0.08f},
          fan::ecs::c_line{offset * 4.f, flash_col, weight, depth_muzzle});
      }

      muzzle_task = render_muzzle_flash(muzzle_pos);

      {
        f32_t facing = aim_dir.x >= 0.f ? 1.f : -1.f;
        fan::vec2 casing_vel = fan::vec2(-facing * 400.f, -450.f);
        if (casings.size() >= cfg_max_casings) {
          casings.pop_front();
        }
        casings.push_back({casing_body_t{{
          .position = fan::vec3(muzzle_pos - aim_dir * (cfg_gun_length * 0.5f), depth_casing),
          .size = fan::vec2(3.f, 2.f),
          .color = fan::colors::orange,
          //.image = fan::graphics::image_t{fan::colors::orange},
          .body_type = fan::physics::body_type_e::dynamic_body,
          .shape_properties = {.friction=0.4f, .filter={4, ~std::uint32_t(2 | 4), 0}},
        }}});
        casings.back().body.set_linear_velocity(casing_vel);
      }

      // camera effects
      ic.shake(w.shake, 0.01f + w.shake);
      ic.bump(-aim_dir, w.bump, 0.06f);
      ic.bump_zoom(-0.01f, 0.04f);
      ic.flash(0.03f, 0.04f);
    }

    void update() {
      fan::graphics::sprite(fan::vec3(fan::vec2(0, 0), 0), 12300, fan::colors::gray);

      f32_t real_dt = engine->get_delta_time();
      f32_t dt = real_dt;

      if (is_hitstop) {
        dt *= 0.08f;
      }
      world_time += dt;

      if (engine->are_keys_down(fan::key_left_control, fan::key_left_shift) && engine->is_key_clicked(fan::key_r)) {
        fan::image::async_cache().clear();
        engine->stage_restart<example_stage_t>();
      }

      bool on_ground = player.body.is_on_ground();
      if (!player.body.was_on_ground && on_ground) {
        auto pos = player.body.get_position();
        particles.spawn_footstep_dust(fan::vec3(pos.x, pos.y + feet_y_offset, depth_feet), feet_x_offset, 6, smoke_image, 0.3f, 0.6f);
        // land squish
        auto squished = fan::vec2(player_size.x * 1.2f, player_size.y * 0.7f);
        player.body.squish(squished, player_size, 0.5f);
      }
      player.body.was_on_ground = on_ground;

      particles.update(dt);

      update_input();

      // bullet trails
      registry.each<fan::ecs::c_pos, fan::ecs::c_vel, fan::ecs::tag_bullet>([&](std::uint32_t, fan::ecs::c_pos& p, fan::ecs::c_vel& v, fan::ecs::tag_bullet&) {
        registry.create_with(
          fan::ecs::c_pos{p.v - v.v * (dt * 0.5f)},
          fan::ecs::c_life{0.25f},
          fan::ecs::c_line{-v.v * (dt * 0.8f), fan::colors::yellow.set_alpha(0.45f), 2.5f, depth_trail}
        );
      });

      update_bullet_hits(dt);
      update_targets(dt);
      update_challenge(real_dt);

      // let casings collide with each other once they have flown clear of the player
      for (auto& c : casings) {
        c.age += dt;
        if (!c.settled && c.age > cfg_casing_settle_time) {
          c.body.set_filter({4, ~std::uint32_t(4), 0});
          c.settled = true;
        }
      }

      fan::ecs::systems::kinematics<fan::ecs::c_pos, fan::ecs::c_vel>(registry, dt);
      fan::ecs::systems::lifetimes<fan::ecs::c_life>(registry, dt);

      spread = std::max(spread - cfg_spread_decay * dt, weapon().base_spread);
      gun_kick = std::max(0.f, gun_kick - dt * 12.f);
      hit_marker = std::max(0.f, hit_marker - real_dt);

      renderer.update(map_id, player.body.get_center());
      engine->camera_follow(player.body.get_position());
      ic.update_fx(dt);

      fan::vec2 plpos = player.body.get_position();
      fan::vec2 lpos = plpos - fan::vec2(0.f, 8.f);
      fan::vec2 spos = renderer.get_spawn(map_id) - fan::vec2(0.f, 8.f);

      engine->shadow_clear_lights();
      engine->shadow_add_light(lpos, pl_rad, pl_col.set_alpha(pl_shadow_a), pl_shadow_softness);
      engine->shadow_add_light(spos, sp_rad, sp_col.set_alpha(sp_shadow_a), sp_shadow_softness);

      fan::graphics::light(fan::vec3(lpos, depth_muzzle), fan::vec2(pl_rad), pl_col.set_alpha(pl_visual_a));
      fan::graphics::light(fan::vec3(spos, depth_muzzle), fan::vec2(sp_rad), sp_col.set_alpha(sp_visual_a));

      fan::graphics::systems::render2d(registry);

      draw_gun();
      draw_crosshair();

      // flash overlay
      f32_t flash_alpha = ic.get_flash_alpha();
      if (flash_alpha > 0) {
        fan::vec2 s = ic.get_viewport_size() / ic.get_zoom();
        fan::vec2 center = fan::graphics::camera_get_position(ic.render_view.camera);
        fan::graphics::rectangle(fan::vec3(center, depth_flash), s, fan::color(1, 1, 1, flash_alpha));
      }

      update_popups(dt);
      draw_hud();
      draw_gui();
    }

    void update_input() {
      mouse_world = fan::graphics::get_mouse_world_pos();
      fan::vec2 to_mouse = mouse_world - aim_origin();
      if (to_mouse.length() > 1.f) {
        aim_dir = to_mouse.normalize();
      }

      for (int i = 0; i < (int)weapons.size(); ++i) {
        if (engine->is_key_clicked(fan::key_1 + i)) {
          weapon_index = i;
          spread = weapon().base_spread;
        }
      }

      if (engine->is_key_clicked(fan::key_f)) {
        start_challenge();
      }
      if (engine->is_key_clicked(fan::key_escape) && challenge != challenge_e::idle) {
        end_challenge();
      }

      bool trigger = weapon().automatic ? engine->is_mouse_down(fan::mouse_left) : engine->is_mouse_clicked(fan::mouse_left);
      if (trigger && !fan::graphics::gui::want_io() && can_shoot) {
        fire();
      }
    }

    void draw_gun() {
      f32_t angle = std::atan2(aim_dir.y, aim_dir.x);
      fan::vec2 grip = aim_origin() + aim_dir * (cfg_gun_length * 0.5f - gun_kick * 6.f);
      fan::graphics::rectangle(fan::vec3(grip, depth_gun), fan::vec2(cfg_gun_length * 0.5f, 3.f), fan::color(0.25f, 0.25f, 0.28f)).set_angle(angle);
    }

    void draw_crosshair() {
      // gap opens with the current spread cone at the cursor's distance
      f32_t dist = (mouse_world - aim_origin()).length();
      f32_t gap = 6.f + std::tan(spread) * dist;
      f32_t len = 8.f;
      fan::color col = fan::color(1.f, 1.f, 1.f, 0.85f);
      for (fan::vec2 d : {fan::vec2(1, 0), fan::vec2(-1, 0), fan::vec2(0, 1), fan::vec2(0, -1)}) {
        fan::graphics::line(fan::vec3(mouse_world + d * gap, depth_crosshair), fan::vec3(mouse_world + d * (gap + len), depth_crosshair), col, 2.f);
      }

      if (hit_marker > 0.f) {
        f32_t a = hit_marker / cfg_hit_marker_life;
        fan::color mc = hit_marker_kill ? fan::color(1.f, 0.25f, 0.2f, a) : fan::color(1.f, 1.f, 1.f, a);
        for (fan::vec2 d : {fan::vec2(1, 1), fan::vec2(-1, 1), fan::vec2(1, -1), fan::vec2(-1, -1)}) {
          fan::graphics::line(fan::vec3(mouse_world + d * 5.f, depth_crosshair), fan::vec3(mouse_world + d * 11.f, depth_crosshair), mc, 2.f);
        }
      }
    }

    void count_shot() {
      ++shots_fired;
      if (challenge == challenge_e::running) {
        ++challenge_shots;
      }
    }

    void on_target_hit(std::size_t index, fan::vec2 hit_pos, fan::vec2 bullet_dir, int damage) {
      target_t& t = targets[index];
      ++shots_hit;
      if (challenge == challenge_e::running) {
        ++challenge_hits;
      }
      t.hp -= damage;
      t.flash = cfg_target_flash;
      t.squish = 1.f;
      t.vel += bullet_dir * cfg_target_knockback * (f32_t)damage;

      particles.spawn_spark(fan::vec3(hit_pos, depth_muzzle), fan::color(1.0, 0.8, 0.3), 6);
      hit_marker = cfg_hit_marker_life;
      hit_marker_kill = !t.alive();
      add_popup(index, hit_pos, damage, !t.alive());

      f32_t stop = weapon().hitstop;
      if (!t.alive()) {
        ++targets_destroyed;
        // challenge targets stay down until the run ends
        t.respawn = challenge == challenge_e::idle || challenge == challenge_e::done ? cfg_target_respawn : std::numeric_limits<f32_t>::max();
        particles.spawn_explosion(fan::vec3(t.pos, depth_muzzle), fan::color(1.0, 0.4, 0.1), 24, smoke_image);
        ic.shake(0.03f, 0.08f);
        stop = std::max(stop, cfg_kill_hitstop);
      }
      else {
        ic.shake(0.015f, 0.03f);
      }
      if (stop > 0.f && !is_hitstop) {
        hitstop_task = apply_hitstop(stop);
      }
    }

    void update_bullet_hits(f32_t dt) {
      registry.destroy_if<fan::ecs::c_pos, fan::ecs::c_vel, c_damage, fan::ecs::tag_bullet>([&](fan::ecs::c_pos& p, fan::ecs::c_vel& v, c_damage& dmg, fan::ecs::tag_bullet&) {
        // sweep the step the bullet is about to take, it moves further than a target is wide
        fan::vec2 next = p.v + v.v * dt;
        std::size_t best = targets.size();
        f32_t best_t = 2.f;
        for (std::size_t i = 0; i < targets.size(); ++i) {
          if (!targets[i].alive()) continue;
          f32_t hit_t = segment_vs_aabb(p.v, next, targets[i].pos, target_size);
          if (hit_t >= 0.f && hit_t < best_t) {
            best_t = hit_t;
            best = i;
          }
        }
        if (best != targets.size()) {
          on_target_hit(best, p.v + (next - p.v) * best_t, v.v.normalize(), dmg.v);
          return true;
        }

        bool hit_wall = false;
        tile_world.query_radius(p.v, 4.f, [&](std::uint32_t) { hit_wall = true; });
        if (hit_wall) {
          particles.spawn_spark(fan::vec3(p.v, depth_muzzle), fan::color(1.0, 0.6, 0.2), 3);
        }
        return hit_wall;
      });
    }

    void update_targets(f32_t dt) {
      for (auto& t : targets) {
        if (!t.hanging()) {
          t.home = t.base + t.axis * std::sin(world_time * t.speed + t.phase);
          // rail
          fan::graphics::line(fan::vec3(t.base - t.axis, depth_target - 2), fan::vec3(t.base + t.axis, depth_target - 2), fan::color(0.35f, 0.33f, 0.3f), 2.f);
        }

        if (!t.alive()) {
          t.respawn -= dt;
          if (t.respawn <= 0.f) {
            t = t.reset();
          }
          continue;
        }

        fan::vec2 accel = (t.home - t.pos) * cfg_target_spring - t.vel * cfg_target_damping;
        t.vel += accel * dt;
        t.pos += t.vel * dt;
        t.flash = std::max(0.f, t.flash - dt);
        t.squish = std::max(0.f, t.squish - dt * 8.f);

        if (t.hanging()) {
          fan::vec2 anchor = t.home - fan::vec2(0.f, 160.f);
          fan::graphics::line(fan::vec3(anchor, depth_target - 1), fan::vec3(t.pos - fan::vec2(0.f, target_size.y), depth_target - 1), fan::color(0.5f, 0.45f, 0.4f), 2.f);
        }

        fan::vec2 size = target_size * fan::vec2(1.f + 0.25f * t.squish, 1.f - 0.2f * t.squish);
        fan::color col = t.flash > 0.f ? fan::colors::white : t.hanging() ? fan::color(0.85f, 0.25f, 0.2f) : fan::color(0.9f, 0.55f, 0.15f);
        fan::graphics::rectangle(fan::vec3(t.pos, depth_target), size, col);
        fan::graphics::rectangle(fan::vec3(t.pos, depth_target + 1), size * fan::vec2(0.5f, 0.15f), fan::color(1.f, 0.9f, 0.8f));

        // hp bar
        f32_t frac = (f32_t)t.hp / cfg_target_hp;
        fan::vec2 bar_pos = t.pos - fan::vec2(0.f, target_size.y + 12.f);
        fan::vec2 bar_half = fan::vec2(target_size.x, 3.f);
        fan::graphics::rectangle(fan::vec3(bar_pos, depth_target + 2), bar_half, fan::color(0.1f, 0.1f, 0.1f, 0.8f));
        fan::graphics::rectangle(fan::vec3(bar_pos - fan::vec2(bar_half.x * (1.f - frac), 0.f), depth_target + 3), fan::vec2(bar_half.x * frac, bar_half.y), fan::color(0.3f, 1.f, 0.4f));
      }
    }

    void add_popup(std::size_t target, fan::vec2 pos, int amount, bool kill) {
      // rapid hits on the same target add up into one number
      for (auto& p : popups) {
        if (p.target == target && p.age < cfg_popup_merge_window && !p.kill) {
          p.amount += amount;
          p.kill = kill;
          p.age = 0;
          return;
        }
      }
      popups.push_back({
        .pos = pos,
        .vel = fan::vec2(fan::random::value(-60.f, 60.f), -260.f),
        .amount = amount,
        .kill = kill,
        .target = target
      });
    }

    void update_popups(f32_t dt) {
      std::erase_if(popups, [](const damage_popup_t& p) { return p.age >= cfg_popup_life; });
      for (auto& p : popups) {
        p.age += dt;
        p.pos += p.vel * dt;
        p.vel *= std::max(0.f, 1.f - dt * 5.f);

        f32_t a = 1.f - p.age / cfg_popup_life;
        fan::color col = p.kill ? fan::color(1.f, 0.35f, 0.25f, a) : fan::color(1.f, 0.95f, 0.6f, a);
        fan::vec2 screen = fan::graphics::world_to_screen(p.pos);
        std::string str = std::to_string(p.amount);
        f32_t size = p.kill ? 26.f : 20.f;
        fan::graphics::gui::text(str, {.color = fan::color(0, 0, 0, a), .font_size = size, .pos = screen + 2.f, .align = fan::graphics::align_e::center, .overlay = true});
        fan::graphics::gui::text(str, {.color = col, .font_size = size, .pos = screen, .align = fan::graphics::align_e::center, .overlay = true});
      }
    }

    void start_challenge() {
      for (auto& t : targets) {
        t = t.reset();
      }
      popups.clear();
      challenge = challenge_e::countdown;
      challenge_time = cfg_challenge_countdown;
    }

    void end_challenge() {
      challenge = challenge_e::idle;
      for (auto& t : targets) {
        if (!t.alive()) {
          t.respawn = cfg_target_respawn;
        }
      }
    }

    void update_challenge(f32_t real_dt) {
      switch (challenge) {
      case challenge_e::countdown:
        challenge_time -= real_dt;
        if (challenge_time <= 0.f) {
          challenge = challenge_e::running;
          challenge_time = 0.f;
          challenge_shots = 0;
          challenge_hits = 0;
        }
        break;
      case challenge_e::running: {
        challenge_time += real_dt;
        bool all_down = std::ranges::none_of(targets, &target_t::alive);
        if (all_down) {
          challenge = challenge_e::done;
          if (best_time == 0.f || challenge_time < best_time) {
            best_time = challenge_time;
          }
          for (auto& t : targets) {
            t.respawn = cfg_target_respawn;
          }
        }
        break;
      }
      default:
        break;
      }
    }

    void draw_hud() {
      using namespace fan::graphics::gui;
      fan::vec2 vs = ic.get_viewport_size();
      auto line = [&](const std::string& s, f32_t y, fan::color col, f32_t size = 20.f) {
        text(s, {.color = fan::color(0, 0, 0, col.a), .font_size = size, .pos = fan::vec2(vs.x * 0.5f, y) + 2.f, .align = fan::graphics::align_e::center, .overlay = true});
        text(s, {.color = col, .font_size = size, .pos = fan::vec2(vs.x * 0.5f, y), .align = fan::graphics::align_e::center, .overlay = true});
      };

      std::string weapons_line;
      for (std::size_t i = 0; i < weapons.size(); ++i) {
        weapons_line += (i == weapon_index ? "> " : "  ") + std::to_string(i + 1) + " " + weapons[i].name + (i == weapon_index ? " <  " : "    ");
      }
      line(weapons_line, 20.f, fan::color(1.f, 1.f, 1.f, 0.9f));

      int alive = (int)std::ranges::count_if(targets, &target_t::alive);
      auto acc = [](std::uint32_t hits, std::uint32_t shots) { return shots ? hits * 100 / shots : 0u; };
      switch (challenge) {
      case challenge_e::idle:
        line(best_time > 0.f ? std::format("F: start challenge   best {:.2f}s", best_time) : std::string("F: start challenge"), 50.f, fan::color(0.8f, 0.8f, 0.8f, 0.8f), 18.f);
        break;
      case challenge_e::countdown:
        line(std::format("{}", (int)std::ceil(challenge_time)), 80.f, fan::color(1.f, 0.85f, 0.3f), 48.f);
        break;
      case challenge_e::running:
        line(std::format("{:.2f}s   targets {}/{}   accuracy {}%   (Esc cancels)", challenge_time, targets.size() - alive, targets.size(), acc(challenge_hits, challenge_shots)), 50.f, fan::colors::white);
        break;
      case challenge_e::done:
        line(std::format("Cleared in {:.2f}s{}   accuracy {}%   F: retry", challenge_time, challenge_time <= best_time ? " (best!)" : std::format(" (best {:.2f}s)", best_time), acc(challenge_hits, challenge_shots)), 50.f, fan::color(0.4f, 1.f, 0.5f));
        break;
      }
    }

    void draw_gui() {
      if (gloco()->is_toggled(fan::key_t))
      if (auto h = fan::graphics::gui::hud_interactive{"##hud"}) {
        fan::graphics::gui::text("Shots: ", shots_fired, "  Hits: ", shots_hit, "  Accuracy: ", shots_fired ? (shots_hit * 100 / shots_fired) : 0, "%");
        fan::graphics::gui::text("Targets destroyed: ", targets_destroyed);
        fan::graphics::gui::text("Player Light");
        fan::graphics::gui::color_edit3("PL Color", (fan::vec3*)&pl_col);
        fan::graphics::gui::slider("PL Size", &pl_rad, 50.f, 1000.f);
        fan::graphics::gui::slider("PL Alpha", &pl_visual_a, 0.f, 1.f);
        fan::graphics::gui::slider("PL Shadow", &pl_shadow_a, 0.f, 1.f);
        fan::graphics::gui::slider("PL Softness", &pl_shadow_softness, 0.f, 200.f);
        fan::graphics::gui::text("Spawn Light");
        fan::graphics::gui::color_edit3("SP Color", (fan::vec3*)&sp_col);
        fan::graphics::gui::slider("SP Size", &sp_rad, 50.f, 2000.f);
        fan::graphics::gui::slider("SP Alpha", &sp_visual_a, 0.f, 1.f);
        fan::graphics::gui::slider("SP Shadow", &sp_shadow_a, 0.f, 1.f);
        fan::graphics::gui::slider("SP Softness", &sp_shadow_softness, 0.f, 200.f);
      }
    }

    fan::event::task_t start_shoot_cd(f32_t seconds) {
      can_shoot = false;
      co_await fan::event::timer_t(seconds * 1000.f);
      can_shoot = true;
    }

    fan::event::task_t apply_hitstop(f32_t seconds) {
      is_hitstop = true;
      co_await fan::event::timer_t(seconds * 1000.f);
      is_hitstop = false;
    }

    fan::event::task_t render_muzzle_flash(fan::vec2 pos) {
      auto t = fan::time::seconds_timer(cfg_muzzle_flash_life);
      while (!t) {
        f32_t muzzle_scale = std::sin(t.seconds() / cfg_muzzle_flash_life * fan::math::pi);
        fan::graphics::light(
          fan::vec3(pos, depth_muzzle),
          fan::vec2(400.f) * muzzle_scale,
          fan::color(1.0, 0.5, 0.05, 0.6) / 4.f
        );
        fan::graphics::circle(
          fan::vec3(pos, depth_muzzle),
          21.f * muzzle_scale,
          fan::color(1.0, 0.5, 0.05, 0.6) * 2.f
        );
        fan::graphics::sprite({
          .position = fan::vec3(pos, depth_muzzle),
          .size = cfg_muzzle_sprite_size * muzzle_scale,
          .color = fan::color(1.0, 0.1, 0.05, 0.6) * 2.f,
          .image = {"images/smoke.webp", fan::graphics::image_presets::pixel_art()}
        });
        co_await fan::graphics::co_next_frame();
      }
    }

    fan::graphics::tilemap_renderer_t::id_t map_id;
    std::vector<fan::physics::entity_t> collisions;
    fan::spatial::world_t<std::uint32_t> tile_world;
    fan::graphics::tilemap_renderer_t renderer;
    player_t player;
    fan::graphics::interactive_camera_t ic;
    fan::graphics::gpu_particle_system_t<> particles;
    fan::graphics::image_t smoke_image{"images/smoke.webp"};
    registry_t registry;
    std::deque<casing_t> casings;
    std::vector<target_t> targets;
    std::vector<damage_popup_t> popups;
    std::uint32_t shots_fired = 0;
    std::uint32_t shots_hit = 0;
    std::uint32_t targets_destroyed = 0;

    std::size_t weapon_index = 0;
    fan::vec2 aim_dir = fan::vec2(1.f, 0.f);
    fan::vec2 mouse_world = 0;
    f32_t spread = weapons[0].base_spread;
    f32_t gun_kick = 0;
    f32_t hit_marker = 0;
    bool hit_marker_kill = false;
    f32_t world_time = 0;

    challenge_e challenge = challenge_e::idle;
    f32_t challenge_time = 0;
    f32_t best_time = 0;
    std::uint32_t challenge_shots = 0;
    std::uint32_t challenge_hits = 0;

    f32_t pl_rad = 964.372f;
    fan::color pl_col = fan::color(1.f, 1.f, 0.8f, 1.f);
    f32_t pl_shadow_a = 0.723f;
    f32_t pl_shadow_softness = 5.264f;
    f32_t pl_visual_a = 0.200f;
    f32_t sp_rad = 500.f;
    fan::color sp_col = fan::color(1.f, 0.7f, 0.3f, 1.f);
    f32_t sp_shadow_a = 0.2f;
    f32_t sp_shadow_softness = 30.f;
    f32_t sp_visual_a = 0.06f;

    bool can_shoot = true;
    bool is_hitstop = false;
    fan::event::task_t shoot_task;
    fan::event::task_t hitstop_task;
    fan::event::task_t muzzle_task;
  };

  pile_t() {
    set_settings({
      .clear_color = fan::colors::black,
      .ambient_color = fan::colors::black,
    });
    update_physics(true);
    stage_open<example_stage_t>();
  }
} pile;

int main() {
  pile.loop([&] {

  });
}
