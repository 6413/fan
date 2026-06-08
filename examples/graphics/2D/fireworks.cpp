#include <cmath>
#include <algorithm>

import fan;

using namespace fan::graphics;

static constexpr int z_particles = 0xfff;
static constexpr int z_trails = 0xffe;
static constexpr int max_sparks = 2048;
static constexpr int max_rockets = 32;
static constexpr int max_trails = max_sparks * 6 + max_rockets * 16;

struct spark_sim_t {
  fan::vec2 pos, vel;
  f32_t life, max_life;
  fan::color col;
  fan::time::interval_t trail_interval {0.03f};
};

struct rocket_sim_t {
  fan::vec2 pos, vel;
  fan::color col;
  fan::time::interval_t trail_interval {0.018f};
};

struct fireworks_t {
  void explode(fan::vec2 pos, fan::color base_col) {
    fan::graphics::emit_radial(spark_pool, pos, sparks_per_burst_count, 120.f, 480.f,
      [&](auto& pool, fan::vec2 pos, fan::vec2 vel) {
      fan::color col = fan::random::color_near(base_col, 30.f);
      pool.spawn(
        [&](shape_t& shape) {
        shape.set_position(fan::vec3(pos, z_particles));
        shape.set_radius(spark_radius);
        shape.set_color(col);
      },
        [&](spark_sim_t& s) {
        s.pos = pos;
        s.vel = vel;
        s.life = s.max_life = fan::random::value(spark_lifetime_min, spark_lifetime_max);
        s.col = col;
        s.trail_interval.reset();
      }
      );
    }
    );
  }
  void launch(fan::vec2 screen_pos) {
    fan::vec2 sz = engine.viewport_get_size();
    fan::vec2 start = {screen_pos.x, sz.y - 10.f};
    fan::vec2 vel = fan::math::launch_to_target(start, screen_pos, gravity);
    fan::color col = fan::random::bright_color(); col.a = 1.f;
    rocket_pool.spawn(
      [&](shape_t& shape) {
      shape.set_position(fan::vec3(start, z_particles));
      shape.set_radius(rocket_radius);
      shape.set_color(col);
    },
      [&](rocket_sim_t& s) {
      s.pos = start;
      s.vel = vel;
      s.col = col;
      s.trail_interval.reset();
    }
    );
  }
  void update(f32_t dt) {
    trail_pool.update_and_cull(dt, fan::graphics::trail_particle_updater_t {gravity * 0.15f, z_trails});
    rocket_pool.update_and_cull(dt, [&](auto& r, f32_t dt) {
      r.vel.y += gravity * dt;
      r.pos += r.vel * dt;
      r.shape.set_position(fan::vec3(r.pos, z_particles));
      if (r.trail_interval.tick(dt)) {
        fan::graphics::spawn_trail(trail_pool, r.pos, r.vel, r.col, rocket_radius * 1.2f, 0.28f, z_trails);
      }
      if (r.vel.y >= 0.f) {
        r.shape.set_radius(0.f);
        explode(r.pos, r.col);
        return false;
      }
      return true;
    });
    spark_pool.update_and_cull(dt, [&](auto& s, f32_t dt) {
      s.life -= dt;
      if (s.life <= 0.f) {
        return false;
      }
      s.vel.y += gravity * dt;
      s.pos += s.vel * dt;
      f32_t t = std::max(0.f, s.life / s.max_life);
      s.col.a = t;
      s.shape.set_position(fan::vec3(s.pos, z_particles));
      s.shape.set_radius(spark_radius * t);
      s.shape.set_color(s.col);
      if (s.trail_interval.tick(dt)) {
        fan::graphics::spawn_trail(trail_pool, s.pos, s.vel, s.col, spark_radius * t * 1.4f, 0.18f, z_trails);
      }
      return true;
    });
  }
  void draw_gui() {
    gui::begin("Fireworks");
    gui::drag("Gravity", &gravity);
    gui::drag("Sparks per burst", &sparks_per_burst_count);
    gui::drag("Spark radius", &spark_radius);
    gui::drag("Rocket radius", &rocket_radius);
    gui::drag("Spark life min", &spark_lifetime_min);
    gui::drag("Spark life max", &spark_lifetime_max);
    gui::end();
  }
  engine_t engine {{.renderer = renderer_t::opengl}};
  particle_pool_t<circle_t, spark_sim_t, max_sparks>  spark_pool;
  particle_pool_t<circle_t, rocket_sim_t, max_rockets> rocket_pool;
  particle_pool_t<circle_t, fan::graphics::trail_particle_t, max_trails> trail_pool;
  f32_t gravity = 300.f;
  f32_t spark_radius = 5.f;
  f32_t rocket_radius = 8.f;
  f32_t spark_lifetime_min = 0.8f;
  f32_t spark_lifetime_max = 2.0f;
  int sparks_per_burst_count = 120;
};

int main() {
  fireworks_t fw;
  fw.engine.loop([&] {
    if (fan::window::is_mouse_clicked()) {
      fw.launch(fan::window::get_mouse_position());
    }
    fw.update(fw.engine.delta_time);
    fw.draw_gui();
  });
}