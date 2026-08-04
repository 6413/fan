#include <cmath>
#include <algorithm>

import fan;

using namespace fan::graphics;

static constexpr int z_particles = 0xfff;
static constexpr int z_trails = 0xffe;
static constexpr int max_sparks = 2048;
static constexpr int max_rockets = 32;
static constexpr int max_trails = max_sparks * 6 + max_rockets * 16;
static constexpr f32_t spark_trail_interval = 0.03f;
static constexpr f32_t spark_trail_lifetime = 0.18f;

struct rocket_sim_t {
  fan::vec2 pos, vel;
  fan::color col;
  fan::graphics::polygon_t visual;
  fan::time::interval_t trail_interval {0.018f};
};

struct fireworks_t {
  void explode(fan::vec2 pos, fan::color base_col) {
    int total = std::max(0, sparks_per_burst_count);
    if (!total) {
      return;
    }
    f32_t lifetime_min = std::max(0.0001f, std::min(spark_lifetime_min, spark_lifetime_max));
    f32_t lifetime_max = std::max(lifetime_min, std::max(spark_lifetime_min, spark_lifetime_max));
    f32_t lifetime_base = (lifetime_min + lifetime_max) * 0.5f;
    f32_t lifetime_random = std::clamp(
      (lifetime_max - lifetime_min) / std::max(0.0001f, lifetime_max + lifetime_min), 0.f, 0.99f
    );
    auto spawn_main = [&] {
      explosion_particles.spawn([&, total, lifetime_base, lifetime_random](auto& p) {
        p.loop = false;
        p.position = fan::vec3(pos, z_particles);
        p.count = total;
        p.alive_time = lifetime_base;
        p.respawn_time = -p.alive_time;
        p.start_velocity = fan::vec2(300.f, 0.f);
        p.end_velocity = fan::vec2(300.f, 0.f);
        p.gravity = fan::vec2(0.f, gravity);
        p.velocity_random = 0.6f;
        p.lifetime_random = lifetime_random;
        p.expansion_power = 1.0f;
        p.start_size = fan::vec2(spark_radius);
        p.end_size = fan::vec2(0.f);
        p.begin_color = base_col;
        p.end_color = base_col.set_alpha(0.f);
        p.color_random_range = fan::vec4(30.f, 15.f, 15.f, 0.f);
        p.shape = fan::graphics::shapes::particles_t::shapes_e::circle;
        p.start_spread = fan::vec2(0.f);
        p.end_spread = fan::vec2(0.f);
        p.begin_angle = 0.f;
        p.end_angle = fan::math::pi * 2.f;
        p.start_angle_velocity = fan::vec3(0.f);
        p.end_angle_velocity = fan::vec3(0.f);
        p.affected_by_lighting = false;
        p.image = particle_image;
      });
    };

    int trail_samples = std::max(1, (int)std::ceil(lifetime_max / spark_trail_interval));
    explosion_particles.spawn([&, total, lifetime_base, lifetime_random, trail_samples](auto& p) {
      p.loop = false;
      p.position = fan::vec3(pos, z_trails);
      p.count = total * trail_samples;
      p.alive_time = lifetime_base;
      p.respawn_time = -p.alive_time;
      p.start_velocity = fan::vec2(300.f, 0.f);
      p.end_velocity = fan::vec2(300.f, 0.f);
      p.gravity = fan::vec2(0.f, gravity);
      p.velocity_random = 0.6f;
      p.lifetime_random = lifetime_random;
      p.expansion_power = 1.0f;
      p.start_size = fan::vec2(spark_radius);
      p.end_size = fan::vec2(0.f);
      p.begin_color = base_col;
      p.end_color = base_col.set_alpha(0.f);
      p.color_random_range = fan::vec4(30.f, 15.f, 15.f, 0.f);
      p.nested_trail = fan::vec4(trail_samples, spark_trail_interval, spark_trail_lifetime, gravity * 0.15f);
      p.shape = fan::graphics::shapes::particles_t::shapes_e::circle;
      p.start_spread = fan::vec2(0.f);
      p.end_spread = fan::vec2(0.f);
      p.begin_angle = 0.f;
      p.end_angle = fan::math::pi * 2.f;
      p.start_angle_velocity = fan::vec3(0.f);
      p.end_angle_velocity = fan::vec3(0.f);
      p.affected_by_lighting = false;
      p.image = particle_image;
    });
    spawn_main();
  }
  void launch(fan::vec2 screen_pos) {
    fan::vec2 sz = engine.viewport_get_size();
    fan::vec2 start = {screen_pos.x, sz.y - 10.f};
    fan::vec2 vel = fan::math::launch_to_target(start, screen_pos, gravity);
    fan::color col = fan::random::bright_color() / 5.f; col.a = 1.f;
    rocket_pool.spawn(
      [&](shape_t& shape) {
      shape.set_visible(false);
      shape.set_position(fan::vec3(start, z_particles));
    },
      [&](rocket_sim_t& s) {
      s.pos = start;
      s.vel = vel;
      s.col = col;
      f32_t length = rocket_radius * 3.f;
      f32_t width = rocket_radius * 1.2f;
      s.visual = fan::graphics::polygon_t {{
        .position = fan::vec3(start, z_particles),
        .vertices = {
          {.position = fan::vec3(0, -length, 0), .color = col},
          {.position = fan::vec3(-width, length * 0.5f, 0), .color = col},
          {.position = fan::vec3(width, length * 0.5f, 0), .color = col},
        },
        .angle = fan::vec3(0, 0, std::atan2(vel.y, vel.x) + fan::math::pi / 2.f),
        .draw_mode = fan::graphics::primitive_topology_t::triangles,
        .enable_culling = false
      }};
      s.trail_interval.reset();
    }
    );
  }
  void update(f32_t dt) {
    explosion_particles.update(dt);
    trail_pool.update_and_cull(dt, fan::graphics::trail_particle_updater_t {gravity * 0.15f, z_trails});
    rocket_pool.update_and_cull(dt, [&](auto& r, f32_t dt) {
      r.vel.y += gravity * dt;
      r.pos += r.vel * dt;
      r.shape.set_position(fan::vec3(r.pos, z_particles));
      r.visual.set_position(fan::vec3(r.pos, z_particles));
      r.visual.set_angle(std::atan2(r.vel.y, r.vel.x) + fan::math::pi / 2.f);
      if (r.trail_interval.tick(dt)) {
        fan::graphics::spawn_trail(trail_pool, r.pos, r.vel, r.col, rocket_radius * 1.2f, 0.28f, z_trails);
      }
      if (r.vel.y >= 0.f) {
        r.shape.set_radius(0.f);
        r.visual.set_visible(false);
        explode(r.pos, r.col);
        return false;
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
  engine_t engine;
  fan::graphics::image_t particle_image{"images/circle.png"};
  fan::graphics::gpu_particle_system_t<> explosion_particles;
  particle_pool_t<circle_t, rocket_sim_t, max_rockets> rocket_pool;
  particle_pool_t<circle_t, fan::graphics::trail_particle_t, max_trails> trail_pool;
  f32_t gravity = 300.f;
  f32_t spark_radius = 5.f;
  f32_t rocket_radius = 8.f;
  f32_t spark_lifetime_min = 0.8f;
  f32_t spark_lifetime_max = 5.0f;
  int sparks_per_burst_count = 1200;
};

int main() {
  fireworks_t fw;
  sprite_t bg(fw.engine.whs(), fw.engine.whs(), fan::colors::black);
  fw.engine.loop([&] {
    if (fan::window::is_mouse_clicked()) {
      fw.launch(fan::window::get_mouse_position());
    }
    fw.update(fw.engine.get_delta_time());
    fw.draw_gui();
  });
}
