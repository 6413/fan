#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

import fan;

using namespace fan::graphics;

enum class game_state_e { playing, game_over };

struct spark_sim_t {
  fan::vec2 pos, vel;
  f32_t life, max_life;
  fan::color col;
};

struct projectile_t {
  fan::vec2 pos, vel;
  f32_t life = 2.0f;
  fan::time::interval_t trail_interval {0.015f};
};

struct enemy_t {
  fan::vec2 pos;
  f32_t hp = 1.0f;
};

struct neon_shooter_t {
  engine_t engine {{.post_process_mode = fan::graphics::post_process_mode_e::bloom}};
  
  game_state_e state = game_state_e::playing;
  int score = 0;
  f32_t difficulty_multiplier = 1.0f;

  fan::vec2 player_pos;
  f32_t player_radius = 20.f;
  f32_t speed = 500.f;
  fan::time::interval_t player_trail_interval {0.02f};
  
  std::vector<projectile_t> projectiles;
  std::vector<enemy_t> enemies;
  
  static constexpr int max_sparks = 4096;
  static constexpr int max_trails = 4096;
  particle_pool_t<circle_t, spark_sim_t, max_sparks> spark_pool;
  particle_pool_t<circle_t, trail_particle_t, max_trails> trail_pool;
  
  f32_t camera_shake = 0.f;
  f32_t enemy_spawn_timer = 0.f;
  
  neon_shooter_t() {
    *gloco()->get_bloom_strength_ptr() = 0.08f;
    restart();
  }

  void restart() {
    state = game_state_e::playing;
    score = 0;
    difficulty_multiplier = 1.0f;
    projectiles.clear();
    enemies.clear();
    player_pos = engine.viewport_get_size() / 2.f;
    fan::graphics::camera_look_at(player_pos, 0.f);
    camera_shake = 0.f;
    enemy_spawn_timer = 1.f;
  }
  
  void explode(fan::vec2 pos, fan::color base_col, int count) {
    fan::graphics::emit_radial(spark_pool, pos, count, 150.f, 800.f,
      [&](auto& pool, fan::vec2 pos, fan::vec2 vel) {
      fan::color col = fan::random::color_near(base_col, 50.f);
      pool.spawn(
        [&](shape_t& shape) {
          shape.set_position(fan::vec3(pos, 0.f));
          shape.set_radius(4.f);
          shape.set_color(col);
        },
        [&](spark_sim_t& s) {
          s.pos = pos;
          s.vel = vel;
          s.life = s.max_life = fan::random::value(0.4f, 1.2f);
          s.col = col;
        }
      );
    });
  }

  void draw_background(fan::vec2 cam_pos) {
    fan::vec2 view_size = engine.viewport_get_size();
    fan::vec2 offset = cam_pos * -0.2f; 
    
    f32_t grid_size = 100.f;
    fan::vec2 top_left = cam_pos - view_size / 2.f;
    fan::vec2 bottom_right = cam_pos + view_size / 2.f;
    
    f32_t start_x = std::floor((top_left.x - offset.x) / grid_size) * grid_size;
    for (f32_t x = start_x; x < bottom_right.x - offset.x + grid_size; x += grid_size) {
      fan::graphics::line(
        fan::vec3(x + offset.x, top_left.y, -1.f),
        fan::vec3(x + offset.x, bottom_right.y, -1.f),
        fan::color(0.05f, 0.1f, 0.2f, 0.5f), 2.f
      );
    }
    
    f32_t start_y = std::floor((top_left.y - offset.y) / grid_size) * grid_size;
    for (f32_t y = start_y; y < bottom_right.y - offset.y + grid_size; y += grid_size) {
      fan::graphics::line(
        fan::vec3(top_left.x, y + offset.y, -1.f),
        fan::vec3(bottom_right.x, y + offset.y, -1.f),
        fan::color(0.05f, 0.1f, 0.2f, 0.5f), 2.f
      );
    }
  }

  void draw_gui() {
    using namespace fan::graphics::gui;
    if (auto h = hud_interactive{"##game_ui"}) {
      fan::vec2 view_size = engine.viewport_get_size();

      push_font(get_font(36, font::bold));
      set_cursor_screen_pos(fan::vec2(view_size.x / 2.f - 80.f, 30.f));
      text(fan::colors::white, std::string("SCORE: ") + std::to_string(score));
      pop_font();
      
      if (state == game_state_e::game_over) {
        push_font(get_font(72, font::bold));
        set_cursor_screen_pos(fan::vec2(view_size.x / 2.f - 200.f, view_size.y / 2.f - 150.f));
        text(fan::colors::red, "GAME OVER");
        pop_font();
        
        set_cursor_screen_pos(fan::vec2(view_size.x / 2.f - 100.f, view_size.y / 2.f + 50.f));
        if (button("Restart", fan::vec2(200.f, 60.f))) {
          restart();
        }
      }
    }
  }

  void update(f32_t dt) {
    static fan::vec2 current_cam_pos = player_pos;
    fan::vec2 view_size = engine.viewport_get_size();
    
    if (state == game_state_e::playing) {
      difficulty_multiplier += dt * 0.02f;
      
      // Player Movement
      player_pos += engine.get_input_vector(speed) * dt;
      player_pos.x = std::clamp(player_pos.x, 0.f, (f32_t)view_size.x);
      player_pos.y = std::clamp(player_pos.y, 0.f, (f32_t)view_size.y);
      
      if (player_trail_interval.tick(dt)) {
        fan::graphics::spawn_trail(trail_pool, player_pos, fan::vec2(0.f), fan::colors::cyan, player_radius * 0.9f, 0.35f, 0.f);
      }
      
      // Shooting
      if (fan::window::is_mouse_clicked(fan::mouse_left)) {
        fan::vec2 mouse_pos = engine.get_mouse_position(engine.orthographic_render_view);
        fan::vec2 diff = mouse_pos - player_pos;
        fan::vec2 dir = diff / std::max((f32_t)diff.length(), 0.001f);
        fan::vec2 vel = dir * 1400.f;
        projectiles.push_back({player_pos, vel});
        
        camera_shake = std::min(camera_shake + 4.f, 20.f);
      }
      
      // Enemy Spawning
      enemy_spawn_timer -= dt;
      if (enemy_spawn_timer <= 0.f) {
        enemy_spawn_timer = fan::random::value(0.4f, 1.2f) / difficulty_multiplier;
        fan::vec2 spawn_pos = fan::random::border_pos(view_size, 50.f);
        enemies.push_back({spawn_pos, 1.0f});
      }
    }
    
    // Projectiles
    for (int i = 0; i < projectiles.size(); ) {
      auto& p = projectiles[i];
      p.pos += p.vel * dt;
      p.life -= dt;
      
      fan::graphics::circle(p.pos, 8.f, fan::colors::yellow);
      
      if (p.trail_interval.tick(dt)) {
        fan::graphics::spawn_trail(trail_pool, p.pos, p.vel * 0.1f, fan::colors::red, 10.f, 0.25f, 0.f);
      }
      
      if (p.life <= 0.f) {
        projectiles.erase(projectiles.begin() + i);
      } else {
        ++i;
      }
    }
    
    // Enemies & Collisions
    for (int i = 0; i < enemies.size(); ) {
      auto& e = enemies[i];
      
      if (state == game_state_e::playing) {
        fan::vec2 diff = player_pos - e.pos;
        fan::vec2 dir = diff / std::max((f32_t)diff.length(), 0.001f);
        e.pos += dir * 150.f * difficulty_multiplier * dt;
      }
      
      fan::graphics::rectangle(fan::vec3(e.pos, 0.f), fan::vec2(18.f), fan::colors::magenta);
      
      bool hit = false;
      for (int j = 0; j < projectiles.size(); ++j) {
        if ((e.pos - projectiles[j].pos).length() < 30.f) {
          hit = true;
          projectiles.erase(projectiles.begin() + j);
          break;
        }
      }
      
      if (hit) {
        explode(e.pos, fan::colors::magenta, 80);
        camera_shake = std::min(camera_shake + 15.f, 40.f);
        enemies.erase(enemies.begin() + i);
        if (state == game_state_e::playing) {
          score += 100;
        }
      } else if (state == game_state_e::playing && (e.pos - player_pos).length() < player_radius + 18.f) {
        explode(player_pos, fan::colors::cyan, 250);
        camera_shake = std::min(camera_shake + 60.f, 100.f);
        state = game_state_e::game_over;
      } else {
        ++i;
      }
    }
    
    // Particles
    trail_pool.update_and_cull(dt, fan::graphics::trail_particle_updater_t {0.f, 0});
    
    spark_pool.update_and_cull(dt, [&](auto& s, f32_t dt) {
      s.life -= dt;
      if (s.life <= 0.f) return false;
      s.vel *= std::pow(0.05f, dt);
      s.pos += s.vel * dt;
      f32_t t = std::max(0.f, s.life / s.max_life);
      s.col.a = t;
      s.shape.set_position(fan::vec3(s.pos, 0.f));
      s.shape.set_radius(5.f * t);
      s.shape.set_color(s.col);
      return true;
    });
    
    if (state == game_state_e::playing) {
      fan::graphics::circle(player_pos, player_radius, fan::colors::cyan);
    }
    
    // Camera Tracking
    if (state == game_state_e::playing) {
      current_cam_pos += (player_pos - current_cam_pos) * 15.f * dt;
    }
    
    fan::vec2 shake_offset(0.f, 0.f);
    if (camera_shake > 0.f) {
      shake_offset = fan::vec2(
        fan::random::value(-camera_shake, camera_shake),
        fan::random::value(-camera_shake, camera_shake)
      );
      camera_shake = std::max(0.f, camera_shake - 60.f * dt);
    }
    
    fan::graphics::camera_look_at(current_cam_pos + shake_offset, 0.f);
    
    draw_background(current_cam_pos + shake_offset);
    draw_gui();
  }
};

int main() {
  neon_shooter_t game;
  game.engine.loop([&] {
    game.update(game.engine.get_delta_time());
  });
}