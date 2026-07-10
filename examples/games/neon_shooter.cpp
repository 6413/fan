#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

import fan;

using namespace fan::graphics;

enum class game_state_e { playing, game_over };

struct projectile_t {
  fan::vec2 pos, vel;
  f32_t life = 2.0f;
  fan::time::interval_t trail_interval {0.015f};
  fan::graphics::shape_t shape;
};

struct enemy_t {
  fan::vec2 pos;
  f32_t hp = 1.0f;
  fan::graphics::shape_t shape;
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
  fan::graphics::shape_t player_shape;
  
  std::vector<projectile_t> projectiles;
  std::vector<enemy_t> enemies;
  
  static constexpr int max_trails = 4096;
  gpu_particle_system_t<> gpu_sparks;
  particle_pool_t<circle_t, trail_particle_t, max_trails> trail_pool;
  fan::graphics::shape_t bg_grid;
  
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
    
    fan::graphics::shapes::circle_t::properties_t p_props;
    p_props.position = fan::vec3(player_pos, 0.f);
    p_props.radius = player_radius;
    p_props.color = fan::colors::cyan;
    player_shape = p_props;
    
    fan::graphics::shapes::grid_t::properties_t bg_p;
    bg_p.position = fan::vec3(0.f, 0.f, -1.f);
    bg_p.size = fan::vec2(engine.viewport_get_size().max() * 3.f);
    bg_p.grid_size = fan::vec2(100.f);
    bg_p.color = fan::color(0.05f, 0.1f, 0.2f, 0.5f);
    bg_grid = bg_p;
    
    fan::graphics::camera_look_at(player_pos, 0.f);
    camera_shake = 0.f;
    enemy_spawn_timer = 1.f;
  }
  
  void explode(fan::vec2 pos, fan::color base_col, int count) {
    gpu_sparks.spawn([&](auto& p) {
      p.loop = false;
      p.position = fan::vec3(pos, 0);
      p.count = count;
      p.alive_time = fan::random::value(0.4f, 1.2f);
      p.respawn_time = -p.alive_time;
      
      p.start_velocity = fan::vec2(150.f, 800.f); 
      p.end_velocity = fan::vec2(5.f, 20.f); 
      p.expansion_power = 1.0f;
      
      p.start_size = fan::vec2(16.f);
      p.end_size = fan::vec2(0.f);

      p.begin_color = base_col;
      p.end_color = base_col.set_alpha(0.0f);
      p.color_random_range = fan::vec4(0.2f);

      p.shape = fan::graphics::shapes::particles_t::shapes_e::circle;
      
      p.start_spread = fan::vec2(0, 0);
      p.end_spread = fan::vec2(0, 0);
      
      p.angle = fan::vec3(0,0,0);
      p.begin_angle = 0;
      p.end_angle = 6.283185f; 
      
      p.start_angle_velocity = fan::vec3(0, 0, 0);
    });
  }

  void draw_background(fan::vec2 cam_pos) {
    fan::vec2 offset = cam_pos * -0.2f; 
    bg_grid.set_position(fan::vec3(cam_pos.x + std::fmod(offset.x, 100.f), cam_pos.y + std::fmod(offset.y, 100.f), -1.f));
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
      player_shape.set_position(fan::vec3(player_pos, 0.f));
      
      if (player_trail_interval.tick(dt)) {
        fan::graphics::spawn_trail(trail_pool, player_pos, fan::vec2(0.f), fan::colors::cyan, player_radius * 0.9f, 0.35f, 0.f);
      }
      
      // Shooting
      if (fan::window::is_mouse_clicked(fan::mouse_left)) {
        fan::vec2 mouse_pos = engine.get_mouse_position(engine.orthographic_render_view);
        fan::vec2 diff = mouse_pos - player_pos;
        fan::vec2 dir = diff / std::max((f32_t)diff.length(), 0.001f);
        fan::vec2 vel = dir * 1400.f;
        
        fan::graphics::shapes::circle_t::properties_t p;
        p.position = fan::vec3(player_pos, 0.f);
        p.radius = 8.f;
        p.color = fan::colors::yellow;
        
        projectiles.push_back({player_pos, vel, 2.0f, fan::time::interval_t{0.015f}, p});
        
        camera_shake = std::min(camera_shake + 4.f, 20.f);
      }
      
      // Enemy Spawning
      enemy_spawn_timer -= dt;
      if (enemy_spawn_timer <= 0.f) {
        enemy_spawn_timer = fan::random::value(0.4f, 1.2f) / difficulty_multiplier;
        fan::vec2 spawn_pos = fan::random::border_pos(view_size, 50.f);
        
        fan::graphics::shapes::rectangle_t::properties_t p;
        p.position = fan::vec3(spawn_pos, 0.f);
        p.size = fan::vec2(18.f);
        p.color = fan::colors::magenta;
        
        enemies.push_back({spawn_pos, 1.0f, p});
      }
    }
    
    // Projectiles
    for (int i = 0; i < projectiles.size(); ) {
      auto& p = projectiles[i];
      p.pos += p.vel * dt;
      p.life -= dt;
      p.shape.set_position(fan::vec3(p.pos, 0.f));
      
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
      e.shape.set_position(fan::vec3(e.pos, 0.f));
      
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
    
    trail_pool.update_and_cull(dt, fan::graphics::trail_particle_updater_t {0.f, 0});
    gpu_sparks.update(dt);
    
    if (state == game_state_e::game_over) {
      player_shape.set_color(fan::colors::transparent);
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