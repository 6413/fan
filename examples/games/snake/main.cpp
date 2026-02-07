#include <string>
#include <vector>

import fan;

using namespace fan::graphics;

static constexpr fan::vec2 tile_count = {16, 16};

struct snake_t {
  void init(fan::vec2 origin, fan::vec2 size, fan::vec2 tile) {
    grid_origin = origin;
    fan::vec2 start = origin + fan::vec2(tile.x * 3, size.y * 0.5).snap_to_grid(tile) + tile * 0.5f;
    
    head = circle_t(fan::vec3(start, 0xf + 1), tile.x * 0.5f, fan::color::from_rgb(0x8AA90E));
    dir = {1, 0};
    next_dir = {0, 0};
    
    trail.clear();
    fan::vec2 base = start - dir * spacing;
    for (int i = length / spacing; i--; ) {
      trail.push_back(base - dir * (i * spacing));
    }
    
    rebuild(tile.x);
  }
  void rebuild(f32_t thickness) {
    if (trail.size() < 2) return;
    body.set({
      .points = std::span<const fan::vec2>(trail.data(), trail.size()),
      .thickness = thickness,
      .color = fan::color::from_rgb(0x8AA90E),
      .depth = 0xf,
      .join = polyline_join_t::round,
      .cap_start = polyline_cap_t::round,
      .cap_end = polyline_cap_t::round
    });
  }
  void update(f32_t dt, fan::vec2 input, fan::vec2 tile) {
    if (input.length() > 0) {
      input = (input.abs().x > input.abs().y) ? fan::vec2(input.sign().x, 0) : fan::vec2(0, input.sign().y);
      if (input != -dir && !(next_dir.length() > 0 && input == -next_dir)) {
        next_dir = input;
      }
    }

    f32_t dist = speed * dt;
    while (dist > 0) {
      fan::vec2 pos = head.get_position();
      fan::vec2 center = grid_origin + ((pos - grid_origin) / tile).floor() * tile + tile * 0.5f;
      fan::vec2 to_center = center - pos;
      f32_t forward = to_center.dot(dir);

      if (forward > 0) {
        f32_t d = to_center.length();
        if (dist >= d) {
          head.set_position(center);
          dist -= d;
          if (next_dir) {
            dir = next_dir;
            next_dir = {0, 0};
            fan::vec2 p = center;
            if (dir.x != 0) p.y = center.y;
            else p.x = center.x;
            head.set_position(p);
          }
          continue;
        }
      }
      head.set_position(pos + dir * dist);
      dist = 0;
    }

    fan::vec2 pos = head.get_position();
    fan::vec2 center = grid_origin + ((pos - grid_origin) / tile).floor() * tile + tile * 0.5f;
    if (dir.x != 0) pos.y = center.y;
    else pos.x = center.x;
    head.set_position(pos);

    fan::vec2 h = head.get_position();
    if (trail.empty()) {
      trail.push_back(h);
    }
    else {
      f32_t d = (h - trail.back()).length();
      if (d > spacing)
        trail.push_back(h);
    }

    f32_t total = 0;
    for (int i = 0; i + 1 < (int)trail.size(); ++i) {
      total += (trail[i + 1] - trail[i]).length();
    }

    while (trail.size() > 1 && total > length) {
      f32_t seg = (trail[1] - trail[0]).length();
      f32_t excess = total - length;
      if (excess >= seg) {
        total -= seg;
        trail.erase(trail.begin());
      }
      else {
        trail[0] += (trail[1] - trail[0]) / seg * excess;
        total -= excess;
        break;
      }
    }

    rebuild(tile.x);
  }
  bool collided(fan::vec2 world_size) {
    fan::vec2 h = head.get_position();
    f32_t r = head.get_radius();
    fan::vec2 mn = grid_origin + fan::vec2(r, r);
    fan::vec2 mx = grid_origin + world_size - fan::vec2(r, r);
    
    if (h.x < mn.x || h.x > mx.x || h.y < mn.y || h.y > mx.y) {
      return true;
    }

    int skip = (int)(length / spacing) / 3;
    for (int i = 0; i + skip < (int)trail.size(); ++i) {
      if ((h - trail[i]).length() < r * 0.8f) {
        return true;
      }
    }
    
    return false;
  }
  bool overlaps(fan::vec2 pos, f32_t radius) {
    for (auto& p : trail) {
      if ((pos - p).length() < radius) {
        return true;
      }
    }
    return false;
  }

  fan::vec2 grid_origin;
  fan::vec2 dir {1, 0};
  fan::vec2 next_dir {0, 0};
  f32_t speed = 350.f;
  f32_t spacing = 4.f;
  f32_t length = 1024.f;
  circle_t head;
  polyline_t body;
  std::vector<fan::vec2> trail;
};

struct food_t {
  void spawn(fan::vec2 origin, fan::vec2 tile, fan::vec2 count, snake_t& snake) {
    std::vector<fan::vec2> available;
    
    for (int y = 0; y < (int)count.y; ++y) {
      for (int x = 0; x < (int)count.x; ++x) {
        fan::vec2 test_pos = origin + fan::vec2(x, y) * tile + tile * 0.5f;
        if (!snake.overlaps(test_pos, tile.x * 0.5f)) {
          available.push_back(fan::vec2(x, y));
        }
      }
    }
    
    if (available.empty()) {
      idx = {0, 0};
      pos = origin + tile * 0.5f;
    }
    else {
      idx = available[fan::random::value(0, (int)available.size() - 1)];
      pos = origin + idx * tile + tile * 0.5f;
    }
    
    sprite = circle_t(fan::vec3(pos, 2), tile.x * 0.3f, fan::colors::red);
  }

  fan::vec2 idx;
  fan::vec2 pos;
  circle_t sprite;
};

struct app_t {
  app_t() {
    engine.window.set_size(1024);
    on_resize(engine.window.get_size());
    
    bg = {{
      .position = fan::vec3(origin, 0),
      .size = size / 2.f,
      .image = engine.create_transparent_texture()
    }};
    
    fan::vec2 grid = origin - size / 2.f;
    snake.init(grid, size, tile);
    food.spawn(grid, tile, tile_count, snake);
    
    resize_handle = engine.window.on_resize([this](const engine_t::resize_data_t& rdata) {
      on_resize(rdata.size);
      
      fan::vec2 grid = origin - size / 2.f;
      snake.grid_origin = grid;
      
      bg.set_position(origin);
      bg.set_size(size / 2.f);
      
      for (auto& p : snake.trail) {
        p = (p - old_origin) / old_size * size + origin;
      }
      snake.head.set_position((snake.head.get_position() - old_origin) / old_size * size + origin);
      
      f32_t scale = tile.x / old_tile.x;
      snake.head.set_radius(snake.head.get_radius() * scale);
      snake.length *= scale;
      snake.rebuild(tile.x);
      
      food.pos = (food.pos - old_origin) / old_size * size + origin;
      food.sprite.set_position(food.pos);
      food.sprite.set_radius(tile.x * 0.3f);
    });
  }
  void on_resize(fan::vec2 win) {
    old_origin = origin;
    old_size = size;
    old_tile = tile;
    
    origin = win / 2.f;
    f32_t min_dim = win.min();
    tile = fan::vec2(min_dim, min_dim) / tile_count;
    size = tile * tile_count;
  }
  void run() {
    engine.loop([&] {
      if (auto hud = gui::hud("snake_hud")) {
        gui::text(score);
      }

      snake.update(engine.delta_time, engine.get_input_vector(), tile);
      
      if (snake.collided(size)) {
        score = 0;
        snake.length = 192.f;
        fan::vec2 grid = origin - size / 2.f;
        snake.init(grid, size, tile);
        food.spawn(grid, tile, tile_count, snake);
      }
      
      if ((snake.head.get_position() - food.pos).length() < snake.head.get_radius()) {
        score++;
        snake.length += tile.x;
        food.spawn(origin - size / 2.f, tile, tile_count, snake);
      }
      
      fan::graphics::update_infinite_tiled_sprite(bg, tile, size);
    });
  } 

  engine_t engine;
  snake_t snake;
  food_t food;
  sprite_t bg;
  fan::vec2 tile;
  fan::vec2 origin;
  fan::vec2 size;
  fan::vec2 old_tile;
  fan::vec2 old_origin;
  fan::vec2 old_size;
  engine_t::resize_handle_t resize_handle;
  int score = 0;
};

int main() {
  app_t app;
  app.run();
}