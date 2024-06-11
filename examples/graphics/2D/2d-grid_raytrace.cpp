#include <fan/pch.h>

#include <fan/graphics/algorithm/raycast_grid.h>

int main() {

  loco_t loco{ {.window_size = 800} };

  std::vector<std::vector<fan::vec2>> grid;

  fan::vec2 map_size(16, 16);
  fan::vec2 tile_size(32, 32);

  std::vector<std::vector<loco_t::shape_t>> map;

  fan::vec2 window_size = loco.window.get_size();

  grid.resize(map_size.y);
  map.resize(map_size.y);

  for (int i = 0; i < map_size.y; i++) {
    grid[i].resize(map_size.x);
    map[i].resize(map_size.x);
    for (int j = 0; j < map_size.x; j++) {
      // grid needs to be mod of tile_size
      grid[i][j] = tile_size / 2 + fan::vec2(j * tile_size.x, i * tile_size.y);
      map[i][j] = fan::graphics::rectangle_t{ {
        .position = fan::vec3(grid[i][j], 0),
        .size = tile_size / 2,
        .color = fan::colors::black
      } };
    }
  }

  fan::vec2 src = 0;
  fan::vec2 dst = 0;

  fan::graphics::line_t line{ {
      .color = fan::colors::red
  } };

  loco.loop([&] {

    bool left = gloco->window.key_state(fan::mouse_left) == 2;
    bool shift = gloco->window.key_state(fan::key_left_shift) == 2;

    if (left && !shift) {
      src = gloco->window.get_mouse_position();
    }
    if (left && shift) {
      dst = gloco->window.get_mouse_position();
    }
    line.set_line(src, dst);

    for (int i = 0; i < map_size.y; i++) {
      for (int j = 0; j < map_size.x; j++) {
        map[i][j].set_color(fan::colors::black);
      }
    }

    std::vector<fan::vec2i> raycast_positions = fan::graphics::algorithm::grid_raycast({ src, dst }, tile_size);
    for (auto& pos : raycast_positions) {
      map[pos.y][pos.x].set_color(fan::colors::green);
    }
  });

  return 0;
}