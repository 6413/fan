#include <fan/pch.h>
#include <fan/graphics/algorithm/AStar.hpp>

#include <fan/pch.h>
#include <string_view>

static constexpr fan::vec2 grid_size = fan::vec2(64, 64);
static constexpr fan::vec2i grid_count = fan::vec2i(50, 50);



int main() {
  constexpr fan::vec2 window_size = 1300;
  loco_t loco = loco_t::properties_t{ .window_size = window_size };

  fan::vec2 viewport_size = loco.window.get_size();

  struct cell_t {
    fan::graphics::rectangle_t r;
  };

  cell_t grid[grid_count.y][grid_count.x]{};

  for (int i = 0; i < grid_count.y; ++i) {
    for (int j = 0; j < grid_count.x; ++j) {
      grid[i][j].r = fan::graphics::rectangle_t{ {
          .position = fan::vec2(i, j) * grid_size + grid_size / 2,
          .size = grid_size / 2,
          .color = fan::colors::gray
      } };
    }
  }
  fan::graphics::line_t grid_linesx[(int)(window_size.x / grid_size.x) + 1];
  fan::graphics::line_t grid_linesy[(int)(window_size.y / grid_size.y) + 1];


  for (int i = 0; i < window_size.x / grid_size.x; ++i) {
    grid_linesx[i] = fan::graphics::line_t{ {
        .src = fan::vec3(i * grid_size.x, 0, 2),
        .dst = fan::vec2(i * grid_size.x, viewport_size.y),
        .color = fan::colors::white
    } };
  }
  for (int j = 0; j < window_size.y / grid_size.y; ++j) {
    grid_linesy[j] = fan::graphics::line_t{ {
      .src = fan::vec3(0, j * grid_size.y, 2),
      .dst = fan::vec2(viewport_size.x, j * grid_size.y),
      .color = fan::colors::white
    } };
  }

  loco.set_vsync(0);

  loco.input_action.add(fan::mouse_left, "set_src");
  loco.input_action.add(fan::mouse_right, "set_dst");
  loco.input_action.add_keycombo({fan::mouse_left, fan::key_left_shift}, "set_wall");
  loco.input_action.add_keycombo({fan::mouse_right, fan::key_left_shift}, "remove_wall");

  static auto clear = [&] {
    for (int i = 0; i < grid_count.y; ++i) {
      for (int j = 0; j < grid_count.x; ++j) {
        grid[i][j].r.set_color(fan::colors::gray);
      }
    }
  };

  fan::vec2i src_p = 0;
  fan::vec2i dst_p = 0;

  AStar::Generator generator;
  generator.setWorldSize(grid_count);
  generator.setHeuristic(AStar::Heuristic::euclidean);
  generator.setDiagonalMovement(false);

  AStar::CoordinateList path;

  loco.loop([&] {
    if (loco.input_action.is_action_down("remove_wall")) {
      fan::vec2 mp = loco.get_mouse_position();
      fan::vec2i gp = (mp / grid_size).floor();
      if (gp > 0 && gp < grid_size) {
        generator.removeCollision(gp);
        grid[gp.x][gp.y].r.set_color(fan::colors::gray);
      }
    }
    else if (loco.input_action.is_action_down("set_wall")) {
      fan::vec2 mp = loco.get_mouse_position();
      fan::vec2i gp = (mp / grid_size).floor();
      if (gp > 0 && gp < grid_size) {
        generator.addCollision(gp);
        grid[gp.x][gp.y].r.set_color(fan::colors::gray / 3);
      }
    }
    else {
      if (loco.input_action.is_action_down("set_src")) {
        grid[src_p.x][src_p.y].r.set_color(fan::colors::gray);
        fan::vec2 mp = loco.get_mouse_position();
        fan::vec2i gp = (mp / grid_size).floor();
        if (gp > 0 && gp < grid_size) {
          grid[gp.x][gp.y].r.set_color(fan::colors::green);
          src_p = gp;
        }
        grid[dst_p.x][dst_p.y].r.set_color(fan::colors::red);
      }
      if (loco.input_action.is_action_down("set_dst")) {
        grid[dst_p.x][dst_p.y].r.set_color(fan::colors::gray);
        fan::vec2 mp = loco.get_mouse_position();
        fan::vec2i gp = (mp / grid_size).floor();
        if (gp > 0 && gp < grid_size) {
          grid[gp.x][gp.y].r.set_color(fan::colors::red);
          dst_p = gp;
        }
        grid[src_p.x][src_p.y].r.set_color(fan::colors::green);
      }
    }

    for (const auto& p : path) {
      grid[p.x][p.y].r.set_color(fan::colors::gray);
    }

    path = generator.findPath(src_p, dst_p);
    for (const auto& p : path) {
      grid[p.x][p.y].r.set_color(fan::colors::cyan);
    }

    grid[src_p.x][src_p.y].r.set_color(fan::colors::green);
    grid[dst_p.x][dst_p.y].r.set_color(fan::colors::red);
  });
}