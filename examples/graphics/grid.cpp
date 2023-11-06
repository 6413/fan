#include fan_pch

static constexpr fan::vec2 grid_size = fan::vec2(512, 512);
static constexpr fan::vec2i grid_count = fan::vec2i(200, 200);

void reset(auto& grid) {
  for (int i = 0; i < grid_count.y; ++i) {
    for (int j = 0; j < grid_count.x; ++j) {
      grid[i][j].r.set_color(fan::colors::red);
    }
  }
}
void set_colors(auto& grid, fan::vec2 world_pos, f32_t radius) {
  fan::vec2f grid_posi = world_pos / grid_size;
  f32_t top = floor(grid_posi.y - radius / grid_size.y);
  f32_t bottom = ceil(grid_posi.y + radius / grid_size.y);
  for (f32_t j = top; j < bottom; ++j) {
    f32_t offsety = j * grid_size.x - (world_pos.y - grid_size.y * !(j >= (top + bottom) / 2));
    f32_t dx;
    fan::print(fmodf(world_pos.y / grid_size.y, 1));
    if (j == (top + bottom) / 2 - (fmodf(world_pos.y / grid_size.y, 1) >= 0.5 ? 1 : 0) || (top + 1 >= bottom)) {
      dx = radius;
    } 
    else {
      dx = fan::math::sqrt(fan::math::abs(radius * radius - offsety * offsety));
    }
    f32_t left = (world_pos.x - dx) / grid_size.x;
    f32_t right = (world_pos.x + dx) / grid_size.x;
    for (int i = left; i < ceil(right); ++i) {
      // this if is only necessary for very big squares
     if (ceil(i) <= floor((world_pos.x + radius) / grid_size.x) &&
          floor(i) >= floor((world_pos.x - radius) / grid_size.x)) {
        grid[std::min(std::max((int)i, 0), (int)grid_size.y)][std::min(std::max((int)j, 0), (int)grid_size.x)].r.set_color(fan::colors::green);
      }
    }
  }

  // this for loop only necessary for edge cases with size near 256
  /*for (int i = 0; i < 2; ++i){
    int x = (world_pos.x - radius * (i * 2 - 1)) / grid_size.x;
    int y = (world_pos.y) / grid_size.y;
    if (ceil(x) <= floor((world_pos.x + radius) / grid_size.x) &&
    floor(y) >= floor((world_pos.x - radius) / grid_size.x)) {
      grid[x][y].r.set_color(fan::colors::green);
    }
  }*/
}

int main() {
  loco_t loco = loco_t::properties_t{.window_size = 600};

  fan::vec2 viewport_size = loco.get_window()->get_size();
  loco.default_camera->camera.set_ortho(
    fan::vec2(0, viewport_size.x),
    fan::vec2(0, viewport_size.y)
  );

  struct cell_t {
    fan::graphics::rectangle_t r;
  };

  cell_t grid[grid_count.y][grid_count.x]{};

  for (int i = 0; i < grid_count.y; ++i) {
    for (int j = 0; j < grid_count.x; ++j) {
      grid[i][j].r = fan::graphics::rectangle_t{{
          .position = fan::vec2(i, j) * grid_size + grid_size / 2,
          .size = grid_size / 2,
          .color = fan::colors::red
      }};
    }
  }
  fan::graphics::line_t grid_linesx[(int)(1300 / grid_size.x) + 1];
  fan::graphics::line_t grid_linesy[(int)(1300 / grid_size.y) + 1];


  for (int i = 0; i < 1300 / grid_size.x; ++i) {
    grid_linesx[i] = fan::graphics::line_t{{
        .src = fan::vec3(i * grid_size.x, 0, 2),
        .dst = fan::vec2(i * grid_size.x, viewport_size.y),
        .color = fan::colors::white
    }};
  }
  for (int j = 0; j < 1300 / grid_size.y; ++j) {
    grid_linesy[j] = fan::graphics::line_t{{
      .src = fan::vec3(0, j * grid_size.y, 2),
      .dst = fan::vec2(viewport_size.x, j * grid_size.y),
      .color = fan::colors::white
    }};
  }

  fan::graphics::circle_t c{{
      .position = fan::vec3(0, 0, 3),
      .radius = 258,
      .color = fan::colors::blue -fan::color(0, 0, 0, 0.3),
      .blending = true
  }};

  loco.loop([&] {
    reset(grid);
    fan::vec2 world_pos = loco.get_mouse_position();
    c.set_position(world_pos);
    set_colors(grid, world_pos, c.get_size().x);
  });
}