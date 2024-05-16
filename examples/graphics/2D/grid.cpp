#include <fan/pch.h>

static constexpr fan::vec2 grid_size = fan::vec2(64, 64);
static constexpr fan::vec2i grid_count = fan::vec2i(50, 50);

void reset(auto& grid) {
  for (int i = 0; i < grid_count.y; ++i) {
    for (int j = 0; j < grid_count.x; ++j) {
      if (grid[i][j].r.get_color() != fan::colors::red) {
        grid[i][j].r.set_color(fan::colors::red);
      }
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
  for (int i = 0; i < 2; ++i){
    int x = (world_pos.x - radius * (i * 2 - 1)) / grid_size.x;
    int y = (world_pos.y) / grid_size.y;
    if (ceil(x) <= floor((world_pos.x + radius) / grid_size.x) &&
    floor(y) >= floor((world_pos.x - radius) / grid_size.x)) {
      grid[x][y].r.set_color(fan::colors::green);
    }
  }
}

f32_t the_magic(f32_t r, f32_t i0, f32_t i1){
  if(i0 <= 0 && i1 >= 0){
    return r;
  }
  f32_t y = std::min(std::min(fabs(i0), fabs(i1)) / r, (f32_t)1);
  f32_t x = std::sqrt((f32_t)1 - y * y) * r;
  return x;
}
template<uint32_t c>
constexpr void _set_colors2(
  auto& grid,
  auto &gi, /* grid index */
  auto wp, /* world position */
  f32_t r, /* radius */
  f32_t er /* end radius */
){
  if constexpr(c + 1 < wp.size()){
    gi[c] = (wp[c] - er) / grid_size[c];
    f32_t sp = (f32_t)gi[c] * grid_size[c]; /* start position */
    while(1){
      f32_t rp = sp - wp[c]; /* relative position */
      f32_t roff = the_magic(r, rp, rp + grid_size[c]); /* relative offset */
      _set_colors2<c + 1>(grid, gi, wp, r, roff);
      gi[c]++;
      sp += grid_size[c];
      if(sp > wp[c] + er){
        break;
      }
    }
  }
  else if constexpr(c < wp.size()){
    gi[c] = (wp[c] - er) / grid_size[c];
    sint32_t to = (wp[c] + er) / grid_size[c];
    for(; gi[c] <= to; gi[c]++){
      if((uint32_t)gi[0] >= grid_count.x || (uint32_t)gi[1] >= grid_count.y){
        continue;
      }
      grid[gi[0]][gi[1]].r.set_color(fan::colors::green);
    }
  }
}
void set_colors2(auto& grid, auto world_pos, f32_t radius){
  auto gi = fan::cast<sint32_t>(decltype(world_pos){});
  _set_colors2<0>(grid, gi, world_pos, radius, radius);
}

int main() {
  loco_t loco = loco_t::properties_t{.window_size = 1300};

  fan::vec2 viewport_size = loco.window.get_size();

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
      .radius = 128,
      .color = fan::colors::blue -fan::color(0, 0, 0, 0.3),
      .blending = true
  }};

loco.set_vsync(0);

  loco.loop([&] {
    reset(grid);
    fan::vec2 world_pos = loco.get_mouse_position();
    c.set_position(world_pos);
    set_colors2(grid, world_pos, c.get_size().x);
    loco.get_fps();
  });
}