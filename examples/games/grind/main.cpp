import std;
import fan;

using namespace fan::graphics;

static constexpr f32_t half_tile = 64.f;
static constexpr f32_t full_tile = half_tile * 2.;

struct conveyor_belt_t : fan::frame_task_t<conveyor_belt_t> {
  conveyor_belt_t(fan::vec2 pos, fan::vec2i8 facing)
    : belt(      fan::vec3(pos, 10.f),             half_tile, {"conveyor_belt.webp"      }), facing(facing),
      belt_sides(fan::vec3(pos, 10.f).offset_z(1), half_tile, {"conveyor_belt_sides.webp"}) {
    set_facing(facing);
  }

  void set_facing(fan::vec2i8 f) {
    facing = f;
    f32_t angle = facing.y ? facing.y * fan::math::half_pi : (facing.x < 0 ? fan::math::pi : 0.f);
    belt.set_angle(fan::vec3(0, 0, angle));
    belt_sides.set_angle(fan::vec3(0, 0, angle));
  }

  void update() {
    belt.uv_uniform_scroll({-1, 0}, scroll_speed);
  }

  sprite_t belt;
  sprite_t belt_sides;
  fan::vec2i8 facing;
  f32_t scroll_speed = 10.f;
};

struct resource_t : fan::frame_task_t<resource_t> {
  resource_t(fan::vec2 pos, fan::vec2i8 facing, grid_brush_t<conveyor_belt_t>* brush = nullptr)
    : img(fan::vec3(pos, 20.f), half_tile, {"images/rock2.webp"}), facing(facing), brush(brush) {}

  void update() {
    auto dt = fan::graphics::get_window().m_delta_time;
    auto pos = img.get_position();

    auto* belt = brush->get(pos);
    if (!belt) {
      return;
    }
    facing = belt->facing;

    f32_t total_dist = full_tile * belt->scroll_speed * dt;
    f32_t max_step = half_tile * 0.25f;
    int steps = std::max(1, (int)(total_dist / max_step) + 1);
    f32_t step_dist = total_dist / steps;
    f32_t snap = dt * belt->scroll_speed / steps;

    fan::vec2i cell = brush->cell_at(pos);
    fan::vec2 center(cell.x * full_tile + half_tile, cell.y * full_tile + half_tile);

    for (int i = 0; i < steps; ++i) {
      pos += fan::vec2(facing) * step_dist;
      if (facing.x) {
        pos.y = std::clamp(pos.y, cell.y * full_tile, cell.y * full_tile + full_tile);
      }
      if (facing.y) {
        pos.x = std::clamp(pos.x, cell.x * full_tile, cell.x * full_tile + full_tile);
      }

      auto new_cell = brush->cell_at(pos);
      if (new_cell != cell) {
        belt = brush->get(new_cell);
        if (!belt) {
          break;
        }
        facing = belt->facing;
        cell = new_cell;
        center = fan::vec2(cell.x * full_tile + half_tile, cell.y * full_tile + half_tile);
      }

      if (facing.x) {
        pos.y += (center.y - pos.y) * snap;
      }
      if (facing.y) {
        pos.x += (center.x - pos.x) * snap;
      }
    }
    img.set_position(pos);
  }

  sprite_t img;
  fan::vec2i8 facing;
  grid_brush_t<conveyor_belt_t>* brush;
};

struct app_t : engine_t {
  app_t() {
    engine_t::loop([&] {loop(); });
    get_lighting().set_target(1.f);
  }

  void loop() {
    if (is_mouse_down()) {
      brush.paint_directional(get_mouse_position());
    }
    else if (is_mouse_down(1)) {
      brush.erase_update(get_mouse_position());
    }
    else if (is_mouse_clicked(2)) {
      if (auto* belt = brush.get(get_mouse_position())) {
        static constexpr fan::vec2i8 dirs[4]{{1,0},{0,1},{-1,0},{0,-1}};
        int i = 0;
        for (; i < 4; ++i) {
          if (dirs[i] == belt->facing) {
            break;
          }
        }
        belt->set_facing(dirs[(i + 1) % 4]);
      }
    }
    else if (is_mouse_down(fan::key_space)) {
      if (auto* belt = brush.get(get_mouse_position())) {
        ores.emplace_back(belt->belt.get_position(), belt->facing, &brush);
      }
    }
    else {
      brush.reset();
    }
  }

  grid_brush_t<conveyor_belt_t> brush{half_tile * 2.0f, {.z_by_y=false}};
  std::deque<resource_t> ores;
  interactive_camera_t ic;
};

int main() {
  app_t app;
  app.loop();
}