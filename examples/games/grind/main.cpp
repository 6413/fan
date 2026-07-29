import fan;
import std;

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
    f32_t angle = facing.y ? facing.y * fan::math::half_pi : ((int)facing.x < 0 ? fan::math::pi : 0.f);
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
    auto* belt = brush->get(img.get_position());
    if (!belt) return;
    facing = belt->facing;
    auto dt = fan::graphics::get_window().m_delta_time;
    auto pos = img.get_position();
    auto cell = brush->cell_at(pos);
    auto center = fan::vec2(cell.x * full_tile + half_tile, cell.y * full_tile + half_tile);
    pos += facing * full_tile * belt->scroll_speed * dt;
    if (facing.x) pos.y += (center.y - pos.y) * dt * belt->scroll_speed;
    if (facing.y) pos.x += (center.x - pos.x) * dt * belt->scroll_speed;
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
      auto cells = brush.paint_overwrite(get_mouse_position());
      for (auto& pos : cells) {
        auto cell = brush.cell_at(pos);
        if (cell == last_cell) continue;

        fan::vec2i8 facing = cell_direction(cell - last_cell);
        if (last_cell.x >= 0)
          if (auto* prev = brush.get(last_cell))
            prev->set_facing(facing);

        brush.insert(pos, {pos, facing});
        last_cell = cell;
      }
    }
    else if (is_mouse_down(1)) {
      brush.erase_update(get_mouse_position());
    }
    else if (is_mouse_clicked(2)) {
      if (auto* belt = brush.get(get_mouse_position())) {
        belt->set_facing(belt->facing.x ? fan::vec2i8{0, 1} : fan::vec2i8{1, 0});
      }
    }
    else if (is_mouse_down(fan::key_space)) {
      if (auto* belt = brush.get(get_mouse_position())) {
        ores.push_back({belt->belt.get_position(), belt->facing, &brush});
      }
    }
    else {
      brush.reset();
      last_cell = {-1, -1};
    }
  }

  grid_brush_t<conveyor_belt_t> brush{half_tile * 2.0f, {.z_by_y=false}};
  std::vector<resource_t> ores;
  fan::vec2i last_cell{-1, -1};
  interactive_camera_t ic;
};

int main() {
  app_t app;
  app.loop();
}