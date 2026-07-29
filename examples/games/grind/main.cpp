import std;
import fan;

using namespace fan::graphics;

static constexpr f32_t half_tile = 64.f;
static constexpr f32_t full_tile = half_tile * 2.;

struct power_source_t {
  power_source_t(fan::vec2 pos, f32_t power = 0.f) 
    : img(fan::vec3(pos, 10.f), half_tile, fan::colors::green), 
      power(power) {}
  sprite_t img;
  f32_t power;
};

struct conveyor_belt_t : fan::frame_task_t<conveyor_belt_t> {
  conveyor_belt_t(fan::vec2 pos, fan::vec2i8 facing)
    : belt(      fan::vec3(pos, 10.f),             half_tile, {"conveyor_belt.webp"      }), 
      facing(facing),
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
  f32_t scroll_speed = 3.f;
};

struct resource_t : fan::frame_task_t<resource_t> {
  resource_t(fan::vec2 pos, fan::vec2i8 facing, grid_brush_t<conveyor_belt_t>* brush = nullptr)
    : img(fan::vec3(pos, 20.f), half_tile, {"images/ruby.webp"}), facing(facing), brush(brush) {}

  void update() {
    auto dt = fan::graphics::get_window().m_delta_time;
    fan::vec2 pos = img.get_position();
    auto* belt = brush->get(pos);
    if (!belt) return;
    facing = belt->facing;
    if (!brush->follow_path(pos, facing, belt->scroll_speed, dt, [&](auto cell, auto& f) {
      auto* b = brush->get(cell);
      if (!b) return false;
      f = b->facing;
      return true;
    })) return;
    img.set_position(pos);
  }

  sprite_t img;
  fan::vec2i8 facing;
  grid_brush_t<conveyor_belt_t>* brush;
};

struct app_t : engine_t {
  app_t() : grass(/*engine.create_transparent_texture()*//**/"images/grass1.jpg") {
    engine_t::loop([&] {loop(); });
    get_lighting().set_target(1.f);
  }

  void loop() {
    update_tiling_background(grass, half_tile);

    if (is_key_down(fan::key_e)) {
      brush_ps.paint_directional(get_mouse_position(), [&](auto cell, auto facing, auto prev) {
        (void)facing;
        brush_ps.insert(cell);
      });
    }
    if (is_mouse_down()) {
      brush.paint_directional(get_mouse_position(), [&](auto cell, auto facing, auto prev) {
        if (prev.x >= 0)
          if (auto* p = brush.get(prev))
            p->set_facing(facing);
        brush.insert(cell, facing);
      });
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
    else if (is_key_down(fan::key_space) && fan::time::every(100)) {
      if (auto* belt = brush.get(get_mouse_position())) {
        ores.emplace_back(belt->belt.get_position(), belt->facing, &brush);
      }
    }
    else {
      brush.reset();
    }
  }

  grid_brush_t<conveyor_belt_t> brush{half_tile * 2.0f, {.z_by_y=false}};
  grid_brush_t<power_source_t> brush_ps{half_tile * 2.0f, {.z_by_y=false}};
  std::deque<resource_t> ores;
  interactive_camera_t ic;
  sprite_t grass;
};

int main() {
  app_t app;
   gloco()->set_settings({
    .mode = fan::graphics::post_process_mode_e::bloom,
    .bloom_strength = 0.095f,
    .bloom_threshold = 0.38f,
    .bloom_knee = 0.039f,
    .bloom_smooth_rate = 6.431f,
    .bloom_luma_scale = 1.144f,
    .bloom_adaptation_blend = 0.203f,
    .bloom_filter_radius = 1.0f,
    .gamma = 2.2f,
    .exposure = 1.0f,
    .contrast = 1.0f,
  });
  app.loop();
}