#include fan_pch

static constexpr f32_t zoom_sensitivity = 1.2;

static constexpr f32_t amplitude = 1;
static constexpr f32_t wavelength = 10;

struct global_t {
  loco_t loco;
  static constexpr f32_t samples = 100;

  f32_t zoom = 1;
  bool move = false;
  fan::vec2 pos = gloco->default_camera->camera.get_position();
  fan::vec2 offset = gloco->get_mouse_position();
}global;

f64_t f(f64_t x) {
  return x * x;
}

void handle_zoom_and_move() {

  gloco->get_window()->add_buttons_callback([&](const auto& d) {

    auto update_zoom = [] {
      auto window_size = gloco->get_window()->get_size();
      gloco->default_camera->camera.set_ortho(
        fan::vec2(-window_size.x, window_size.x) / (global.zoom),
        fan::vec2(-window_size.y, window_size.y) / (global.zoom)
      );
     };

    switch (d.button) {
      case fan::mouse_middle: {
        global.move = (bool)d.state;
        global.pos = gloco->default_camera->camera.get_position();
        global.offset = gloco->get_mouse_position();
        break;
      }
      case fan::mouse_scroll_up: {
        global.zoom *= zoom_sensitivity;
        update_zoom();
        break;
      }
      case fan::mouse_scroll_down: {
        global.zoom /= zoom_sensitivity;
        update_zoom();
        break;
      }
    };
 });
}

int main() {
  fan::vec2 window_size = global.loco.get_window()->get_size();
  global.loco.default_camera->camera.set_ortho(
    fan::vec2(-window_size.x, window_size.x),
    fan::vec2(-window_size.y, window_size.y)
  );

  handle_zoom_and_move();

  global.loco.get_window()->add_mouse_move_callback([&](const auto& d) {
    if (global.move) {
      gloco->default_camera->camera.set_position(global.pos - (d.position - global.offset) / global.zoom * 2);
    }
  });

  std::vector<loco_t::shape_t> lines(global.samples, fan::graphics::line_t{{
      .src = fan::vec2(),
      .dst = fan::vec2(),
      .color = fan::colors::white
  }});

  f32_t divider = global.samples / 10.f;

  static auto generate_line = [&](int line_idx, f32_t x) {
    fan::vec2 src = f(x / divider) * amplitude * global.samples;
    fan::vec2 dst = f((x + 1.f) / divider) * amplitude * global.samples;
    src.x = line_idx * wavelength;
    dst.x = (line_idx + 1) * wavelength;
    lines[line_idx].set_line(src, dst);
  };

  for (f32_t i = 0; i < global.samples; i += 1) {
    generate_line(i, (i));
  }

  f32_t animator_index = 0;
  global.loco.loop([&] {

    //for (f32_t i = 0; i < global.samples; i += 1) {
    //  generate_line(i, -(-i + animator_index));
    //}
    //animator_index += global.loco.get_delta_time() * 20;
  });

  return 0;
}