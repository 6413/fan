#include <fan/types/types.h>

import fan.event;
import fan.audio;
import fan.print;
import fan.graphics.loco;
import fan.graphics;
static constexpr f32_t zoom_sensitivity = 1.2;

static constexpr f32_t amplitude = 1;
static constexpr f32_t wavelength = 10;

struct global_t {
  loco_t loco;
  static constexpr f32_t samples = 100;

  f32_t zoom = 1;
  bool move = false;
  fan::vec2 pos = gloco->camera_get_position(gloco->orthographic_render_view.camera);
  fan::vec2 offset = gloco->get_mouse_position();
}global;

f64_t f(f64_t x) {
  return exp(x);
}

f64_t f2(f64_t x) {
  return pow(x, 2);
}

void handle_zoom_and_move() {

  gloco->window.add_buttons_callback([&](const auto& d) {

    auto update_zoom = [] {
      auto window_size = gloco->window.get_size();
      gloco->camera_set_ortho(gloco->orthographic_render_view.camera,
        fan::vec2(-window_size.x, window_size.x) / (global.zoom),
        fan::vec2(-window_size.y, window_size.y) / (global.zoom)
      );
     };

    switch (d.button) {
      case fan::mouse_middle: {
        global.move = (bool)d.state;
        global.pos = gloco->camera_get_position(gloco->orthographic_render_view.camera);
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
  fan::vec2 window_size = global.loco.window.get_size();
  gloco->camera_set_ortho(gloco->orthographic_render_view.camera, 
    fan::vec2(-window_size.x, window_size.x),
    fan::vec2(-window_size.y, window_size.y)
  );

  handle_zoom_and_move();

  global.loco.window.add_mouse_move_callback([&](const auto& d) {
    if (global.move) {
      gloco->camera_set_position(gloco->orthographic_render_view.camera, global.pos - (d.position - global.offset) / global.zoom * 2);
    }
  });

  std::vector<loco_t::shape_t> lines(global.samples, fan::graphics::line_t{{
      .src = fan::vec2(),
      .dst = fan::vec2(),
      .color = fan::colors::white
  }});
  std::vector<loco_t::shape_t> lines2(global.samples, fan::graphics::line_t{{
      .src = fan::vec2(),
      .dst = fan::vec2(),
      .color = fan::colors::white
  }});

  f32_t divider = global.samples / 10.f;

  static auto generate_line = [&](auto& line, auto f, int line_idx, f32_t x) {
    fan::vec2 src = f(x / divider) * amplitude * global.samples;
    fan::vec2 dst = f((x + 1.f) / divider) * amplitude * global.samples;
    src.x = line_idx * wavelength;
    dst.x = (line_idx + 1) * wavelength;
    line.set_line(src, dst);
  };

  //for (f32_t i = 0; i < global.samples; i += 1) {
  //  generate_line(lines[i], i, (i));
  //}

  f32_t animator_index = 0;
  global.loco.loop([&] {

    for (f32_t i = 0; i < global.samples; i += 1) {
      generate_line(lines[i], f, i, (i + animator_index));
      generate_line(lines2[i], f2, i, (i + animator_index));
    }
    animator_index -= global.loco.delta_time * 20;
  });

  return 0;
}