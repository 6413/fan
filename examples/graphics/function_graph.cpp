#include fan_pch

f64_t f(f64_t x) {
  return sin(x / 10);
}

int main() {
  loco_t loco;

  fan::vec2 window_size = loco.get_window()->get_size();
  loco.default_camera->camera.set_ortho(
    fan::vec2(0, window_size.x),
    fan::vec2(0, window_size.y)
  );

  loco_t::shape_t line = fan_init_struct(
    loco_t::shapes_t::line_t::properties_t,
    .src = fan::vec2(),
    .dst = fan::vec2(),
    .color = fan::colors::white
  );

  f32_t accuracy = 100;

  std::vector<loco_t::shape_t> positive_lines(accuracy, line);
  std::vector<loco_t::shape_t> negative_lines(accuracy, line);

  fan::vec2 src = 0;
  fan::vec2 dst = 0;

  for (f32_t i = 0; i < accuracy; i += 1) {

    auto inputfx = [](f32_t x) {
      return x * 10;
    };

    {
      src = f(i) * 100;
      dst = f(i + 1) * 100;

      src.x = inputfx(i);
      dst.x = inputfx(i + 1);

      positive_lines[i].set_color(fan::colors::white);
      positive_lines[i].set_line(src + 400, dst + 400);
    }
    {
      src = f(-i) * 100;
      dst = f(-i - 1) * 100;

      src.x = inputfx(-i);
      dst.x = inputfx(-i - 1);

      negative_lines[i].set_color(fan::colors::white);
      negative_lines[i].set_line(src + 400, dst + 400);
    }

  }

  f32_t zoom = 1;

  bool move = false;
  fan::vec2 pos = loco.default_camera->camera.get_camera_position();
  fan::vec2 offset = loco.get_mouse_position();

  loco.get_window()->add_buttons_callback([&](const auto& d) {

    auto update_zoom = [zoom] {
      auto window_size = gloco->get_window()->get_size();
      gloco->default_camera->camera.set_ortho(
        fan::vec2(0, window_size.x * zoom),
        fan::vec2(0, window_size.y * zoom)
      );
    };

    switch (d.button) {
      case fan::mouse_middle: {
        break;
      }
      case fan::mouse_scroll_up: {
        zoom -= 0.1;
        update_zoom();
        return;
      }
      case fan::mouse_scroll_down: {
        zoom += 0.1;
        update_zoom();
        return;
      }
      default: {
        return;
      }
   };
    move = (bool)d.state;
    pos = gloco->default_camera->camera.get_camera_position();
    offset = gloco->get_mouse_position();
  });

  loco.get_window()->add_mouse_move_callback([&](const auto& d) {
    if (move) {
      gloco->default_camera->camera.set_camera_position(pos - (d.position - offset) * zoom);
    }
  });


  loco.loop([&] {

  });

  return 0;
}