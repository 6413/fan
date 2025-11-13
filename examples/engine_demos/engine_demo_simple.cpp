#include <fan/utility.h>
#include <vector>

import fan;

#include <fan/graphics/types.h>

struct simple_engine_demo_t {
   // Always initialize engine before making shapes
  fan::graphics::engine_t engine;

  simple_engine_demo_t() {
    init_shapes();
    setup_input();
  }

  fan::vec2 get_mouse_position() const {
    // Since camera is transformable, we want the mouse position inside the world, not the window
    return engine.get_mouse_position(engine.orthographic_render_view);
  }

  void init_shapes() {
    fan::vec2 window_size = fan::window::get_size();
    fan::vec2 center = window_size / 2;

    rectangles.push_back(fan::graphics::rectangle_t{
      center - fan::vec2(200, 0),
      fan::vec2(64, 64),
      fan::colors::red
    });

    circles.push_back(fan::graphics::circle_t{{
      .position = center + fan::vec2(200, 0),
      .radius = 32.f,
      .color = fan::colors::blue
    }});
    circle_velocities.push_back(fan::vec2(80.f, -60.f));

    sprites.push_back(fan::graphics::sprite_t{{
      .position = center,
      .size = fan::vec2(96, 96),
      .color = fan::colors::white,
      .image = tire_image
    }});

    sprites.push_back(fan::graphics::sprite_t{{
      .position = center + fan::vec2(0, 200),
      .size = fan::vec2(64, 64),
      .color = fan::random::bright_color(),
      .image = tire_image
    }});

    lines.push_back(fan::graphics::line_t{
      fan::vec3(100, 100, 0),
      fan::vec3(300, 300, 0),
      fan::colors::cyan,
      3.f
    });

    lights.push_back(fan::graphics::light_t{{
      .position = fan::vec3(center, 0),
      .size = fan::vec2(200, 200),
      .color = fan::colors::yellow
    }});
  }

  void setup_input() {
    interactive_camera.pan_with_middle_mouse = true;

    // We are using a centered coordinate system where the origin (0, 0) is in the middle of the window.
    // That means the visible world spans from -half_window_size to +half_window_size.
    // To align the camera with this world space, we move it to window_size / 2,
    // so that the camera's view matches the centered coordinates
    interactive_camera.set_position(fan::window::get_size() / 2.f);

    // Update the camera view
    interactive_camera.update();

    mouse_move_nr = engine.on_mouse_move([&](fan::vec2 pos, fan::vec2 delta) {
      // Since camera is transformable, we want the mouse position inside the world, not the window
      mouse_position = fan::graphics::transform_position(pos, engine.orthographic_render_view);
    });

    mouse_click_nr = engine.on_mouse_click(fan::mouse_left, [&]() {
      circles.push_back(fan::graphics::circle_t{{
        .position = get_mouse_position(),
        .radius = fan::random::value(16.f, 48.f),
        .color = fan::random::bright_color()
      }});
      circle_velocities.push_back(fan::random::vec2(-120.f, 120.f));
    });

    key_click_nr = engine.on_key_click(fan::key_space, [&] {
      fan::vec2 window_size = fan::window::get_size();
      rectangles.push_back(fan::graphics::rectangle_t{
        fan::vec3(fan::random::vec2(0, window_size), 0),
        fan::random::vec2(32, 128),
        fan::random::color()
      });
    });

    key_click_nr2 = engine.on_key_click(fan::key_r, [&] {
      rectangles.clear();
      circles.clear();
      sprites.clear();
      circle_velocities.clear();
    });
  }

  void update() {
    if (fan::window::is_key_down(fan::key_w)) {
      rectangles[0].set_position(rectangles[0].get_position() + fan::vec3(0, -200 * engine.delta_time, 0));
    }
    if (fan::window::is_key_down(fan::key_s)) {
      rectangles[0].set_position(rectangles[0].get_position() + fan::vec3(0, 200 * engine.delta_time, 0));
    }
    if (fan::window::is_key_down(fan::key_a)) {
      rectangles[0].set_position(rectangles[0].get_position() + fan::vec3(-200 * engine.delta_time, 0, 0));
    }
    if (fan::window::is_key_down(fan::key_d)) {
      rectangles[0].set_position(rectangles[0].get_position() + fan::vec3(200 * engine.delta_time, 0, 0));
    }

    sprites[0].set_angle(sprites[0].get_angle() + fan::vec3(0, 0, engine.delta_time));

    lights[0].set_position(fan::vec3(mouse_position, 0));
    lights[0].set_color(fan::color::hsv(fmod(engine.time * 50, 360), 100, 100));

    size_t m = std::min(circles.size(), circle_velocities.size());
    for (size_t i = 0; i < m; ++i) {
      fan::vec3 p = circles[i].get_position();
      // Raise the Z value slightly so overlapping shapes don't fight for the same depth
      // This avoids flickering artifacts (z-fighting) when two objects share the same plane
      p += fan::vec3(circle_velocities[i] * engine.delta_time, 10 + i);
      circles[i].set_position(p);
    }

    fan::graphics::gui::text("WASD - Move red rectangle");
    fan::graphics::gui::text("Space - Spawn random rectangle");
    fan::graphics::gui::text("Left Click - Spawn moving circle");
    fan::graphics::gui::text("R - Clear all shapes");
    fan::graphics::gui::text("Mouse Move - Light follows cursor, affects sprites");
  }

  fan::graphics::image_t tire_image = engine.image_load("images/tire.webp");
  std::vector<fan::graphics::rectangle_t> rectangles;
  std::vector<fan::graphics::circle_t> circles;
  std::vector<fan::graphics::sprite_t> sprites;
  std::vector<fan::graphics::line_t> lines;
  std::vector<fan::graphics::light_t> lights;
  std::vector<fan::vec2> circle_velocities;
  fan::vec2 mouse_position;
  fan::graphics::engine_t::mouse_move_nr_t mouse_move_nr;
  fan::graphics::engine_t::mouse_click_nr_t mouse_click_nr;
  fan::graphics::engine_t::key_click_nr_t key_click_nr;
  fan::graphics::engine_t::key_click_nr_t key_click_nr2;

  // Enables interactive camera controls
  // - Hold the middle mouse button to pan the view.
  // - Use the mouse scroll wheel to zoom in and out
  fan::graphics::interactive_camera_t interactive_camera;
};

int main() {
  simple_engine_demo_t demo;

  fan_window_loop {
    demo.update();
  };
}