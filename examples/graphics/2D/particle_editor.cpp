#include <climits>

import fan;
int main() {
  using namespace fan::graphics;

  engine_t engine;
  engine.clear_color = fan::colors::black;
  gui::particle_editor_t editor;

  interactive_camera_t ic;

  editor.particle_shape = shapes::particles_t::properties_t {
    .position = fan::vec3(engine.window.get_size() / 2.f, 10.0f),
    .size = 32.0f,
    /*.color = fan::color::from_rgba(0xFF6600FF),*/
    .alive_time = 1.0,
    .count = 100,
    .position_velocity = fan::vec2(100.0f, 100.0f),
    .begin_angle = 0.0f,
    .end_angle = 6.28f,
    .angle = fan::vec3(0.0f, 0.0f, 0.0f),
    .gap_size = fan::vec2(0.0f, 0.0f),
    .max_spread_size = fan::vec2(0.0f, 0.0f),
    .shape = shapes::particles_t::shapes_e::circle,
    .image = image_load("examples/games/platformer/effects/bubble.png", image_presets::pixel_art())
  };

  engine.loop([&] {
    gui::begin("##ViewerTest", 0, gui::render_window_flags());
    gui::set_viewport();
    rectangle({
      .position = fan::vec3(ic.camera_offset, 0),
      .size = gui::get_window_size() / 2.f / ic.zoom,
      .color = editor.bg_color
    });
    editor.render();
    gui::end();
  });
}