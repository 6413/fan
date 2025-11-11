import fan;

int main() {
  fan::graphics::engine_t engine;
  engine.clear_color = fan::colors::black;
  fan::graphics::gui::particle_editor_t editor;

  engine.loop([&] {
    fan::graphics::rectangle({
      .position = fan::vec3(engine.window.get_size() / 2.f, 0),
      .size = engine.window.get_size() / 2.f,
      .color = editor.bg_color
    });

    editor.render();
  });
}