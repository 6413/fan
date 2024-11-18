#include <fan/pch.h>

int main() {
  //fan::opengl::context_t::major = 2;
  //fan::opengl::context_t::minor = 1;
  fan::graphics::engine_t engine;

  fan::graphics::image_t image = engine.image_load("images/duck.webp");
  fan::graphics::image_t image2 = engine.image_load("images/folder.webp");

  fan::graphics::sprite_t s{ {
      .position = fan::vec3(200,200, 0),
      .size = 200,
      .image = image
  } };

  fan::graphics::sprite_t s2{ {
      .position = fan::vec3(600,600, 0),
      .size = 200,
      .image = image2
  } };

  engine.loop([&] {
    ImGui::Begin("test");
    ImGui::Text("A");
    ImGui::End();
    s.set_position(engine.get_mouse_position());
    s2.set_position(engine.get_mouse_position() * 1.5);
  });
}