#include <fan/pch.h>

int main() {
  loco_t loco;

  loco_t::text_t::properties_t p;
  p.camera = &fan::graphics::default_camera->camera;
  p.viewport = &fan::graphics::default_camera->viewport;
  p.position = fan::vec2(-0.7, 0);

  p.font_size = 0.1;
  p.text = "hello";
  p.color = fan::colors::white;

  loco_t::shape_t text0 = p;

  loco.loop([&] {
    loco.get_fps();
  });

  return 0;
}