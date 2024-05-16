#include <fan/pch.h>

int main() {
  loco_t loco;

  loco_t::shapes_t::rectangle_t::properties_t rectangle_properties;
  rectangle_properties.camera = &fan::graphics::default_camera->camera;
  rectangle_properties.viewport = &fan::graphics::default_camera->viewport;

  rectangle_properties.position = fan::vec3(0, 0, 0);
  rectangle_properties.size = fan::vec2(0.1, 0.1);
  rectangle_properties.color = fan::colors::red;

  loco_t::shape_t rectangle = rectangle_properties;

  loco.loop([&] {

  });

  return 0;
}