#include <fan/pch.h>

int main() {

  loco_t loco;
  loco_t::gradient_t::properties_t p;
  p.position = fan::vec3(loco.window.get_size() / 2, 0);
  p.size = 300;
  p.color[0] = fan::color(1, 0, 0, 1);
  p.color[1] = fan::color(1, 0, 0, 1);
  p.color[2] = fan::color(0, 0, 1, 1);
  p.color[3] = fan::color(0, 0, 1, 1);
  loco_t::shape_t rect = p;


  loco.loop([&] {

  });
}