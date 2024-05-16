#include <fan/pch.h>

constexpr uint32_t count = 1000;

int main() {

  fan::time::clock c;
  loco_t loco;

  loco_t::letter_t::properties_t p;

  p.camera = &fan::graphics::default_camera->camera;
  p.viewport = &fan::graphics::default_camera->viewport;

  std::vector<loco_t::shape_t> letters;
  letters.reserve(count);
  for (uint32_t i = 0; i < count; i++) {
    p.position = fan::vec3(fan::random::value_f32(-1, 1), fan::random::value_f32(-1, 1), i);
    p.color = fan::color(1, 0, f32_t(i) / count, 1);
    fan::string str = fan::random::string(1);
    p.letter_id = str.get_utf8(0);
    p.font_size = 0.1;
    
    letters.push_back(p);
  }

  
  loco.loop([&] {
    loco.get_fps();
  });

  return 0;
}