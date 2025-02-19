#include <fan/pch.h>

int main() {
  loco_t loco;

  loco_t::image_t image;
  image = loco.image_load("images/tire.webp");
  std::vector<loco_t::shape_t> shapes;
  for (int i = 0; i < 50; ++i)
  shapes.push_back(fan::graphics::sprite_t{ {
    .position = fan::vec3(fan::random::vec2(0, 800), 254),
    .size = 50,
    .image = image,
    .flags = 0x4
  } });
  for (int i = 0; i < 50; ++i)
  shapes.push_back(fan::graphics::circle_t{ {
  .position = fan::vec3(fan::random::vec2(0, 800), 254),
  .radius = 50,
  .color = fan::colors::white,
  .blending = true,
  .flags = 0x4
} });

  loco.set_vsync(0);
  loco.loop([&] {
    static bool x = 0;
    if (ImGui::ToggleButton("abc", &x)) {
      for (std::size_t i = 0; i < shapes.size(); ++i) {
        if (fan::random::value_i64(0, 3) == 0) {
          shapes[i].set_flags(0x4 * x);
        }
        else {
          shapes[i].set_flags(0x0);
        }
      }
    }
    
    });

  return 0;
}