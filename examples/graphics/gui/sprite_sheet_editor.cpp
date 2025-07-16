#include <fan/types/types.h>

import fan;

using namespace fan::graphics;

int main() {
  engine_t engine;
  //image_t image = engine.image_load("game/enemies/silver knight/hooded knight attack.png");

  //sprite_t sprite_sheet{ {
  //  .position = 400,
  //  .size = 256,
  //  .image = image
  //} };
  //sprite_sheet.set_sprite_sheet_frames(17, 1);
  //sprite_sheet.set_sprite_sheet_update_frequency(0.05);

  //sprite_t sprite_sheet2 = sprite_sheet;
  //sprite_sheet2.set_position(sprite_sheet2.get_position() + fan::vec3(0, 200, 0));

  fan::graphics::gui::sprite_animations_t animations_application;

  engine.loop([&] {
    animations_application.render();
  });
}