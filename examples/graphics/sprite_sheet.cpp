#include fan_pch

int main() {

  loco_t loco;
  loco.clear_color = fan::color::hex(0x596988ff);

  fan::vec2 viewport_size = loco.window.get_size();
  loco.default_camera->camera.set_ortho(
    fan::vec2(0, viewport_size.x),
    fan::vec2(0, viewport_size.y)
  );

  loco_t::image_t frames[7];
  for (int i = 0; i < std::size(frames); ++i) {
    frames[i].load(fan::format("frames/frame{}.webp", i));
  }

  loco_t::sheet_t sheet;
  for (uint32_t i = 0; i < std::size(frames); ++i) {
    sheet.push_back(&frames[i]);
  }
  sheet.animation_speed = 1e+8;
  loco_t::sprite_sheet_t::properties_t p;
  p.position = viewport_size / 2;
  p.size = 100;
  p.sheet = &sheet;
  auto nr = loco.sprite_sheet.push_back(p);

  loco.sprite_sheet.start(nr);

  loco.loop([&] {
    loco.get_fps();
  });

  return 0;
}