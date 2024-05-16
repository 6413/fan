#include <fan/pch.h>

#include _FAN_PATH(system.h)

int main() {

  loco_t loco;
  loco.default_camera->camera.set_ortho(
  fan::vec2(-1, 1),
  fan::vec2(-1, 1)
  );

  fan::sys::MD_SCR_t md;

  wglMakeCurrent(loco.window.m_hdc, loco.window.m_context);
  int i = 0;
  while (i < 1) {
    fan::sys::MD_SCR_open(&md);
    fan::sys::MD_SCR_close(&md);
    i++;
    fan::print(i);
  }


  loco_t::shapes_t::sprite_t::properties_t p;

  p.size = fan::vec2(1, 1);

  fan::sys::MD_SCR_open(&md);
  uint8_t* ptr = fan::sys::MD_SCR_read(&md);
  ptr = fan::sys::MD_SCR_read(&md);
  while (!ptr) {
    ptr = fan::sys::MD_SCR_read(&md);
  }
  if (ptr == nullptr) {
    return 1;
  }
  fan::webp::image_info_t ii;
  ii.size = fan::sys::get_screen_resolution();
  ii.data = ptr;

  loco_t::image_t image;
  loco_t::image_t::load_properties_t lp;
  lp.format = loco_t::image_t::format::b8g8r8a8_unorm;
  lp.internal_format = loco_t::image_t::format::r8g8b8a8_srgb;
  image.load(ii, lp);
  p.image = &image;
  p.position.z = 0;
  loco_t::shape_t shape = p;


  loco.set_vsync(false);
  uint32_t x = 0;
  loco.loop([&] {
    ptr = fan::sys::MD_SCR_read(&md);
    if (ptr) {
      ii.data = ptr;
      image.unload();
      image.load(ii, lp);
    }
  });

  return 0;
}