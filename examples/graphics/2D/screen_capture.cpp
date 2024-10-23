#include <fan/pch.h>

#include _FAN_PATH(system.h)

int main() {

  loco_t loco;
  loco.camera_set_ortho(
    loco.orthographic_camera.camera,
    fan::vec2(-1, 1),
    fan::vec2(-1, 1)
  );

  fan::sys::MD_SCR_t md;

  
  int i = 0;
  while (i < 3) {
    fan::sys::MD_SCR_open(&md);
    fan::sys::MD_SCR_close(&md);
    i++;
  }


  loco_t::sprite_t::properties_t p;

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
  fan::image::image_info_t ii;
  ii.size = fan::sys::get_screen_resolution();
  ii.data = ptr;

  loco_t::image_t image;
  loco_t::image_load_properties_t lp;
  lp.format = loco_t::image_format::b8g8r8a8_unorm;
  lp.internal_format = fan::opengl::GL_RGBA;
  image = loco.image_load(ii, lp);
  p.image = image;
  p.position.z = 0;
  loco_t::shape_t shape = p;


  loco.set_vsync(false);
  uint32_t x = 0;
  loco.loop([&] {
    ptr = fan::sys::MD_SCR_read(&md);
    if (ptr) {
      ii.data = ptr;
      loco.image_unload(image);
      image = loco.image_load(ii, lp);
      shape.set_image(image);
    }
  });

  return 0;
}