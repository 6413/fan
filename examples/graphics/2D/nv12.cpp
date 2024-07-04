#include <fan/pch.h>

int main() {

  loco_t loco;

  loco.camera_set_ortho(
    loco.orthographic_camera.camera,
    fan::vec2(-1, 1),
    fan::vec2(-1, 1)
  );

  loco_t::universal_image_renderer_t::properties_t p;

  p.size = fan::vec2(1, 1);

  constexpr fan::vec2ui image_size = fan::vec2ui(512, 512);

  fan::string str;
  fan::io::file::read("durum2nv12", &str);

  p.position = fan::vec3(0, 0, 0);
  p.size = 1;

  loco_t::shape_t s = p;

  {
    void* d = str.data();

    void* datas[2];
    uint64_t offset = 0;
    datas[0] = d;
    datas[1] = (uint8_t*)d + image_size.multiply();

    s.reload(fan::pixel_format::nv12, datas, image_size);
  }

  loco.loop([&] {
    loco.get_fps();
  });

  return 0;
}