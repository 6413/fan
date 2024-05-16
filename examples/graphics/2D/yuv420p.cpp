#include <fan/pch.h>

int main() {

  loco_t loco;

  loco.camera_set_ortho(
    loco.orthographic_camera.camera,
    fan::vec2(-1, 1),
    fan::vec2(-1, 1)
  );

  loco_t::universal_image_renderer_t::properties_t p;

  //p.pixel_format = fan::pixel_format::yuv420p;
  p.size = fan::vec2(1, 1);

  constexpr fan::vec2ui image_size = fan::vec2ui(1920, 1080);

  fan::string str;
  fan::io::file::read("output1920.yuv", &str);

  //fan::string str2;
  //fan::io::file::read("output1920.yuv", &str2);

 // p.load_yuv(&pile->loco, (uint8_t*)str.data(), image_size);
  p.position = fan::vec3(0, 0, 0);
  p.position.z = 0;
  p.size = 1;

  loco_t::shape_t s = p;

  void* d = str.data();

  void* datas[3];
  uint64_t offset = 0;
  datas[0] = d;
  datas[1] = (uint8_t*)d + (offset += image_size.multiply());
  datas[2] = (uint8_t*)d + (offset += image_size.multiply() / 4);

  //pile->loco.yuv420p.reload_yuv(&pile->cids[0], datas, image_size);

  /*loco_t::shapes_t::sprite_t::properties_t sp;
  sp.position = fan::vec3(0, 0, 0);
  loco_t::image_t image;
  image.load(&pile->loco, "images/test.webp");
  sp.image = &image;
  sp.size = 0.1;
  sp.viewport = &pile->viewport;
  sp.camera = &pile->camera;
  pile->loco.sprite.push_back(&pile->cids[0], sp);*/

  //fan::print(y);

  auto data = str.data();
  //auto data2 = str2.data();


  {
    void* d = str.data();

    void* datas[3];
    uint64_t offset = 0;
    datas[0] = d;
    datas[1] = (uint8_t*)d + (offset += image_size.multiply());
    datas[2] = (uint8_t*)d + (offset += image_size.multiply() / 4);

    s.reload(fan::pixel_format::yuv420p, datas, image_size);
  }

  loco.loop([&] {
    loco.get_fps();


    //pile->loco.pixel_format_renderer.set_position(&pile->cids[0], pile->loco.get_mouse_position(pile->viewport));
  });

  return 0;
}