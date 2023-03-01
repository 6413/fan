// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 3
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

//#define loco_vulkan

#define loco_window
#define loco_context


#define loco_sprite
#define loco_yuv420p
//#define loco_nv12
//#define loco_cuda
#define loco_pixel_format_renderer
#include _FAN_PATH(graphics/loco.h)

constexpr uint32_t count = 1;

struct pile_t {
  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  pile_t() {
    loco.open_camera(
      &camera,
      ortho_x,
      ortho_y
    );
     loco.get_window()->add_resize_callback([&](const auto& data) {
       //pile_t* pile = (pile_t*)userptr;

       viewport.set(loco.get_context(), 0, data.size, data.size);
     });
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, loco.get_window()->get_size(), loco.get_window()->get_size());
  }

  loco_t loco{ loco_t::properties_t{.vsync = false } };
  loco_t::camera_t camera;
  fan::graphics::viewport_t viewport;
  fan::graphics::cid_t cids[count];
};

int main() {

  pile_t* pile = new pile_t;

  loco_t::pixel_format_renderer_t::properties_t p;

  //p.pixel_format = fan::pixel_format::yuv420p;
  p.size = fan::vec2(1, 1);
  //p.block_properties.
  p.camera = &pile->camera;
  p.viewport = &pile->viewport;

  constexpr fan::vec2ui image_size = fan::vec2ui(1920, 1080);

  fan::string str;
  fan::io::file::read("output1920.yuv", &str);

  //fan::string str2;
  //fan::io::file::read("output1920.yuv", &str2);

 // p.load_yuv(&pile->loco, (uint8_t*)str.data(), image_size);

  p.position = fan::vec3(0, 0, 0);
  p.position.z = 0;
  p.size = 1;
  pile->loco.pixel_format_renderer.push_back(&pile->cids[0], p);

  void* d = str.data();

  void* datas[3];
  uint64_t offset = 0;
  datas[0] = d;
  datas[1] = (uint8_t*)d + (offset += image_size.multiply());
  datas[2] = (uint8_t*)d + (offset += image_size.multiply() / 4);

  //pile->loco.yuv420p.reload_yuv(&pile->cids[0], datas, image_size);

  /*loco_t::sprite_t::properties_t sp;
  sp.position = fan::vec3(0, 0, 0);
  loco_t::image_t image;
  image.load(&pile->loco, "images/test.webp");
  sp.image = &image;
  sp.size = 0.1;
  sp.viewport = &pile->viewport;
  sp.camera = &pile->camera;
  pile->loco.sprite.push_back(&pile->cids[0], sp);*/

  //fan::print(y);
  pile->loco.set_vsync(false);

  auto data = str.data();
  //auto data2 = str2.data();




  pile->loco.loop([&] {
    pile->loco.get_fps();

    void* d = str.data();
    
    void* datas[3];
    uint64_t offset = 0;
    datas[0] = d;
    datas[1] = (uint8_t*)d + (offset += image_size.multiply());
    datas[2] = (uint8_t*)d + (offset += image_size.multiply() / 4);

    pile->loco.pixel_format_renderer.reload(&pile->cids[0], fan::pixel_format::yuv420p, datas, image_size);
    //pile->loco.pixel_format_renderer.set_position(&pile->cids[0], pile->loco.get_mouse_position(pile->viewport));
  });

  return 0;
}