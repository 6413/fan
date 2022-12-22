// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_vulkan

#define loco_window
#define loco_context

#define loco_yuv420p
//#define loco_sprite
#include _FAN_PATH(graphics/loco.h)

constexpr uint32_t count = 1;

struct pile_t {
  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  pile_t() {
    loco.open_matrices(
      &matrices,
      ortho_x,
      ortho_y
    );
    /* loco.get_window()->add_resize_callback(this, [](fan::window_t* w, const fan::vec2i& size, void* userptr) {
       pile_t* pile = (pile_t*)userptr;

       pile->viewport.set(pile->loco.get_context(), 0, size, w->get_size());
     });*/
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, loco.get_window()->get_size(), loco.get_window()->get_size());
  }

  loco_t loco;
  loco_t::matrices_t matrices;
  fan::graphics::viewport_t viewport;
  fan::graphics::cid_t cids[count];
};

int main() {

  pile_t* pile = new pile_t;

  loco_t::yuv420p_t::properties_t p;

  p.size = fan::vec2(1, 1);
  //p.block_properties.
  p.matrices = &pile->matrices;
  p.viewport = &pile->viewport;

  constexpr fan::vec2ui image_size = fan::vec2ui(1920, 1080);

  fan::string str;
  fan::io::file::read("output1920_2.yuv", &str);

  fan::string str2;
  fan::io::file::read("output1920.yuv", &str2);

  p.load_yuv(&pile->loco, (uint8_t*)str.data(), image_size);

  p.position = fan::vec3(0, 0, 0);
  p.position.z = 0;
  p.size = 1;
  pile->loco.yuv420p.push_back(&pile->cids[0], p);

  /*loco_t::sprite_t::properties_t sp;
  sp.position = fan::vec3(0, 0, 0);
  loco_t::image_t image;
  image.load(&pile->loco, "images/test.webp");
  sp.image = &image;
  sp.size = 0.1;
  sp.viewport = &pile->viewport;
  sp.matrices = &pile->matrices;
  pile->loco.sprite.push_back(&pile->cids[0], sp);*/

  //fan::print(y);
  pile->loco.set_vsync(false);

  auto data = str.data();
  auto data2 = str2.data();




  pile->loco.loop([&] {
    pile->loco.get_fps();

    void* d = data2;

    void* datas[3];
    uint64_t offset = 0;
    datas[0] = d;
    datas[1] = (uint8_t*)d + (offset += image_size.multiply());
    datas[2] = (uint8_t*)d + (offset += image_size.multiply() / 4);

    pile->loco.yuv420p.reload_yuv(&pile->cids[0], datas, image_size);

  });

  return 0;
}