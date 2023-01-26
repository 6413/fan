// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

//#define loco_vulkan

#define loco_window
#define loco_context

//#define loco_rectangle
#define loco_sprite
#include _FAN_PATH(graphics/loco.h)

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  pile_t() {
    fan::vec2 window_size = loco.get_window()->get_size();
    loco.open_matrices(
      &matrices,
      ortho_x,
      ortho_y
    );
    loco.get_window()->add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      fan::vec2 window_size = d.size;
      fan::vec2 ratio = window_size / window_size.max();
      std::swap(ratio.x, ratio.y);
      //matrices.set_ortho(
      //  ortho_x * ratio.x, 
      //  ortho_y * ratio.y
      //);
      viewport.set(loco.get_context(), 0, d.size, d.size);
    });
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, window_size, window_size);
  }

  loco_t loco;
  loco_t::matrices_t matrices;
  fan::graphics::viewport_t viewport;
  fan::graphics::cid_t cid[(unsigned long long)1e+7];
};

int main() {

  pile_t* pile = new pile_t;

  loco_t::sprite_t::properties_t p;

  p.size = fan::vec2(1);
  p.matrices = &pile->matrices;
  p.viewport = &pile->viewport;

  loco_t::image_t image;
  image.load(&pile->loco, "images/brick.webp");
  loco_t::image_t image2;
  image2.load(&pile->loco, "images/brick.webp");
  p.image = &image;
  p.position = fan::vec2(0, 0);
  pile->loco.sprite.push_back(&pile->cid[0], p);
  //pile->loco.sprite.set_image(&pile->cid[0], &image2);
  //p.image = &image;
  //p.position = fan::vec3(fan::random::vec2(-1, 1), 0);
  //pile->loco.sprite.push_back(&pile->cid[0], p);
  //p.position = fan::vec3(-0.1, -0.1, 0);

  ////pile->loco.sprite.push_back(&pile->cid[1], p);
  //// 
  ////p.position = fan.;
  //
  //pile->loco.sprite.push_back(&pile->cid[0], p);
  //p.image = &image2;
  //p.position = fan::vec3(-0.2, -0.2, 3);
  //pile->loco.sprite.push_back(&pile->cid[1], p);

  //p.image = &image2;
  //p.position = fan::vec2(0.3, 0.3);
  //pile->loco.sprite.push_back(&pile->cid[2], p);

  /*for (uint32_t i = 0; i < 1; i++) {
    
  }*/
  pile->loco.set_vsync(false);
  //uint8_t* ptr = new uint8_t[640 * 640 * 4];
  uint32_t x = 0;

  fan::time::clock cc;
  pile->loco.loop([&] {
    //fan::print(cc.elapsed());
    //cc.start();
    //pile->loco.get_context()->set_depth_test(true);
   // if (x < count) 
    //pile->loco.sprite.erase(&pile->cid[x++]);
    //image2.unload(&pile->loco);
    //image2.load(&pile->loco, "images/asteroid.webp");
    //fan::webp::image_info_t ii;
    //ii.data = ptr;
    //ii.size = fan::vec2(640, 640);
    //image2.reload_pixels(&pile->loco, ii);
    pile->loco.get_fps(); 
  });

  return 0;
}