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

#define loco_rectangle
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
  fan::graphics::cid_t cid[1];
};

int main() {

  fan::time::clock c;
  c.start();

  pile_t* pile = new pile_t;

  loco_t::sprite_t::properties_t p;

  p.size = fan::vec2(0.1);
  p.matrices = &pile->matrices;
  p.viewport = &pile->viewport;

  loco_t::image_t images[2];
  images[0].load(&pile->loco, "images/brick.webp");
  images[1].load(&pile->loco, "images/brick_inverted.webp");
  p.image = images;
  p.position = fan::vec3(0, 0, 2);
  pile->loco.sprite.push_back(&pile->cid[0], p);

  loco_t::rectangle_t::properties_t rp;
  rp.matrices = &pile->matrices;
  rp.viewport = &pile->viewport;

  rp.position = fan::vec3(0.1, 0.1, 1);
  rp.size = fan::vec2(0.1);

  rp.color = fan::colors::red;
  rp.color.a = 1;

  pile->loco.rectangle.push_back(&pile->cid[0], rp);

  p.position = fan::vec3(0.15, 0, 0);
  p.image = images + 1;
  pile->loco.sprite.push_back(&pile->cid[0], p);

  p.position = fan::vec3(0.05, 0.1, 0);
  p.image = images + 1;
  pile->loco.sprite.push_back(&pile->cid[0], p);

  pile->loco.set_vsync(false);

  pile->loco.loop([&] {
    pile->loco.get_fps();
  });

  return 0;
}