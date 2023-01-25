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
#define loco_light
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
    viewport.set(loco.get_context(), 0, d.size, d.size);
      });
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, window_size, window_size);
  }

  loco_t loco;
  loco_t::matrices_t matrices;
  fan::graphics::viewport_t viewport;
  fan::graphics::cid_t cid[10];
};

int main() {
  pile_t* pile = new pile_t;

  pile->loco.lighting.ambient = fan::vec3(0.3, 0.3, 0.3);

  loco_t::sprite_t::properties_t p;

  p.size = fan::vec2(1);
  p.matrices = &pile->matrices;
  p.viewport = &pile->viewport;

  loco_t::image_t image;
  image.load(&pile->loco, "images/lighting.webp");
  p.image = &image;
  p.position = fan::vec2(0, 0);
  pile->loco.sprite.push_back(&pile->cid[0], p);

  loco_t::light_t::properties_t lp;
  lp.matrices = &pile->matrices;
  lp.viewport = &pile->viewport;
  lp.position = fan::vec3(0, 0, 1);
  lp.size = fan::vec2(0.3);
  lp.color = fan::color(1, 0.8, 0);
  pile->loco.light.push_back(&pile->cid[0], lp);

  //lp.size = fan::vec2(0.3);
  //lp.color = fan::color(0, 0.8, 2);
  //pile->loco.light.push_back(&pile->cid[0], lp);

  pile->loco.set_vsync(false);

  pile->loco.loop([&] {
    pile->loco.get_fps();
    pile->loco.light.set(&pile->cid[0], &loco_t::light_t::vi_t::position, pile->loco.get_mouse_position(pile->viewport));
  });

  return 0;
}