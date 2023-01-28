// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

//#define loco_vulkan

//#define loco_wboit

#define loco_window
#define loco_context

//#define loco_post_process

#define loco_framebuffer

#define loco_rectangle
//#define loco_sprite
#include _FAN_PATH(graphics/loco.h)

constexpr uint32_t count = 5000;

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  pile_t() {
    loco.open_matrices(
      &matrices,
      ortho_x,
      ortho_y
    );
    loco.get_window()->add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      //viewport.set(loco.get_context(), 0, d.size, d.size);
    });
    viewport[0].open(loco.get_context());
    viewport[0].set(loco.get_context(), 0, loco.get_window()->get_size(), loco.get_window()->get_size());
    viewport[1].open(loco.get_context());
    viewport[1].set(loco.get_context(), 0, loco.get_window()->get_size() / 2, loco.get_window()->get_size());
  }

  loco_t loco;
  loco_t::matrices_t matrices;
  fan::graphics::viewport_t viewport[2];
  fan::graphics::cid_t cids[count];
};

int main() {
  //sizeof(loco_t)
  pile_t* pile = new pile_t;

  loco_t::rectangle_t::properties_t p;
  p.matrices = &pile->matrices;
  p.viewport = &pile->viewport[0];

  p.size = fan::vec2(0.05);

  p.color = fan::colors::red;
  p.color.a = 1;

  p.position = fan::vec3(0, 0, 0);
  pile->loco.rectangle.push_back(&pile->cids[0], p);
  p.position = fan::vec3(0.01, 0, 0);
  p.color = fan::colors::blue;
  p.color.a = 1;
  //p.viewport = &pile->viewport[1];
  pile->loco.rectangle.push_back(&pile->cids[0], p);

  pile->loco.set_vsync(false);
  

  pile->loco.loop([&] {

    pile->loco.get_fps();

  });

  return 0;
}