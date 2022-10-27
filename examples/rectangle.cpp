// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_vulkan

#define loco_window
#define loco_context

//#define loco_post_process

#define loco_rectangle
#include _FAN_PATH(graphics/loco.h)

constexpr uint32_t count = 1.0e+1;

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(0, 800);
  static constexpr fan::vec2 ortho_y = fan::vec2(0, 800);

  pile_t() {
    fan::graphics::open_matrices(
      loco.get_context(),
      &matrices,
      ortho_x,
      ortho_y
    );
    loco.get_window()->add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      //fan::vec2 window_size = window->get_size();
      viewport.set(loco.get_context(), 0, d.size, d.size);
    });
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, loco.get_window()->get_size(), loco.get_window()->get_size());
  }

  loco_t loco;
  fan::graphics::matrices_t matrices;
  fan::graphics::viewport_t viewport;
  fan::graphics::cid_t cids[count];
};

int main() {

  pile_t* pile = new pile_t;

  loco_t::rectangle_t::properties_t p;
  
  //p.block_properties.
  p.get_matrices() = &pile->matrices;
  p.get_viewport() = &pile->viewport;

  p.size = fan::vec2(50, 50);
  p.position = fan::vec2(400, 400);
  p.color = fan::colors::blue;
  for (uint32_t i = 0; i < 100000; i++)
  pile->loco.rectangle.push_back(&pile->cids[0], p);


  pile->loco.set_vsync(false);

  fan::vec2 suunta = fan::random::vec2(-1500, 1500);

  auto& rectangle = pile->loco.rectangle;

  auto& window = *pile->loco.get_window();
  
  pile->loco.loop([&] {
    pile->loco.get_fps();
    //rectangle.
   /* fan::vec2 sijainti = rectangle.get(&pile->cids[0], &loco_t::rectangle_t::instance_t::position);

    if (sijainti.x >= 750 || sijainti.x <= 50) {
      suunta.x *= -1;
    }
    if (sijainti.y >= 750 || sijainti.y <= 50) {
      suunta.y *= -1;
    }

    rectangle.set(
      &pile->cids[0],
      &loco_t::rectangle_t::instance_t::position,
      sijainti + suunta * window.get_delta_time()
    );*/

  });

  return 0;
}