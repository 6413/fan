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
    loco.open_camera(
      &camera,
      ortho_x,
      ortho_y
    );
    loco.get_window()->add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      fan::vec2 window_size = d.size;
      //fan::vec2 ratio = window_size / window_size.max();
      //std::swap(ratio.x, ratio.y);
      //camera.set_ortho(
      //  ortho_x * ratio.x, 
      //  ortho_y * ratio.y
      //);
      viewport.set(0, d.size, d.size);
    });
    viewport.open();
    viewport.set(0, window_size, window_size);
  }

  loco_t loco;
  loco_t::camera_t camera;
  fan::graphics::viewport_t viewport;
};

pile_t* pile = new pile_t;

int main() {

  loco_t::image_t image;
  image.load("images/tire.webp");

  loco_t::unlit_sprite_t::properties_t p;

  p.size = fan::vec2(0.1);
  p.camera = &pile->camera;
  p.viewport = &pile->viewport;
  p.image = &image;


  loco_t::shape_t id3[100];
  p.position = -1;
  for (uint32_t i = 0; i < 100; ++i) {
    p.position = fan::random::vec2(-1, 1);
    id3[i] = p;
  }

  fan::vec2 v(-1, -1);

  f32_t angle = 0;

  id3->set_image((loco_t::image_t*)0);

  pile->loco.loop([&] {
    angle += pile->loco.get_delta_time() * 2;
    for (uint32_t i = 0; i < 100; ++i) {
      id3[i].set_angle(angle);
    }
    //v = v + 0.1 * pile->loco.get_window()->get_delta_time();
    //id3.set_position(v);
    //id3.set_angle(angle);

    //id3.set_position(pile->loco.get_mouse_position(pile->camera, pile->viewport));

    pile->loco.get_fps(); 
  });

  return 0;
}