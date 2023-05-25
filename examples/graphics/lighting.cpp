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
#define loco_light_sun
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

  pile->loco.lighting.ambient = fan::vec3(0.3, 0.3, 0.3);

  loco_t::sprite_t::properties_t p;

  p.size = fan::vec2(1);
  p.camera = &pile->camera;
  p.viewport = &pile->viewport;

  loco_t::image_t image;
  image.load("images/lighting.webp");

  loco_t::image_t image2;
  image2.load("images/brick.webp");
  p.image = &image;
  p.position = fan::vec3(0, 0, 0);
  p.color.a = 1;
  loco_t::shape_t s0 = p;
  p.position.x += 0.4;
  p.size = 0.2;
  p.position.z += 2;
  p.color.a = 1;
  p.image = &image2;
  loco_t::shape_t s1 = p;

  loco_t::light_t::properties_t lp;
  lp.camera = &pile->camera;
  lp.viewport = &pile->viewport;
  lp.position = fan::vec3(0, 0, 0);
  lp.size = 1;
  lp.color = fan::colors::yellow * 10;
  loco_t::shape_t l0 = lp;
  
  //for (uint32_t i = 0; i < 1000; i++) {
  //  lp.position = fan::random::vec2(-1, 1);
  //  lp.color = fan::random::color();
  //  lp.position.z = 0;
  //  pile->loco.light.push_back(&pile->cid[0], lp);
  //}

  //offset = vec4(view * vec4(vec2(tc[id] * get_instance().tc_size + get_instance().tc_position), 0, 1)).xy * 2;
  pile->loco.set_vsync(false);

  fan::vec3 camerapos = 0;


  pile->loco.get_window()->add_keys_callback([&](const auto& d) {
    if (d.key == fan::key_left) {
      camerapos.x -= 0.1;
      pile->camera.set_camera_position(camerapos);
    }
  if (d.key == fan::key_right) {
    camerapos.x += 0.1;
    pile->camera.set_camera_position(camerapos);
  }
    });

  pile->loco.loop([&] {
    pile->loco.get_fps();
  /*if (c.finished()) {
    lp.color = fan::random::color();
      lp.size = 0.2;
      lp.position = pile->loco.get_mouse_position(pile->viewport);
      pile->loco.light.push_back(&pile->cid[1], lp);
      c.restart();
  }*/

  #if 1
//  pile->loco.light.set(&pile->cid[0], &loco_t::light_t::vi_t::position, pile->loco.get_mouse_position(pile->viewport));
  #else
  #endif
  #if 0
  pile->loco.light.set(&pile->cid[0], &loco_t::light_t::vi_t::position, pile->loco.get_mouse_position(pile->viewport));
  #endif
  });

  return 0;
}