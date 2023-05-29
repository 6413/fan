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
  image.load("images/1.WEBP");

  loco_t::unlit_sprite_t::properties_t p;
  loco_t::sprite_t::properties_t pp;

  p.size = fan::vec2(1);
  p.camera = &pile->camera;
  p.viewport = &pile->viewport;
  
  pp.size = fan::vec2(1);
  pp.camera = &pile->camera;
  pp.viewport = &pile->viewport;  
  pp.image = &image;
  
  pp.position = fan::vec2(0.75, 0.75);
  pp.size = 0.25;
  pp.color.a = 0.49;
  pp.blending = true;

  

  //p.image = &image;
  //p.blending = true;
  //p.size = 0.5;
  //p.position = fan::vec2(-0.5, -0.5);
  //loco_t::shape_t id = p;
  //pile->loco.set_vsync(false);


  //p.position = fan::vec2(0.25, 0.25);
  //p.size = 0.25;
  //p.blending = false;
  //loco_t::shape_t id2 = p;

  ////id2.erase();
  ////id3.erase();
  ////id.erase();
  //id2.set_depth(3);

  //pile->loco.process_loop([] {});
  fan::print(loco_bdbt_usage(&gloco->bdbt));
  loco_t::shape_t id3 = pp;
  id3.erase();
  fan::print(loco_bdbt_usage(&gloco->bdbt));
  id3 = pp;
  id3.erase();
  fan::print(loco_bdbt_usage(&gloco->bdbt));

  pile->loco.loop([&] {
    pile->loco.get_fps(); 
  });

  return 0;
}