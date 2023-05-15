// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)
#define loco_window
#define loco_context

#define loco_sprite
#define loco_rectangle
#define loco_responsive_text
#define loco_text
#include _FAN_PATH(graphics/loco.h)

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(0, 800);
  static constexpr fan::vec2 ortho_y = fan::vec2(0, 800);

  pile_t() {
    loco.open_camera(
      &camera,
      ortho_x,
      ortho_y
    );
    loco.get_window()->add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      viewport.set(0, d.size, d.size);
    });
    viewport.open();
    viewport.set(0, loco.get_window()->get_size(), loco.get_window()->get_size());
  }

  loco_t loco;
  loco_t::camera_t camera;
  loco_t::viewport_t viewport;
};

int main() {
  pile_t* pile = new pile_t;

  loco_t::image_t image;
  image.load("images/brick.webp");

  loco_t::sprite_t::properties_t pp;

  pp.size = fan::vec2(1);
  pp.camera = &pile->camera;
  pp.viewport = &pile->viewport;  
  pp.image = &image;
  
  pp.position = fan::vec2(400, 400);
  pp.size = fan::vec2(100, 100);
  pp.color.a = 1;
  pp.blending = true;

  loco_t::rectangle_t::properties_t rp;
  rp.camera = &pile->camera;
  rp.viewport = &pile->viewport;

  rp.position = fan::vec3(400, 400, 0);
  rp.size = fan::vec2(100);
  rp.color = fan::colors::red;

  //loco_t::shape_t r0 = p;
  //p.position = fan::vec3(0.1, 0, 1);
  //p.color = fan::colors::blue;

  //loco_t::shape_t r1 = p;
  //loco_t::shape_t r2;

  //pile->loco.set_vsync(false);
  
  loco_t::text_t::properties_t tp;
  tp.text = "test";
  tp.font_size = 0.49;
  auto p = loco_t::responsive_text_custom_t::make_properties(pp, tp);
  loco_t::responsive_text_custom_t responsive_box = p;

  /*{
    uint32_t i = 0;
    while (responsive_box.does_text_fit(std::to_string(i))) {
      responsive_box.push_back(std::to_string(i));
      i++;
    }
  }*/


  //fan::vec2
//  responsive_box.set_size(fan::vec2(1, 100));

  gloco->get_window()->add_keys_callback([&](const auto& d) {
    if (d.key == fan::key_up) {
      pp.size.x += 1;
      responsive_box.set_size(pp.size);
    }
    if (d.key == fan::key_down) {
      pp.size.x -= 1;
      responsive_box.set_size(pp.size);
    }
  });

  pile->loco.loop([&] {
    pile->loco.get_fps();

    //r0.set_position(pile->loco.get_mouse_position(pile->camera, pile->viewport));
  });

  return 0;
}