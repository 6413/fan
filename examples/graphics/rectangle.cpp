// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 3
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_vulkan

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

  //p.position = fan::vec3(-0.5, -0.5, 0);
  p.color = fan::colors::red;
  //p.color.a = 0.5;

  //for (uint32_t i = 0; i < count; i++) {
  //  p.position = fan::vec3(fan::random::vec2(-1, 1), i);
  //  pile->loco.rectangle.push_back(&pile->cids[i], p);
  //}

      p.position = fan::vec3(0, 0, 2);
    pile->loco.rectangle.push_back(&pile->cids[0], p);
    p.viewport = &pile->viewport[1];
    p.position = fan::vec3(0, 0, 0);
    pile->loco.rectangle.push_back(&pile->cids[0], p);
    //pile->loco.
  //p.position = fan::vec2(0, 0);
  //p.color = fan::colors::white;
  //pile->loco.rectangle.push_back(&pile->cids[2], p);

  //pile->matrices.set_ortho();
  //pile->loco.rectangle.m_shader.

  //for (uint32_t i = 0; i < count; i++) {
  //  p.position = fan::random::vec2(0, 1920);
  //  pile->loco.rectangle.push_back(&pile->cids[i], p);
  //}

    //{

    //  loco_t::sprite_t::properties_t p;

    //  p.size = fan::vec2(1);
    //  p.matrices = &pile->matrices;
    //  p.viewport = &pile->viewport;

    //  loco_t::image_t image;
    //  image.load(&pile->loco, "images/test.webp");
    //  p.image = &image;
    //  p.position = fan::vec3(0, 0, 1);
    //  pile->loco.sprite.push_back(&pile->cids[0], p);
    //}


  pile->loco.set_vsync(false);
  

  //MapVirtualKey()

  pile->loco.loop([&] {

    pile->loco.get_fps();

  });

  return 0;
}