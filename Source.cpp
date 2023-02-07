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
    //fan::vec2 ratio = window_size / window_size.max();
    //std::swap(ratio.x, ratio.y);
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

  loco_t::image_t image[3];
  image[0].load(&pile->loco, "images/hull.webp");
  image[1].load(&pile->loco, "images/tire_left.webp");
  image[2].load(&pile->loco, "images/tire_right.webp");
  p.image = &image[0];
  p.size = image[0].size / 2000;
  p.position = fan::vec2(1 - p.size.x, 0);

  pile->loco.sprite.push_back(&pile->cid[0], p);

  p.image = &image[1];
  p.size = image[1].size / 2000;
  p.position += fan::vec3(-0.124, 0.115, 1);
  pile->loco.sprite.push_back(&pile->cid[1], p);

  p.size = image[2].size / 2000;
  p.position = fan::vec2(1 - image[0].size.x / 2000, 0);
  p.position += fan::vec3(0.108, 0.115, 2);
  pile->loco.sprite.push_back(&pile->cid[2], p);

  // pile->loco.set_vsync(false);

  f32_t x = 0;

  pile->loco.get_window()->add_keys_callback([&](const auto& d) {
    if (d.key != fan::key_a) {
      return;
    }
  if (d.state == fan::keyboard_state::release) {
    return;
  }

  auto p = pile->loco.sprite.get(&pile->cid[0], &loco_t::sprite_t::vi_t::position);
  p.x -= pile->loco.get_delta_time() * 1;
  pile->loco.sprite.set(&pile->cid[0], &loco_t::sprite_t::vi_t::position, p);

  p = pile->loco.sprite.get(&pile->cid[1], &loco_t::sprite_t::vi_t::position);
  p.x -= pile->loco.get_delta_time() * 1;
  pile->loco.sprite.set(&pile->cid[1], &loco_t::sprite_t::vi_t::position, p);

  p = pile->loco.sprite.get(&pile->cid[2], &loco_t::sprite_t::vi_t::position);
  p.x -= pile->loco.get_delta_time() * 1;
  pile->loco.sprite.set(&pile->cid[2], &loco_t::sprite_t::vi_t::position, p);


  pile->loco.sprite.set(&pile->cid[2], &loco_t::sprite_t::vi_t::angle, x);

  pile->loco.sprite.set(&pile->cid[1], &loco_t::sprite_t::vi_t::angle, x);
  x += pile->loco.get_delta_time() * 50;
    });


  pile->loco.loop([&] {
    pile->loco.get_fps();


    });

  return 0;
}