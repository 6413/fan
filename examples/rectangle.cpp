// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define WITCH_INCLUDE_PATH C:/libs/WITCH
#include _INCLUDE_TOKEN(WITCH_INCLUDE_PATH, WITCH.h)
#define _WITCH_PATH(p0) <WITCH_INCLUDE_PATH/p0>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(graphics/graphics.h)

#define loco_window
#define loco_context

#define loco_rectangle
#define loco_sprite
#include _FAN_PATH(graphics/loco.h)

#define ETC_BCOL_set_prefix BCOL
#define ETC_BCOL_set_DynamicDeltaFunction \
  ObjectData->VelocityX /= 1 + delta * 2; \
  ObjectData->VelocityY /= 1 + delta * 1; \
  ObjectData->VelocityY += delta * 500;


#include _WITCH_PATH(ETC/BCOL/BCOL.h)

struct pile_t {
  
  static constexpr fan::vec2 ortho_x = fan::vec2(0, 800);
  static constexpr fan::vec2 ortho_y = fan::vec2(0, 800);

  void open() {
    loco.open(loco_t::properties_t());
    fan::vec2 window_size = loco.get_window()->get_size();
    fan::graphics::open_matrices(
      loco.get_context(),
      &matrices,
      ortho_x,
      ortho_y
    );
    loco.get_window()->add_resize_callback([&](fan::window_t* window, const fan::vec2i& size) {
      fan::vec2 window_size = window->get_size();
      fan::vec2 ratio = window_size / window_size.max();
      std::swap(ratio.x, ratio.y);
      matrices.set_ortho(
        ortho_x * ratio.x, 
        ortho_y * ratio.y
      );
    });
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, window_size, window_size);
  }
  fan::opengl::matrices_t matrices;
  fan::opengl::viewport_t viewport;

  loco_t loco;
  fan::opengl::cid_t cid;
};

int main() {

  pile_t* pile = new pile_t;
  pile->open();

  loco_t::sprite_t::properties_t p;

  //p.block_properties.
  p.get_matrices() = &pile->matrices;
  p.get_viewport() = &pile->viewport;

  p.position = fan::vec2(400, 400);
  p.size = 100;

  fan::opengl::image_t image;
  image.load(pile->loco.get_context(), "smile.webp");

  fan::opengl::image_t image2;
  image2.load(pile->loco.get_context(), "images/asteroid.webp");

  p.get_image() = &image;
  pile->loco.sprite.push_back(&pile->cid, p);

  auto window = pile->loco.get_window();
 /* window->add_keys_callback([&](fan::window_t*, uint16_t key, fan::key_state key_state) {
    if (key_state != fan::key_state::press) {
      return;
    }

  });*/

  pile->loco.set_vsync(false);

  float nopeus = 250;

  pile->loco.loop([&] {
    pile->loco.get_fps();

    fan::vec2 suunta = 0;

    if (window->key_pressed(fan::key_w)) {
      suunta.y -= nopeus;
    }
    if (window->key_pressed(fan::key_a)) {
      suunta.x -= nopeus;
    }
    if (window->key_pressed(fan::key_s)) {
      suunta.y += nopeus;
    }
    if (window->key_pressed(fan::key_d)) {
      suunta.x += nopeus;
    }

    auto nelio_sijainti = pile->loco.sprite.get(
      &pile->cid, 
      &loco_t::sprite_t::instance_t::position
    );
    pile->loco.sprite.set(
      &pile->cid,
      &loco_t::sprite_t::instance_t::position,
      nelio_sijainti + suunta * window->get_delta_time()
    );

    if (rand() & 1) {
      pile->loco.sprite.set_image(&pile->cid, &image2);
    }
    else {
      pile->loco.sprite.set_image(&pile->cid, &image);
    }

  });
  return 0;
}