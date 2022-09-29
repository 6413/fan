// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(graphics/graphics.h)

#define loco_window
#define loco_context

#define loco_sprite
#include _FAN_PATH(graphics/loco.h)

constexpr uint32_t count = 1;

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  void open() {
    loco.open(loco_t::properties_t());
    fan::graphics::open_matrices(
      loco.get_context(),
      &matrices,
      ortho_x,
      ortho_y
    );
    loco.get_window()->add_resize_callback([&](fan::window_t*, const fan::vec2i& size) {
      viewport.set(loco.get_context(), 0, size, size);
    });
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, loco.get_window()->get_size(), loco.get_window()->get_size());
  }

  loco_t loco;
  fan::opengl::matrices_t matrices;
  fan::opengl::viewport_t viewport;
  fan::opengl::cid_t cids[count];
};

int main() {

  pile_t* pile = new pile_t;
  pile->open();

  loco_t::sprite_t::properties_t p;

  p.size = fan::vec2(1, 1);
  //p.block_properties.
  p.get_matrices() = &pile->matrices;
  p.get_viewport() = &pile->viewport;

  fan::sys::MD_SCR_t md;
  fan::sys::MD_SCR_open(&md);
  uint8_t* ptr = fan::sys::MD_SCR_read(&md);
  if (ptr == nullptr) {
    return 1;
  }
  fan::webp::image_info_t ii;
  ii.size = fan::vec2ui(1920, 1080);
  ii.data = ptr;

  fan::opengl::image_t image;
  fan::opengl::image_t::load_properties_t lp;
  lp.format = fan::opengl::GL_BGRA;
  lp.internal_format = fan::opengl::GL_RGBA;
  image.load(pile->loco.get_context(), ii, lp);
  p.get_image() = &image;
  p.position = fan::vec2(0, 0);
  p.position.z = 0;
  //p.color = 10;
  // p.color = fan::color((f32_t)i / count, (f32_t)i / count + 00.1, (f32_t)i / count);
  pile->loco.sprite.push_back(&pile->cids[0], p);

  pile->loco.set_vsync(false);
  uint32_t x = 0;
  pile->loco.loop([&] {
    ptr = fan::sys::MD_SCR_read(&md);
    ii.data = ptr;
    image.unload(pile->loco.get_context());
    image.load(pile->loco.get_context(), ii, lp);
    pile->loco.sprite.set_image(&pile->cids[0], &image);
    pile->loco.get_fps();
  });

  return 0;
}