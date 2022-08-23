// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(graphics/graphics.h)

#define loco_window
#define loco_context

#define loco_sprite
#define loco_post_process
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
      loco.get_window()->get_size(),
      ortho_x,
      ortho_y,
      1
    );
    loco.get_window()->add_resize_callback(this, [](fan::window_t* window, const fan::vec2i& size, void* userptr) {
      fan::vec2 window_size = window->get_size();
      fan::vec2 ratio = window_size / window_size.max();
      std::swap(ratio.x, ratio.y);
      pile_t* pile = (pile_t*)userptr;
      pile->matrices.set_ortho(
        ortho_x * ratio.x,
        ortho_y * ratio.y,
        1
      );
    });
    loco.get_window()->add_resize_callback(this, [](fan::window_t*, const fan::vec2i& size, void* userptr) {
      pile_t* pile = (pile_t*)userptr;

      pile->viewport.set_viewport(pile->loco.get_context(), 0, size, pile->loco.get_window()->get_size());
    });
    viewport.open(loco.get_context());
    viewport.set_viewport(loco.get_context(), 0, loco.get_window()->get_size(), loco.get_window()->get_size());
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

  //p.block_properties.
  p.matrices = &pile->matrices;
  p.viewport = &pile->viewport;

  fan::opengl::image_t image;
  image.load(pile->loco.get_context(), "images/grass.webp");
  p.image = &image;
  p.size = 1;
  p.position = fan::vec2(0, 0);
  p.position.z = 0;
  pile->loco.sprite.push_back(&pile->cids[0], p);

  pile->loco.post_process.push(&pile->viewport, &pile->matrices);

  pile->loco.set_vsync(false);
  uint32_t x = 0;

  pile->loco.loop([&] {
    pile->loco.get_fps();
  });

  return 0;
}