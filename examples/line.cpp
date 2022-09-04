// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(graphics/graphics.h)

#define loco_window
#define loco_context

#define loco_line
#include _FAN_PATH(graphics/loco.h)

constexpr uint32_t count = 1e+4;

struct pile_t {

  void open() {
    loco.open(loco_t::properties_t());
    auto window_size = loco.get_window()->get_size();
    fan::graphics::open_matrices(
      loco.get_context(),
      &matrices,
      fan::vec2(0, window_size.x),
      fan::vec2(0, window_size.y)
    );
    loco.get_window()->add_resize_callback(this, [](fan::window_t* window, const fan::vec2i& size, void* userptr) {
      fan::vec2 window_size = window->get_size();
      fan::vec2 ratio = window_size / window_size.max();
      pile_t* pile = (pile_t*)userptr;
      pile->matrices.set_ortho(
        fan::vec2(0, window_size.x) * ratio.x, 
        fan::vec2(0, window_size.y) * ratio.y
      );
      pile->viewport.set(pile->loco.get_context(), 0, size, size);
    });
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, window_size, window_size);
  }

  loco_t loco;
  fan::opengl::matrices_t matrices;
  fan::opengl::viewport_t viewport;
  fan::opengl::cid_t cids[count];
};

int main() {

  pile_t* pile = new pile_t;
  pile->open();

  loco_t::line_t::properties_t p;

  //p.block_properties.
  p.get_matrices() = &pile->matrices;
  p.get_viewport() = &pile->viewport;

  fan::time::clock c; 
  c.start();
  for (uint32_t i = 0; i < count; i++) {
    p.src = fan::random::vec2(0, 800);
    p.dst = fan::random::vec2(0, 800);
    p.color = fan::random::color();
    pile->loco.line.push_back(&pile->cids[i], p);
  }

  fan::print((f32_t)c.elapsed() / 1e+9);

  pile->loco.set_vsync(false);

  pile->loco.loop([&] {
    for (uint32_t i = 0; i < count; i++) {
      pile->loco.line.set(&pile->cids[i], &loco_t::line_t::instance_t::dst, pile->loco.get_mouse_position());
    }
    pile->loco.get_fps();
  });

  return 0;
}