// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_window
#define loco_context

#define loco_yuv420p
#include _FAN_PATH(graphics/loco.h)

constexpr uint32_t count = 1;

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  pile_t() {
    loco.open_matrices(
      &matrices,
      ortho_x,
      ortho_y
    );
   /* loco.get_window()->add_resize_callback(this, [](fan::window_t* w, const fan::vec2i& size, void* userptr) {
      pile_t* pile = (pile_t*)userptr;

      pile->viewport.set(pile->loco.get_context(), 0, size, w->get_size());
    });*/
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, loco.get_window()->get_size(), loco.get_window()->get_size());
  }

  loco_t loco;
  loco_t::matrices_t matrices;
  fan::opengl::viewport_t viewport;
  fan::opengl::cid_t cids[count];
};

int main() {

  pile_t* pile = new pile_t;

  loco_t::yuv420p_t::properties_t p;

  p.size = fan::vec2(1, 1);
  //p.block_properties.
  p.matrices = &pile->matrices;
  p.viewport = &pile->viewport;

  constexpr fan::vec2ui image_size = fan::vec2ui(3840, 1080);

  fan::string str;
  fan::io::file::read("output.yuv", &str);

  p.load_yuv(&pile->loco, (uint8_t*)str.data(), image_size);

  p.position = fan::vec2(0, 0);
  p.position.z = 0;
  pile->loco.yuv420p.push_back(&pile->cids[0], p);

  //fan::print(y);
  pile->loco.set_vsync(false);

  pile->loco.loop([&] {
    pile->loco.get_fps();
  });

  return 0;
}