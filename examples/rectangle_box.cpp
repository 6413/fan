// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(graphics/graphics.h)

#define loco_window
#define loco_context

#define loco_box
#include _FAN_PATH(graphics/loco.h)

constexpr uint32_t count = 1;

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  void open() {
    loco.open(loco_t::properties_t());
    fan::graphics::open_matrices(
      loco.get_context(),
      &matrices[0],
      loco.get_window()->get_size(),
      ortho_x,
      ortho_y
    );
    fan::graphics::open_matrices(
      loco.get_context(),
      &matrices[1],
      loco.get_window()->get_size(),
      fan::vec2(0, 800),
      fan::vec2(0, 600)
    );
    loco.get_window()->add_resize_callback(this, [](fan::window_t* window, const fan::vec2i& size, void* userptr) {
      fan::vec2 window_size = window->get_size();
      fan::vec2 ratio = window_size / window_size.max();
      std::swap(ratio.x, ratio.y);
      pile_t* pile = (pile_t*)userptr;
      pile->matrices[0].set_ortho(
        ortho_x * ratio.x, 
        ortho_y * ratio.y
      );
      pile->matrices[1].set_ortho(
        ortho_x * ratio.x, 
        ortho_y * ratio.y
      );
      });
    loco.get_window()->add_resize_callback(this, [](fan::window_t*, const fan::vec2i& size, void* userptr) {
      pile_t* pile = (pile_t*)userptr;

      pile->viewport.set_viewport(pile->loco.get_context(), 0, size);
      });
    viewport.open(loco.get_context(), 0, loco.get_window()->get_size());
  }

  loco_t loco;
  fan::opengl::matrices_t matrices[2];
  fan::opengl::viewport_t viewport;
  fan::opengl::cid_t cids[count];
};

int main() {

  pile_t* pile = new pile_t;
  pile->open();

  loco_t::rectangle_box_t::properties_t p;

  p.viewport = &pile->viewport;
  p.matrices = &pile->matrices[0];

  p.size = fan::vec2(0.3, 0.1);

  for (uint32_t i = 0; i < count; i++) {
    p.position = fan::vec2(0, 0);
    p.theme = fan_2d::graphics::gui::themes::deep_red();
    pile->loco.box.push_back(&pile->loco, &pile->cids[i], p);
  }

  pile->loco.set_vsync(false);
  uint32_t x = 0;
  while(pile->loco.window_open(pile->loco.process_frame())) {
    /* if(x < count) {
    pile->loco.rectangle.erase(&pile->loco, &pile->cids[x]);
    x++;
    }*/
    pile->loco.get_fps();
  }

  return 0;
}