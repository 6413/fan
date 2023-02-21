// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

//#define loco_vulkan

#define loco_window
#define loco_context

#define loco_letter
#define loco_text
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
    loco.get_window()->add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      fan::vec2 window_size = d.size;
      fan::vec2 ratio = window_size / window_size.max();
      matrices.set_ortho(
        &loco,
        ortho_x * ratio.x, 
        ortho_y * ratio.y
      );
      viewport.set(loco.get_context(), 0, window_size, window_size);
    });
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, loco.get_window()->get_size(), loco.get_window()->get_size());
  }

  loco_t loco;
  loco_t::matrices_t matrices;
  fan::graphics::viewport_t viewport;
  fan::graphics::cid_t cid[count];
};

int main() {

  pile_t* pile = new pile_t;

  loco_t::text_t::properties_t p;

  p.matrices = &pile->matrices;
  p.viewport = &pile->viewport;

  p.font_size = 0.05;
  p.text = "01234";
  for (uint32_t i = 0; i < count; i++) {
    if (!i) {
      p.color = fan::colors::red;
    }
    else {
      p.color = fan::colors::white;
    }
    p.position = fan::random::vec2(-1, 1);
    //p.text = fan::random::string(5);
    
    pile->loco.text.push_back(&pile->cid[i], p);
  }
  pile->loco.text.erase(&pile->cid[0]);
  p.text = "56789";
  pile->loco.text.push_back(&pile->cid[0], p);
  pile->loco.text.set_text(&pile->cid[0], "56789");

  pile->loco.text.set(&pile->cid[0], &loco_t::text_t::vi_t::color, fan::color(1, 0, 0, 0.3));

  pile->loco.set_vsync(false);

  pile->loco.loop([&] {
    pile->loco.get_fps();
  });

  return 0;
}