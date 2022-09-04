// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(graphics/graphics.h)

#define loco_window
#define loco_context

#define loco_letter
#include _FAN_PATH(graphics/loco.h)

constexpr uint32_t count = 10;

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
    loco.get_window()->add_resize_callback(this, [](fan::window_t* window, const fan::vec2i& size, void* userptr) {
      fan::vec2 window_size = window->get_size();
      fan::vec2 ratio = window_size / window_size.max();
      pile_t* pile = (pile_t*)userptr;
      pile->matrices.set_ortho(
        ortho_x * ratio.x, 
        ortho_y * ratio.y
      );
    });
    loco.get_window()->add_resize_callback(this, [](fan::window_t*, const fan::vec2i& size, void* userptr) {
      pile_t* pile = (pile_t*)userptr;

      pile->viewport.set(pile->loco.get_context(), 0, size, size );
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

  loco_t::letter_t::properties_t p;

  p.get_viewport() = &pile->viewport;
  p.get_matrices() = &pile->matrices;

  for (uint32_t i = 0; i < count; i++) {
    p.position = fan::vec2(fan::random::value_f32(-1, 1), fan::random::value_f32(-1, 1));
    p.color = fan::color(1, 0, f32_t(i) / count, 1);
    p.font_size = 0.5;
    std::string str = fan::random::string(1);
    std::wstring w(str.begin(), str.end());
    p.letter_id = pile->loco.font.decode_letter(w[0]);

    pile->loco.letter.push_back(&pile->cids[i], p);
  }

  fan::print(pile->loco.letter.get(&pile->cids[0], &loco_t::letter_t::instance_t::position));

  pile->loco.set_vsync(false);

  pile->loco.loop([&] {
    pile->loco.get_fps();
  });

  return 0;
}