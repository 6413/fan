// Creates window, opengl context

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

//#define loco_vulkan

#define loco_window
#define loco_context
#include _FAN_PATH(graphics/loco.h)

struct pile_t {
  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  void open() {
    loco.open_matrices(
      &matrices,
      ortho_x,
      ortho_y
    );
    //loco.get_window()->add_resize_callback([&](fan::window_t* window, const fan::vec2i& size) {
    //  fan::vec2 window_size = window->get_size();
    //  fan::vec2 ratio = window_size / window_size.max();
    //  std::swap(ratio.x, ratio.y);
    //  matrices.set_ortho(
    //    ortho_x * ratio.x,
    //    ortho_y * ratio.y
    //  );
    //  viewport.set(loco.get_context(), 0, size, size);
    //});*/

    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, loco.get_window()->get_size(), loco.get_window()->get_size());
    loco.get_window()->add_keys_callback([](const auto& c) {
      fan::print(std::hex, c.scancode);
     });
  }

  loco_t loco;
  loco_t::matrices_t matrices;
  fan::graphics::viewport_t viewport;
};

int main() {
  pile_t* pile = new pile_t;
  pile->open();
  pile->loco.get_context()->set_vsync(pile->loco.get_window(), 0);

  pile->loco.get_window()->add_keys_callback([&](const auto& d) {
    if (d.state != fan::keyboard_state::press) {
      return;
    }
    static int x = 0;
    pile->loco.get_window()->set_flag_value<fan::window_t::flags::no_mouse>(x % 2);
    x++;
  });

  pile->loco.loop([&] {
    
  });

  return 0;
}