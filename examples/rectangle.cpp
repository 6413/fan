// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 1
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(graphics/graphics.h)

#define loco_window
#define loco_context

#define loco_rectangle
#include _FAN_PATH(graphics/loco.h)

constexpr uint32_t count = 1000;

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
      ortho_y
    );
    loco.get_window()->add_resize_callback(this, [](fan::window_t* window, const fan::vec2i& size, void* userptr) {
      fan::vec2 window_size = window->get_size();
      fan::vec2 ratio = window_size / window_size.max();
      std::swap(ratio.x, ratio.y);
      pile_t* pile = (pile_t*)userptr;
      pile->matrices.set_ortho(
        ortho_x * ratio.x, 
        ortho_y * ratio.y
      );
    });
  }

  loco_t loco;
  fan::opengl::matrices_t matrices;
  fan::opengl::cid_t cids[count];
};

int main() {

  pile_t pile;
  pile.open();

  fan_2d::graphics::rectangle_t::properties_t p;

  p.size = fan::vec2(1.0 / count, 1);
  p.matrices_reference = &pile.matrices;

  for (uint32_t i = 0; i < count; i++) {
    p.position = fan::vec2(-1.0 + (f32_t)i / (count / 2), 0);
    p.color = fan::color((f32_t)i / count, (f32_t)i / count + 00.1, (f32_t)i / count);
    pile.loco.rectangle.push_back(&pile.cids[i], p);

     //EXAMPLE ERASE
    //r.erase(&pile.context, &pile.cids[i]);
  }

  pile.loco.set_vsync(false);

  pile.loco.rectangle.m_shader.use(pile.loco.get_context());
  pile.loco.rectangle.m_shader.set_matrices(pile.loco.get_context(), &pile.matrices);  

  while(pile.loco.window_open(pile.loco.process_frame())) {
    pile.loco.get_fps();
  }

  return 0;
}