// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

//#define loco_vulkan

#define loco_window
#define loco_context

#define loco_line
#include _FAN_PATH(graphics/loco.h)

constexpr uint32_t count = 1e+7;

struct pile_t {

  void open() {
    auto window_size = loco.get_window()->get_size();
    loco.open_matrices(
      &matrices,
      fan::vec2(-1, 1),
      fan::vec2(-1, 1)
    );
    /* loco.get_window()->add_resize_callback(this, [](fan::window_t* window, const fan::vec2i& size, void* userptr) {
       fan::vec2 window_size = window->get_size();
       fan::vec2 ratio = window_size / window_size.max();
       pile_t* pile = (pile_t*)userptr;
       pile->matrices.set_ortho(
         fan::vec2(0, window_size.x) * ratio.x,
         fan::vec2(0, window_size.y) * ratio.y
       );
       pile->viewport.set(pile->loco.get_context(), 0, size, size);
     });*/
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, window_size, window_size);
  }

  loco_t loco;
  loco_t::matrices_t matrices;
  fan::graphics::viewport_t viewport;
  fan::graphics::cid_t cids[count];
};

int main() {

  pile_t* pile = new pile_t;
  pile->open();

  loco_t::line_t::properties_t p;

  p.matrices = &pile->matrices;
  p.viewport = &pile->viewport;

  uint32_t idx = 0;

  p.src = fan::vec2(0, 0);
  p.dst = p.src;
  p.color = fan::colors::white;
  pile->loco.line.push_back(&pile->cids[idx], p);

  pile->loco.set_vsync(0);


  fan::vec3 dst;
  auto get_random_direction = [&] {
    dst = fan::random::vec2(-1, 1);
    return (dst - p.src).normalize();
  };

  fan::vec2 dir = get_random_direction();

  pile->loco.get_context()->opengl.glLineWidth(1);

  fan::time::clock c;
  c.start(fan::time::nanoseconds(1e+9));

  pile->loco.loop([&] {
    //if (c.finished()) {
    for (uint32_t i = 0; i < 100; ++i) {
      f32_t dt = pile->loco.get_delta_time();
      pile->loco.line.set(&pile->cids[idx], &loco_t::line_t::vi_t::dst, p.dst);
      p.dst = dst;
      //if (fan_2d::math::distance(p.dst, dst) < 0.01) {
      p.dst = p.src;
      p.src = dst;
      dir = get_random_direction();
      p.color = fan::random::color();
      pile->loco.line.push_back(&pile->cids[++idx], p);
      // c.restart();
    }
  //}
// }
  pile->loco.get_fps();
  if (c.finished()) {
    fan::print("line count:", idx + 1);
    c.restart();
  }
    });

  return 0;
}