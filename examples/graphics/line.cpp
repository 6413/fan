// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_vulkan

#define loco_window
#define loco_context

#define loco_line
#include _FAN_PATH(graphics/loco.h)

constexpr uint32_t count = 1e+3;

struct pile_t {

  void open() {
    auto window_size = loco.get_window()->get_size();
    loco.open_camera(
      &camera,
      fan::vec2(0, window_size.x),
      fan::vec2(0, window_size.y)
    );
   /* loco.get_window()->add_resize_callback(this, [](fan::window_t* window, const fan::vec2i& size, void* userptr) {
      fan::vec2 window_size = window->get_size();
      fan::vec2 ratio = window_size / window_size.max();
      pile_t* pile = (pile_t*)userptr;
      pile->camera.set_ortho(
        fan::vec2(0, window_size.x) * ratio.x, 
        fan::vec2(0, window_size.y) * ratio.y
      );
      pile->viewport.set(pile->loco.get_context(), 0, size, size);
    });*/
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, window_size, window_size);
  }

  loco_t loco;
  loco_t::camera_t camera;
  fan::graphics::viewport_t viewport;
  fan::graphics::cid_t cids[count];
};

int main() {

  pile_t* pile = new pile_t;
  pile->open();

  loco_t::line_t::properties_t p;

  //p.block_properties.
  p.camera = &pile->camera;
  p.viewport = &pile->viewport;

  p.src = fan::vec2(400, 400);
  p.dst = fan::vec2(400 + 9 * 20, 400 - 5 * 20);
  p.color = fan::colors::white;
  pile->loco.line.push_back(&pile->cids[0], p);

  fan::vec3 taso = p.dst - p.src;

  p.src = fan::vec2(400 + 9 * 20 / 2, 400 - 5 * 20 / 2);
  f32_t x = taso.x;
  f32_t y = taso.y;
  f32_t cs = cos(-fan::math::pi / 2);
  f32_t sn = sin(-fan::math::pi / 2);
  taso.x = x * cs - y * sn; 
  taso.y = x * sn + y * cs;
  p.dst = p.src + taso;
  p.color = fan::colors::red;
  pile->loco.line.push_back(&pile->cids[0], p);
  f32_t d = 0;

  pile->loco.set_vsync(0);

  pile->loco.loop([&] {
      f32_t x = taso.x;
  f32_t y = taso.y;
    f32_t cs = cos(d + -fan::math::pi / 2);
    f32_t sn = sin(d + -fan::math::pi / 2);
    //taso.x = ; 
    //taso.y = ;
    p.dst = p.src + 
      fan::vec2(
        x * cs - y * sn,
        x * sn + y * cs
      )
      ;
    pile->loco.line.set(&pile->cids[0], &loco_t::line_t::vi_t::dst, p.dst);
    d += pile->loco.get_delta_time();
    pile->loco.get_fps();
  });

  return 0;
}