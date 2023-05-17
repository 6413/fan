// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_window
#define loco_context

#define loco_line
#define loco_rectangle
#include _FAN_PATH(graphics/loco.h)

constexpr fan::vec2 f(const fan::vec2& p1, const fan::vec2& p2, const fan::vec2& p3, f32_t t) {
  if (p1 == p2 || p2 == p3 || p3 == p1) {
    fan::throw_error("");
  }

  auto f = [&] (f32_t x) {
    f32_t a = (p3.x * (p2.y - p1.y) + p2.x * (p1.y - p3.y) + p1.x * (p3.y - p2.y)) / ((p1.x - p2.x) * (p1.x - p3.x) * (p2.x - p3.x));
    f32_t b = (((p1.x * p1.x) * (p2.y - p3.y) + (p3.x * p3.x) * (p1.y - p2.y) + (p2.x * p2.x) * (p3.y - p1.y)) / ((p1.x - p2.x) * (p1.x - p3.x) * (p2.x - p3.x)));
    f32_t c = ((p2.x * p2.x) * (p3.x * p1.y - p1.x * p3.y) + p2.x * ((p1.x * p1.x) * p3.y - (p3.x * p3.x) * p1.y) + p1.x * p3.x*(p3.x - p1.x) * p2.y) / ((p1.x - p2.x) * (p1.x - p3.x) * (p2.x - p3.x));

    return a * (x * x) + b * x + c;
  };
  t *= (p3.x - p1.x) + p1.x;
  return fan::vec2(t, f(t));
}

constexpr fan::vec2 f2(const fan::vec2& p1, const fan::vec2& p2, const fan::vec2& p3, f32_t t) {
  f32_t u = 1 - t;
  f32_t tt = t * t;
  f32_t uu = u * u;
  f32_t ut = u * t;

  fan::vec2 p;
  p.x = uu * p1.x + 2 * ut * p2.x + tt * p3.x;
  p.y = uu * p1.y + 2 * ut * p2.y + tt * p3.y;

  return p;
}

struct pile_t {

  pile_t() {
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
    viewport.open();
    viewport.set(0, window_size, window_size);
  }

  loco_t loco;
  loco_t::camera_t camera;
  fan::graphics::viewport_t viewport;
}*pile = new pile_t;

int main() {
  fan::vec2 p0(0, 0);
  fan::vec2 p1(300, 300);
  fan::vec2 p2(600, 0);
  p0.y += 300;
  p1.y += 300;
  p2.y += 300;

  loco_t::rectangle_t::properties_t p;
  p.camera = &pile->camera;
  p.viewport = &pile->viewport;

  p.position = fan::vec3(0, 0, 0);
  p.size = fan::vec2(5);
  p.color = fan::colors::red;

  loco_t::shape_t r0;

  loco_t::shape_t r1;
  loco_t::shape_t r2;
  loco_t::shape_t r3;

  loco_t::line_t::properties_t lp;
  lp.camera = &pile->camera;
  lp.viewport = &pile->viewport;
  lp.src = fan::vec2(0, 0);
  lp.dst = fan::vec2(800, 800);
  lp.color = fan::colors::white;

  f32_t count = 50;
  std::vector<loco_t::shape_t> curve(count);
  for (uint32_t i = 0; i < count; ++i) {
    lp.src = f(p0, p1, p2, i / count);
    lp.dst = f(p0, p1, p2, (i + 1) / count);
    if (i == 0) {
      p.color = fan::colors::red;
      p.position = p0;
      p.position.z = 1;
      r1 = p;
    }
    if (i == count / 2) {
      p.color = fan::colors::green;
      p.position = p1;
      p.position.z = 1;
      r2 = p;
    }
    if (i == count - 1) {
      p.color = fan::colors::blue;
      p.position = p2;
      p.position.z = 1;
      r3 = p;
    }
    curve[i] = lp;
  }

  //fan::print(f(p0, p1, p2, 0));
  //fan::print(f2(p0, p1, p2, 0));

  //fan::print(f(p0, p1, p2, 0.5));
  //fan::print(f2(p0, p1, p2, 0.5));

  //fan::print(f(p0, p1, p2, 1));
  //fan::print(f2(p0, p1, p2, 1));

  pile->loco.loop([] {

  });

  return 0;
}