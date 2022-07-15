// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(graphics/graphics.h)

constexpr uint32_t count = 1;

struct pile_t {
  fan::opengl::matrices_t matrices;
  fan::window_t window;
  fan::opengl::context_t context;
  fan::opengl::cid_t cids[count];
};

// filler                         
using sprite_t = fan_2d::graphics::sprite_t;

int main() {

  pile_t pile;

  pile.window.open();

  pile.context.init();
  pile.context.bind_to_window(&pile.window);
  pile.context.set_viewport(0, pile.window.get_size());
  pile.window.add_resize_callback(&pile, [](fan::window_t*, const fan::vec2i& size, void* userptr) {
    pile_t* pile = (pile_t*)userptr;

    pile->context.set_viewport(0, size);

    fan::vec2 window_size = pile->window.get_size();
    fan::vec2 ratio = window_size / window_size.max();
    std::swap(ratio.x, ratio.y);
    pile->matrices.set_ortho(&pile->context, fan::vec2(-1, 1) * ratio.x, fan::vec2(-1, 1) * ratio.y);
    });

  pile.matrices.open();

  sprite_t s;
  s.open(&pile.context);
  s.bind_matrices(&pile.context, &pile.matrices);
  s.enable_draw(&pile.context);

  sprite_t::properties_t p;

  fan::opengl::image_t::load_properties_t lp;
  lp.filter = fan::opengl::GL_NEAREST;
  p.image.load(&pile.context, "images/test.webp", lp);
  p.size = 0.5;

  for (uint32_t i = 0; i < count; i++) {
    p.position = fan::vec2(0, 0);
    p.angle = fan::math::pi / 2;
    s.push_back(&pile.context, &pile.cids[i], p);

    /* EXAMPLE ERASE
    s.erase(&pile.context, pile.ids[it]);
    pile.ids.erase(it);
    */
  }

  fan::vec2 window_size = pile.window.get_size();
  fan::vec2 ratio = window_size / window_size.max();
  std::swap(ratio.x, ratio.y);
  pile.matrices.set_ortho(&pile.context, fan::vec2(-1, 1), fan::vec2(-1, 1));

  uint32_t i = 0;

  pile.context.set_vsync(&pile.window, 0);

  f32_t x = 0;

  while(1) {

    pile.window.get_fps();
    s.set(&pile.context, &pile.cids[0], &sprite_t::instance_t::angle, x);
    x += pile.window.get_delta_time();
   // s.erase(&pile.context, &pile.cids[i++]);

    uint32_t window_event = pile.window.handle_events();
    if(window_event & fan::window_t::events::close){
      pile.window.close();
      break;
    }

    pile.context.process();
    pile.context.render(&pile.window);
  }

  return 0;
}