// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(graphics/graphics.h)

constexpr uint32_t count = 10;

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
  
  p.size = 0.05;

 /* uint32_t c = 0;
  for (f32_t i = 0; i < 5; i++) {
    for (f32_t j = 0; j < 5; j++) {
      p.position = fan::vec2(i / 5, j / 5) * 2 - 1 + 0.05;
      s.push_back(&pile.context, &pile.cids[c], p);
      c++;
    }
  }*/

  p.position = fan::vec2(-0.8, -0.8);
  p.image.load(&pile.context, "images/asteroid.webp", lp);
  s.push_back(&pile.context, &pile.cids[0], p);

  p.position = fan::vec2(-0.5, -0.5);
  p.image.load(&pile.context, "images/planet.webp", lp);
  s.push_back(&pile.context, &pile.cids[1], p);

  p.position = fan::vec2(0, 0);
  p.image.load(&pile.context, "images/test.webp", lp);
  s.push_back(&pile.context, &pile.cids[2], p);

  for (uint32_t i = 0; i < count; i++) {

    /* EXAMPLE ERASE
    s.erase(&pile.context, pile.ids[it]);
    pile.ids.erase(it);
    */
  }

  fan::vec2 window_size = pile.window.get_size();
  fan::vec2 ratio = window_size / window_size.max();
  std::swap(ratio.x, ratio.y);
  pile.matrices.set_ortho(&pile.context, fan::vec2(-1, 1), fan::vec2(-1, 1));

  uint32_t i = 1;

  pile.context.set_vsync(&pile.window, 0);


  bool x = 0;

  s.erase(&pile.context, &pile.cids[0]);

  while(1) {
  /*  for (f32_t i = 0; i < 5; i++) {
      for (f32_t j = 0; j < 5; j++) {
        s.erase(&pile.context, &pile.cids[(uint32_t)i * 5 + (uint32_t)j]);
      }
    }

    uint32_t c = 0;
    for (f32_t i = 0; i < 5; i++) {
      for (f32_t j = 0; j < 5; j++) {
        p.position = fan::vec2(i / 5, j / 5) * 2 - 1 + 0.05;
        s.push_back(&pile.context, &pile.cids[c], p);
        c++;
      }
    }*/

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