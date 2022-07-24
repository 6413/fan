// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 1
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(graphics/graphics.h)

constexpr uint32_t count = 1000;

struct pile_t {
  fan::opengl::matrices_t matrices;
  fan::window_t window;
  fan::opengl::context_t context;
  fan::opengl::cid_t cids[count];
};

// filler                         
using rectangle_t = fan_2d::graphics::rectangle_t;

int main() {

  pile_t pile;

  pile.window.open(fan::vec2(600, 600));

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

  pile.context.set_depth_test(true);

  pile.matrices.open();

  rectangle_t r;
  r.open(&pile.context);
  r.bind_matrices(&pile.context, &pile.matrices);
  r.enable_draw(&pile.context);

  rectangle_t::properties_t p;

  p.size = fan::vec2(0.5, 0.5);

  uint32_t i = 0;
  p.position = fan::vec3(0, 0, 0.2);
  p.color = fan::colors::red;
  r.push_back(&pile.context, &pile.cids[i], p);
  i++;
  p.position = fan::vec3(0.2, 0, 0.1);
  p.color = fan::colors::blue;
  r.push_back(&pile.context, &pile.cids[i], p);
  i++;

  fan::vec2 window_size = pile.window.get_size();
  fan::vec2 ratio = window_size / window_size.max();
  std::swap(ratio.x, ratio.y);
  pile.matrices.set_ortho(&pile.context, fan::vec2(-1, 1) * ratio.x, fan::vec2(-1, 1) * ratio.y);

  uint32_t s = 0;

  pile.context.set_vsync(&pile.window, 0);
  //pile.window.set_max_fps(25);

  while(1) {
    // fan::print(s, pile.cids[s].id);
    //r.erase(&pile.context, &pile.cids[s]);

    s++;

    pile.window.get_fps();

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