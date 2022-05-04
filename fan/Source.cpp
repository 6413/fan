// Creates window and opengl context
#define fan_debug 3
#include <fan/graphics/graphics.h>

int main() {

  fan::window_t window;
  window.open();

  fan::opengl::context_t context;
  context.init();
  context.bind_to_window(&window);
  //context.set_viewport(0, window.get_size());


  fan_2d::graphics::rectangle_t r[2];
  r[0].open(&context);
  fan_2d::graphics::rectangle_t::properties_t rp;
  rp.position = fan::vec2(0, 0);
  rp.size = 10;
  rp.color = fan::colors::red;
  r[0].push_back(&context, rp);
  rp.color = fan::colors::blue;
  r[1].open(&context);
  r[1].push_back(&context, rp);

  fan::opengl::matrices_t matrices;
  matrices.open();

  r[0].m_shader.bind_matrices(&context, &matrices);
  r[1].m_shader.bind_matrices(&context, &matrices);

  matrices.set_ortho(&context, fan::vec2(-1, 1), fan::vec2(-1, 1));

  fan::vec2 window_size = window.get_size();

  fan::graphics::viewport_t viewport[2];
  fan::graphics::viewport_t::properties_t vp;
  vp.position = 0;
  vp.size = fan::vec2(window_size.x / 2, window_size.y); 
  viewport[0].open(&context);
  viewport[0].set(&context, vp);
  vp.position = fan::vec2(window_size.x / 2, 0);
  viewport[1].open(&context);
  viewport[1].set(&context, vp);

  viewport[0].enable(&context);
  r[0].enable_draw(&context);

  viewport[1].enable(&context);
  r[1].enable_draw(&context);

  context.set_vsync(&window, 0);

  struct pile_t {
    fan::graphics::viewport_t* v;
    fan::opengl::context_t* context;
  }pile;

  pile.v = viewport;
  pile.context = &context;
  window.add_resize_callback(&pile, [](fan::window_t* w, const fan::vec2i& window_size, void* userptr) {
    pile_t* pile = (pile_t*)userptr;
    fan::graphics::viewport_t::properties_t p;
    p.position = 0;
    p.size = fan::vec2(window_size.x / 2, window_size.y);
    pile->v[0].set(pile->context, p);
    p.position = fan::vec2(window_size.x / 2, 0);
    pile->v[1].set(pile->context, p);


    //((fan::opengl::context_t*)userptr)->set_viewport(0, size);
  });

  while(1) {

    window.get_fps();

    uint32_t window_event = window.handle_events();
    if(window_event & fan::window_t::events::close){
      window.close();
      break;
    }

    context.process();
    context.render(&window);
  }

  return 0;
}