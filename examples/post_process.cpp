// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define fan_debug 0

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

  pile.context.open();
  pile.context.bind_to_window(&pile.window);
  pile.context.set_viewport(0, pile.window.get_size());
  pile.window.add_resize_callback(&pile, [](fan::window_t*, const fan::vec2i& size, void* userptr) {
    pile_t* pile = (pile_t*)userptr;

    pile->context.set_viewport(0, size);

    fan::vec2 window_size = pile->window.get_size();
    fan::vec2 ratio = window_size / window_size.max();
    std::swap(ratio.x, ratio.y);
    //pile->matrices.set_ortho(&pile->context, fan::vec2(-1, 1) * ratio.x, fan::vec2(-1, 1) * ratio.y);
    });

  pile.matrices.open();

  fan_2d::opengl::post_process_t post_process;
  fan::opengl::core::renderbuffer_t::properties_t rp;
  rp.size = pile.window.get_size();
  if (post_process.open(&pile.context, rp)) {
    fan::throw_error("failed to initialize frame buffer");
  }

  post_process.start_capture(&pile.context);

  sprite_t s;
  s.open(&pile.context);
  s.enable_draw(&pile.context);

  sprite_t::properties_t p;

  fan::opengl::image_t::load_properties_t lp;
  lp.filter = fan::opengl::GL_NEAREST;

  p.size = 1;

  p.position = 0;
  p.image.load(&pile.context, "images/grass.webp");
  s.push_back(&pile.context, &pile.cids[0], p);

  pile.context.set_vsync(&pile.window, 0);

  for (uint32_t i = 0; i < count; i++) {

  }

  fan::vec2 window_size = pile.window.get_size();
  fan::vec2 ratio = window_size / window_size.max();
  std::swap(ratio.x, ratio.y);
  pile.matrices.set_ortho(fan::vec2(-1, 1), fan::vec2(-1, 1));

  pile.context.set_vsync(&pile.window, 0);

  while(1) {
    pile.window.get_fps();
    s.m_shader.use(&pile.context);
    s.m_shader.set_matrices(&pile.context, &pile.matrices);  

    uint32_t window_event = pile.window.handle_events();
    if(window_event & fan::window_t::events::close){
      pile.window.close();
      break;
    }
    pile.context.process();
    post_process.sprite.m_shader.use(&pile.context);
    post_process.sprite.m_shader.set_matrices(&pile.context, &pile.matrices);
    post_process.draw(&pile.context);
    pile.context.render(&pile.window);
  }

  return 0;
}