// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(graphics/graphics.h)
#include _FAN_PATH(io/directory.h)

#define gui_demo

struct pile_t {
  fan::opengl::matrices_t matrices;
  fan::window_t window;
  fan::opengl::context_t context;
};

int main() {

  pile_t pile;

  pile.window.open();

  pile.context.init();
  pile.context.bind_to_window(&pile.window);
  pile.context.set_viewport(0, pile.window.get_size());
  pile.window.add_resize_callback(&pile, [](fan::window_t*, const fan::vec2i& size, void* userptr) {
    pile_t* pile = (pile_t*)userptr;

    pile->context.set_viewport(0, size);
    pile->matrices.set_ortho(&pile->context, fan::vec2(0, size.x), fan::vec2(0, size.y));
  });

  pile.matrices.open();


  fan_2d::graphics::sprite_t r;
  r.open(&pile.context);
  r.m_shader.bind_matrices(&pile.context, &pile.matrices);
  r.enable_draw(&pile.context);

  fan_2d::graphics::sprite_t::properties_t p;

  std::vector<std::string> paths;
  fan::io::iterate_directory("render", [&](const std::string path) {
    paths.push_back(path);
  });

  std::sort(paths.begin(), paths.end());

  uint32_t render_id = 0;

  p.position = pile.window.get_size() / 2;
  p.size = 100;
  p.image.load(&pile.context, paths[render_id]);
  r.push_back(&pile.context, p);

  fan::vec2 window_size = pile.window.get_size();

  pile.matrices.set_ortho(&pile.context, fan::vec2(0, window_size.x), fan::vec2(0, window_size.y));

  fan::time::clock c;
  c.start(fan::time::nanoseconds(1e+8));

  while(1) {

    if (c.finished()) {
      fan::opengl::image_t image;
      fan::opengl::image_t::load_properties_t lp;
      lp.visual_output = fan::opengl::GL_CLAMP_TO_EDGE;
      image.load(&pile.context, paths[render_id], lp);
      fan::opengl::image_t old = r.reload_sprite(&pile.context, 0, image);
      old.unload(&pile.context);
      c.restart();
      render_id = (render_id + 1) % paths.size();
    }

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