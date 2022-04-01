#include <fan/graphics/gui.h>

int main() {
  fan::window_t w;
  w.open();

  fan::opengl::context_t context;
  context.init();
  fan::opengl::context_t::properties_t cp;
  cp.samples = 0;
  context.bind_to_window(&w, cp);
  context.set_viewport(0, w.get_size());
  w.add_resize_callback(&context, [](fan::window_t*, const fan::vec2i& s, void* p) {
    ((fan::opengl::context_t*)(p))->set_viewport(0, s);
  });

  fan_2d::opengl::gui::text_renderer_t x;
  x.open(&context);

  fan_2d::graphics::gui::checkbox_t sb;
  fan_2d::graphics::gui::checkbox_t::open_properties_t op;
  op.position = fan::vec2(400, 100);
  op.background_color = fan::color(1, 1, 1, 0.1);
  op.gui_size = 50;
  op.max_text_length = fan_2d::opengl::gui::text_renderer_t::get_text_size(&context, "raspberries", op.gui_size).x;
  sb.open(&w, &context, op);


  fan_2d::graphics::gui::checkbox_t::properties_t p;
  p.theme = fan_2d::graphics::gui::themes::transparent();
  p.text = "apples";
  sb.push_back(&w, &context, p);
  p.text = "pineapples";
  sb.push_back(&w, &context, p);
  p.text = "raspberries";
  sb.push_back(&w, &context, p);
  sb.enable_draw(&w, &context);

  while(1) {


    uint32_t window_event_flag = w.handle_events();
     if(window_event_flag & fan::window_t::events::close){
       w.close();
       break;
     }

    context.process();
    context.render(&w);
  }

  return 0;
}
