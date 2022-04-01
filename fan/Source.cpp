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

  fan_2d::graphics::gui::select_box_t sb;
  fan_2d::graphics::gui::select_box_t::open_properties_t op;
  op.position = fan::vec2(400, 400);
  op.gui_size = 25;
  op.max_text_length = 130;
  sb.open(&w, &context, op);
  sb.enable_draw(&w, &context);

  fan_2d::graphics::gui::select_box_t::properties_t sp;
  sp.text = "rectangle";
  sb.push_back(&w, &context, sp);
  sp.text = "sprite";
  sb.push_back(&w, &context, sp);
  sb.set_on_select_action(&w, &context, 0, [](fan_2d::graphics::gui::select_box_t*, fan::window_t*, uint32_t i, void*) {
    fan::print(i);
  });


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
