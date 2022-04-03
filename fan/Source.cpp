#include <fan/graphics/gui.h>
#include <fan/graphics/gui/be.h>

struct pile_t {
  fan::window_t w;
  fan::opengl::context_t c;
  fan_2d::graphics::rectangle_t r;
  union {
    struct{

    }outside;
    struct{

    }inside;
  };
};

void hover_cb(fan_2d::graphics::gui::be_t* be, uint32_t index, fan_2d::graphics::gui::mouse_stage mouse_stage) {
  pile_t* pile = (pile_t*)be->get_userptr();
  switch (mouse_stage) {
    case fan_2d::graphics::gui::mouse_stage::inside: {
      fan::print("i", index);
      break;
    }
    case fan_2d::graphics::gui::mouse_stage::outside: {
      fan::print("o", index);
      break;
    }
  }
}

int main() {

  pile_t p;
  p.w.open();

  p.c.init();
  p.c.bind_to_window(&p.w);
  p.c.set_viewport(0, p.w.get_size());

  p.r.open(&p.c);
  p.r.enable_draw(&p.c);
  {
    fan_2d::graphics::rectangle_t::properties_t pp;
    pp.position = p.w.get_size() / 2;
    pp.size = 50;
    pp.color = fan::colors::white;
    p.r.push_back(&p.c, pp);
  }

  fan_2d::graphics::gui::be_t rtbs;
  rtbs.open();
  rtbs.bind_to_window(&p.w);
  
  fan_2d::graphics::gui::be_t::properties_t pp;
  pp.size = p.r.get_size(&p.c, 0);
  pp.position = p.r.get_position(&p.c, 0);

  rtbs.push_back(pp);

  rtbs.set_userptr(&p);

  while(1) {

    uint32_t window_event = p.w.handle_events();
    if(window_event & fan::window_t::events::close){
      p.w.close();
      break;
    }

    p.c.process();
    p.c.render(&p.w);
  }

  return 0;
}
