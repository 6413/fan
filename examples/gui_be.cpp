// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(graphics/graphics.h)
#include _FAN_PATH(graphics/gui/be.h);

#include _FAN_PATH(tp/tp.h)

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
    #ifdef gui_demo
      pile->matrices.set_ortho(&pile->context, fan::vec2(0, size.x), fan::vec2(0, size.y));
    #endif
  });

  pile.matrices.open();

  fan::opengl::texturepack texturepack;
  texturepack.open(&pile.context, "texturepack");

  fan_2d::graphics::sprite_t r;
  r.open(&pile.context);
  r.m_shader.bind_matrices(&pile.context, &pile.matrices);
  r.enable_draw(&pile.context);

  fan_2d::graphics::sprite_t::properties_t p;
#ifdef gui_demo
#else
  p.position = 0;
  p.size = 0.5;
#endif
 
  p.image.load(&pile.context, "images/entity_ship.webp");
  p.size = p.image.size / 2;
  p.position = p.size;
  r.push_back(&pile.context, p);

  fan::vec2 window_size = pile.window.get_size();
#ifdef gui_demo
  pile.matrices.set_ortho(&pile.context, fan::vec2(0, window_size.x), fan::vec2(0, window_size.y));
#else
  pile.matrices.set_ortho(&pile.context, fan::vec2(-1, 1), fan::vec2(1, -1));
#endif

  fan_2d::graphics::gui::be_t be;
  be.open();
  be.bind_to_window(&pile.window);

  fan_2d::graphics::gui::be_t::properties_t be_p;
  be_p.hitbox_type = fan_2d::graphics::gui::be_t::hitbox_type_t::rectangle;
  be_p.hitbox_rectangle.position = r.get_position(&pile.context, 0);
  be_p.hitbox_rectangle.size = r.get_size(&pile.context, 0);
  be.push_back(be_p);

  be.set_on_input([](fan_2d::graphics::gui::be_t*, uint32_t index, uint16_t key, fan::key_state key_state, fan_2d::graphics::gui::mouse_stage mouse_stage) {
    if (key_state != fan::key_state::release) {
      return;
    }
    if (key != fan::mouse_left) {
      return;
    }
    fan::print(index, key, (int)key_state);
  });

  
  while(1) {

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