// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(graphics/graphics.h)

using id_holder_t = bll_t<uint32_t>;

struct pile_t {
  fan::opengl::matrices_t matrices;
  fan::window_t window;
  fan::opengl::context_t context;
  id_holder_t ids;
};

// filler                         
using sprite_t = fan_2d::graphics::sprite_t<pile_t*, uint32_t>;

void cb(sprite_t* l, uint32_t src, uint32_t dst, uint32_t *p) {
  l->user_global_data->ids[*p] = dst;
}

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


  pile.ids.open();

  sprite_t s;
  s.open(&pile.context, (sprite_t::move_cb_t)cb, &pile);
  s.bind_matrices(&pile.context, &pile.matrices);
  s.enable_draw(&pile.context);

  sprite_t::properties_t p;

  fan::opengl::image_t::load_properties_t lp;
  lp.filter = fan::opengl::GL_LINEAR;
  p.image.load(&pile.context, "images/planeetta.webp", lp);
  p.size = fan::cast<f32_t>(p.image.size) / pile.window.get_size();

    p.position = fan::random::vec2(0, 0);
  for (uint32_t i = 0; i < 1; i++) {
    uint32_t it = pile.ids.push_back(s.push_back(&pile.context, p));
    s.set_user_instance_data(&pile.context, pile.ids[it], it);

    /* EXAMPLE ERASE
    s.erase(&pile.context, pile.ids[it]);
    pile.ids.erase(it);
    */
  }

  fan::vec2 window_size = pile.window.get_size();
  fan::vec2 ratio = window_size / window_size.max();
  std::swap(ratio.x, ratio.y);
  pile.matrices.set_ortho(&pile.context, fan::vec2(-1, 1), fan::vec2(-1, 1));

  pile.window.add_keys_callback(&s, [](fan::window_t*, uint16_t key, fan::key_state key_state, void* user_ptr) {

    sprite_t& pile = *(sprite_t*)user_ptr;

    switch (key) {
      case fan::mouse_scroll_up: {
        pile.input += 0.1;
        break;
      }
      case fan::mouse_scroll_down: {
        pile.input -= 0.1;
        break;
      }
    }
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