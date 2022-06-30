// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(graphics/graphics.h)
#include _FAN_PATH(graphics/gui/rectangle_text_box_sized.h)

using id_holder_t = bll_t<uint32_t>;

struct pile_t {
  fan::opengl::matrices_t matrices;
  fan::window_t window;
  fan::opengl::context_t context;
  id_holder_t ids;
};

// filler                         
using rectangle_t = fan_2d::graphics::rectangle_t<pile_t*, uint32_t>;
using text_box_t = fan_2d::graphics::gui::rectangle_text_box_sized_t<pile_t*, uint32_t>;

void cb(rectangle_t* l, uint32_t src, uint32_t dst, uint32_t *p) {
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

  //rectangle_t r;
  //r.open(&pile.context, (rectangle_t::move_cb_t)cb, &pile);
  //r.bind_matrices(&pile.context, &pile.matrices);
  //r.enable_draw(&pile.context);

  //rectangle_t::properties_t p;

  ////struct input_shapes_t {
  ////  button;
  ////  text_box;
  ////};

  ////p.size = fan::vec2(0.2, 0.2);
  ////pile.window.add_mouse_move_callback(&pile, [](fan::window_t* window, const fan::vec2i& position, void* user_ptr) {
  ////  pile->stage[current_stage].input_shapes.set_mouse_position(cursor_position);
  ////});
  //pile.window.add_keys_callback(&pile, [](fan::window_t*, uint16_t key, fan::key_state key_state, void* user_ptr) {
  //  fan::print(key, (int)key_state);
  //});


  //for (uint32_t i = 0; i < 1; i++) {
  //  p.position = fan::vec2(0, 0);
  //  p.color = fan::colors::red;
  //  uint32_t it = pile.ids.push_back(r.push_back(&pile.context, p));
  //  r.set_user_instance_data(&pile.context, pile.ids[it], it);

  //  /* EXAMPLE ERASE
  //  r.erase(&pile.context, pile.ids[it]);
  //  pile.ids.erase(it);
  //  */
  //}

  fan_2d::graphics::font_t font;
  font.open(&pile.context, "fonts/bitter");

  text_box_t text_box;
  text_box.open(&pile.context, &font, cb, &pile);
  text_box.bind_matrices(&pile.context, &pile.matrices);
  text_box_t::properties_t tp;
  tp.theme.button.outline_thickness = 0.005;
  tp.position = 0;
  tp.size = fan::vec2(0.4, 0.1);
  tp.text = "HeLoWoRlD_";
  text_box.push_back(&pile.context, tp);
  text_box.enable_draw(&pile.context);

  fan::vec2 window_size = pile.window.get_size();
  fan::vec2 ratio = window_size / window_size.max();
  std::swap(ratio.x, ratio.y);
  pile.matrices.set_ortho(&pile.context, fan::vec2(-1, 1) * ratio.x, fan::vec2(-1, 1) * ratio.y);

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