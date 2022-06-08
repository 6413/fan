// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define fan_debug fan_debug_high

#include _FAN_PATH(graphics/graphics.h)

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
    pile->matrices.set_ortho(&pile->context, fan::vec2(-1, 1), fan::vec2(1, -1));
#endif
    });

  pile.matrices.open();

  fan_3d::opengl::model_t model;
  model.open(&pile.context);
  model.bind_matrices(&pile.context, &pile.matrices);
  model.enable_draw(&pile.context);
  
  fan_3d::opengl::model_t::properties_t mp;
  mp.loaded_model = fan_3d::opengl::model_t::load_model(&pile.context, "models/test.obj");
  mp.position = fan::vec3(0, 0, 0);

  fan::vec3 camera_position = 0;
  mp.camera_position = &camera_position;

  model.set(&pile.context, mp);
  
  fan::vec2ui window_size = pile.window.get_size();

  pile.matrices.set_ortho(&pile.context, fan::vec2(-20, 20), fan::vec2(20, -20));

  f32_t a = 0;
  model.set_rotation_vector(&pile.context, 0, fan::vec3(1, 1, 0));
  pile.context.set_vsync(&pile.window, false);
  model.set_light_position(&pile.context, fan::vec3(5, 0, 0));
  while(1) {

    model.set_angle(&pile.context, a += pile.window.get_delta_time());
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