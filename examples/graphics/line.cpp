// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)


#define loco_window
#define loco_context
#define loco_no_inline

#define loco_line
#include _FAN_PATH(graphics/loco.h)


struct pile_t {

  void open() {
    auto window_size = loco.get_window()->get_size();
    loco.open_camera(
      &camera,
      fan::vec2(0, window_size.x),
      fan::vec2(0, window_size.y)
    );
   /* loco.get_window()->add_resize_callback(this, [](fan::window_t* window, const fan::vec2i& size, void* userptr) {
      fan::vec2 window_size = window->get_size();
      fan::vec2 ratio = window_size / window_size.max();
      pile_t* pile = (pile_t*)userptr;
      pile->camera.set_ortho(
        fan::vec2(0, window_size.x) * ratio.x, 
        fan::vec2(0, window_size.y) * ratio.y
      );
      pile->viewport.set(pile->loco.get_context(), 0, size, size);
    });*/
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, window_size, window_size);
  }

  loco_t loco;
  loco_t::camera_t camera;
  fan::graphics::viewport_t viewport;
};

int main() {

  pile_t* pile = new pile_t;
  pile->open();

  loco_t::id_t line = fan_init_struct(
    loco_t::line_t::properties_t,
    .camera = &pile->camera,
    .viewport = &pile->viewport,
    .src = fan::vec2(0, 0),
    .dst = fan::vec2(800, 800),
    .color = fan::colors::white
  );

  pile->loco.set_vsync(0);

  pile->loco.loop([&] {

    pile->loco.get_fps();
  });

  return 0;
}