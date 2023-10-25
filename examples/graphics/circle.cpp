#include fan_pch

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  pile_t() {
    fan::vec2 window_size = loco.get_window()->get_size();
    loco.open_camera(
      &camera,
      ortho_x,
      ortho_y
    );
    loco.get_window()->add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      fan::vec2 window_size = d.size;
    //fan::vec2 ratio = window_size / window_size.max();
   // std::swap(ratio.x, ratio.y);
    //camera.set_ortho(
    //  ortho_x * ratio.x, 
    //  ortho_y * ratio.y
    //);
   // viewport.set(d.size, d.size);
      });
    viewport.open();
    viewport.set(0, window_size, window_size);
  }

  loco_t loco;
  loco_t::camera_t camera;
  fan::graphics::viewport_t viewport;
};

int main() {

  pile_t* pile = new pile_t;

  loco_t::shape_t circle = fan_init_struct(
    typename loco_t::circle_t::properties_t,
    .camera = &pile->camera,
    .viewport = &pile->viewport,
    .position = fan::vec2(0, 0),
    .radius = 0.5,
    .color = fan::colors::white
  );

  pile->loco.set_vsync(false);

  pile->loco.loop([&] {
    pile->loco.get_fps();
  });

  return 0;
}