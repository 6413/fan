#include fan_pch

//struct pile_t {
//
//  void open() {
//    auto window_size = loco.get_window()->get_size();
//    loco.open_camera(
//      &camera,
//      fan::vec2(0, window_size.x),
//      fan::vec2(0, window_size.y)
//    );
//    viewport.open(loco.get_context());
//    viewport.set(loco.get_context(), 0, window_size, window_size);
//  } 
//  loco_t::camera_t camera;
//  fan::graphics::viewport_t viewport;
//};


int main() {

  loco_t loco;

  auto camera0 = fan::graphics::add_camera(fan::graphics::direction_e::right);

  fan::graphics::rectangle_t r{{
      .color = fan::colors::red
  }};
  fan::graphics::rectangle_t r2{{
      .camera = camera0,
      .color = fan::colors::green
    }};

  loco.loop([&] {

  });
}