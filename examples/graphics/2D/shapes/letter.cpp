#include <fan/pch.h>

int main() {
  loco_t loco;

  auto v = loco.open_viewport(fan::vec2(0), fan::vec2(600, 600));

  fan::graphics::camera_t camera;
  camera.camera = loco.orthographic_camera.camera;
  camera.viewport = loco.orthographic_camera.viewport;

  fan::string str = "T";

  fan::graphics::letter_t ltr{ {
      .camera = &camera,
      .position = 400,
      .size = 50,
      .color = fan::colors::red,
      .outline_color = fan::colors::black,
      .letter_id = str.get_utf8(0),
      .font_size = 42,
  } };

  fan::string str2 = "W";

  fan::graphics::letter_t ltr2{ {
    .camera = &camera,
    .position = 600,
    .size = 50,
    .color = fan::colors::blue,
    .outline_color = fan::colors::black,
    .letter_id = str2.get_utf8(0),
    .font_size = 60,
} };


  //std::vector<loco_t::shape_t> shapes;
  //for (int i = 0; i < 5; ++i) {
  //  shapes.push_back(r);
  //  shapes[i].set_position(fan::random::vec2(0, 800));
  //}

  loco.loop([&] {
    //  fan::print(r.get_position());
    });
}