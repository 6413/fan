#include <fan/pch.h>

int main() {
  loco_t loco{{
    .renderer=fan::graphics::engine_t::renderer_t::opengl
  }};
  
  loco_t::image_t image = loco.image_load("images/tire.webp");
  loco_t::image_t image2 = loco.image_load("images/star.webp");

  fan::graphics::sprite_t us{ {
    .position = fan::vec3(400, 400, 254),
    .size = 100,
    .image = image,
  } };

  fan::graphics::sprite_t us2{ {
    .position = fan::vec3(300, 300, 253),
    .size = 100,
    .image = image2,
  } };

  fan::graphics::sprite_t us3{ {
    .position = fan::vec3(100, 100, 252),
    .size = 100,
    .image = image2,
  } };
  us3.erase();
  fan::graphics::sprite_t us4{ {
    .position = fan::vec3(500, 100, 252),
    .size = 100,
    .image = image,
  } };

  std::vector<loco_t::shape_t> shapes;

  //fan::graphics::rectangle_t rect{ {
  //  .position = fan::vec3(200, 200, 0),
  //  .size = 35,
  //  .color = fan::colors::red
  //}};

  using namespace fan::graphics;
  using namespace fan::window;

  loco.loop([&] {
    if (is_mouse_clicked()) {
      shapes.push_back(fan::graphics::sprite_t{ {
        .position = fan::vec3(get_mouse_position(), 252),
        .size = 100,
        .image = image,
      } });
    }
    else if (is_mouse_clicked(fan::mouse_right)) {
      shapes.push_back(fan::graphics::sprite_t{ {
        .position = fan::vec3(get_mouse_position(), 252),
        .size = 100,
        .image = image2,
      } });
    }
 //   rect.set_position(fan::graphics::get_mouse_position());
  });

  return 0;
}