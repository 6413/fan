#include <fan/pch.h>

loco_t::shape_t get_shape() {
  fan::graphics::camera_t camera;
  camera.camera = gloco->orthographic_camera.camera;
  camera.viewport = gloco->orthographic_camera.viewport;

  fan::graphics::rectangle_t r{ {
    .camera = &camera,
    .position = 400,
    .size = 50,
    .color = fan::colors::red,
} };
  return *dynamic_cast<loco_t::shape_t*>(&r);
}




int main() {

  loco_t loco;

  //std::vector<> shapes;
 // shapes.reserve(2);

  std::vector<loco_t::shape_t> shapes;
  loco_t::rectangle_t::properties_t p{
    .position = fan::vec3(fan::random::vec2(0, 600), 0),
    .size = 50,
    .color = fan::colors::red,
  };


  loco.loop([&] {
    for (int i = 0; i < 2000; ++i) {
      p.position.z = rand();
      shapes.push_back(p);
    }
    std::vector<int> numbers;

    for (std::size_t i = 0; i < shapes.size(); i++) 
      numbers.push_back(i);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(numbers.begin(), numbers.end(), std::default_random_engine(seed));

    for (std::size_t j = 0; j < shapes.size(); ++j) {
      shapes[numbers[j]].erase();
    }

    shapes.clear();

    //shape.set_position(loco.get_mouse_position());
  //  fan::print(r.get_position());
  });
}