#include fan_pch

int main() {

  loco_t loco;
  loco.default_camera->camera.set_ortho(
    fan::vec2(0, loco.window.get_size().x),
    fan::vec2(0, loco.window.get_size().y)
  );

  loco.lighting.ambient = fan::vec3(0.3, 0.3, 0.3);
  loco_t::shapes_t::sprite_t::properties_t p;

  p.size = fan::vec2(1);

  loco_t::image_t image;
  image.load("images/tire.webp");
  p.image = &image;
  p.position = fan::vec3(400, 400, 0);
  p.size = 300;
  p.color.a = 1;
  loco_t::shape_t s0 = p;

  loco_t::shapes_t::light_t::properties_t lp;
  lp.position = fan::vec3(400, 400, 0);
  lp.size = 100;
  lp.color = fan::colors::yellow * 10;
  {
    loco_t::shape_t l0 = lp;

  }

  //{
  //  std::vector<loco_t::shape_t> shapes;

  //  for (int i = 0; i < 100; ++i) {
  //    if (i % 2) {
  //      shapes.push_back(fan::graphics::sprite_t{{}});
  //    }
  //    else {
  //      shapes.push_back(loco_t::shapes_t::light_t::properties_t{{}});
  //    }
  //  }
  //  for (int i = 0; i < 100; ++i) {
  //    int idx = fan::random::value_i64(0, shapes.size() - 1);
  //    shapes.erase(shapes.begin() + idx);
  //  }
  //}
  //

  f32_t x = 0;
  loco.loop([&] {
    loco.get_fps();
   // l0.set_position(loco.get_mouse_position());
  });

  return 0;
}