#include fan_pch

int main() {
  loco_t loco;
  loco.clear_color = 0;
  loco.lighting.ambient = 0.1;

  loco_t::cid_nt_t nt;
  nt.init();

  loco_t::image_t image;
  image.load("images/grass_blade.webp");

  loco_t::shapes_t::grass_2d_t::properties_t p;
  p.position = fan::vec3(100, 1200, 0);
  p.size = 64.f / loco.shapes.grass_2d.planes_per_leaf;
  p.image = &image;
  p.blending = true;

  loco.shapes.grass_2d.push_back(nt, p);
  
  loco_t::shapes_t::light_t::properties_t lp;
  lp.position = fan::vec3(800, 800, 0);
  lp.size = 1000;
  lp.color = fan::colors::yellow * 10;
  loco_t::shape_t l0 = lp;


  for (int i = 0; i < 1000; ++i) {
   ++p.position.z;
    p.position.x += 3;
    //p.color.randomize();
    p.size = fan::vec2(512, fan::random::value_f32(32, 128)) / loco.shapes.grass_2d.planes_per_leaf;
    loco.shapes.grass_2d.push_back(nt, p);
  }
  loco.set_vsync(0);
  loco.loop([&] {
    loco.get_fps();
  });

}