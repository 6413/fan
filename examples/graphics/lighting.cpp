#include fan_pch

int main() {

  loco_t loco;
  loco.default_camera->camera.set_ortho(
    fan::vec2(-1, 1),
    fan::vec2(-1, 1)
  );

  loco.lighting.ambient = fan::vec3(0.3, 0.3, 0.3);
  loco_t::shapes_t::sprite_t::properties_t p;

  p.size = fan::vec2(1);

  loco_t::image_t image;
  image.load("images/lighting.webp");

  loco_t::image_t image2;
  image2.load("images/brick.webp");
  p.image = &image;
  p.position = fan::vec3(0, 0, 0);
  p.color.a = 1;
  loco_t::shape_t s0 = p;
  p.position.x += 0.4;
  p.size = 0.2;
  p.position.z += 2;
  p.color.a = 1;
  p.image = &image2;
  loco_t::shape_t s1 = p;

  loco_t::shapes_t::light_t::properties_t lp;
  lp.position = fan::vec3(0, 0, 0);
  lp.size = 0.5;
  lp.color = fan::colors::yellow * 10;
  loco_t::shape_t l0 = lp;
  
  //for (uint32_t i = 0; i < 1000; i++) {
  //  lp.position = fan::random::vec2(-1, 1);
  //  lp.color = fan::random::color();
  //  lp.position.z = 0;
  //  pile->loco.light.push_back(&pile->cid[0], lp);
  //}

  //offset = vec4(view * vec4(vec2(tc[id] * get_instance().tc_size + get_instance().tc_position), 0, 1)).xy * 2;

  fan::vec3 camerapos = 0;


  loco.get_window()->add_keys_callback([&](const auto& d) {
    if (d.key == fan::key_left) {
      camerapos.x -= 0.1;
      loco.default_camera->camera.set_camera_position(camerapos);
    }
  if (d.key == fan::key_right) {
    camerapos.x += 0.1;
    loco.default_camera->camera.set_camera_position(camerapos);
  }
    });

  f32_t x = 0;
  loco.loop([&] {
    loco.get_fps();
    l0.set_position(loco.get_mouse_position());
    //l0.set_color(fan::color::hsv(x, 100, 100) * 10);
    //x += loco.get_delta_time() * 100;
  /*if (c.finished()) {
    lp.color = fan::random::color();
      lp.size = 0.2;
      lp.position = pile->loco.get_mouse_position(pile->viewport);
      pile->loco.light.push_back(&pile->cid[1], lp);
      c.restart();
  }*/

  #if 1
//  pile->loco.light.set(&pile->cid[0], &loco_t::light_t::vi_t::position, pile->loco.get_mouse_position(pile->viewport));
  #else
  #endif
  #if 0
  pile->loco.light.set(&pile->cid[0], &loco_t::light_t::vi_t::position, pile->loco.get_mouse_position(pile->viewport));
  #endif
  });

  return 0;
}