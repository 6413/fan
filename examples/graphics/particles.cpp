#include fan_pch

struct a_t {
  void f(int x) {

  }
};

int main() {

  loco_t loco;
  loco.lighting.ambient -= 0.5;

  loco.clear_color = fan::colors::black;

  fan::vec2 viewport_size = loco.get_window()->get_size();
  loco.default_camera->camera.set_ortho(
    fan::vec2(0, viewport_size.x),
    fan::vec2(0, viewport_size.y)
  );
  /*
  bool click = false;

  loco.get_window()->add_buttons_callback([&](const auto& d) {
    if (d.button != fan::mouse_left) {
      return;
    }
    if (d.state != fan::mouse_state::press) {
      click = false;
      return;
    }
    click = true;
  });

  f32_t z = 0;

  particle_system_t ps;*/

  loco_t::image_t smoke_texture{"images/smoke.webp"};

  //loco_t::particles_t ps;
  loco_t::particles_t::properties_t p;
  p.position = fan::vec3(1300.f / 2 + 100, 1300.f/2, 10);
  p.count = 10;
  p.size = 100;
  p.begin_angle = fan::math::pi / 3;
  p.end_angle = -fan::math::pi / 3.5;
  p.image = &smoke_texture;
  p.color = fan::color(0.4, 0.4, 0.4);
  loco_t::shape_t s = p;

  loco_t::image_t tnt_texture{"images/tnt.webp"};

  fan::graphics::sprite_t tnt{{
      .position = fan::vec2(1300.f / 2, 1300.f / 2),
      .size = fan::vec2(100, 100),
      .image = &tnt_texture
  }};


  loco.loop([&] {
    loco.get_fps();
  });
}