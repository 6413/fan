// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

//#define loco_vulkan

#define loco_window
#define loco_context
//#define loco_rectangle
#define loco_sprite
#include _FAN_PATH(graphics/loco.h)

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
      //std::swap(ratio.x, ratio.y);
      //camera.set_ortho(
      //  ortho_x * ratio.x, 
      //  ortho_y * ratio.y
      //);
      viewport.set(0, d.size, d.size);
    });
    viewport.open();
    viewport.set(0, window_size, window_size);
  }

  loco_t loco;
  loco_t::camera_t camera;
  fan::graphics::viewport_t viewport;
};

pile_t* pile = new pile_t;

int main() {
  loco_t::image_t image;
  image.load("smoke.webp");

  loco_t::sprite_t::properties_t p;
  
  p.camera = &pile->camera;
  p.viewport = &pile->viewport;  
  p.image = &image;
  
  p.position = fan::vec2(0, 0);
  p.size = 0.001;
  p.color = fan::color(1, 0, 0);
  p.color.a = .5;
  p.blending = true;

  struct particle_t {
    fan::time::clock c;
    fan::vec2 velocity;
    loco_t::shape_t shape;
  };

  std::deque<particle_t> particles;

  bool left_click = false;
  uint16_t depth = 0;
  gloco->get_window()->add_buttons_callback([&](const auto& d) {
    if (d.button != fan::mouse_left) {
      return;
    }
    
    left_click = (bool)d.state;
    if (!left_click) {
      return;
    }
    //for (uint32_t i = 0; i < 10; i++) {
    //  particle_t particle;
    //  p.position.z = depth++;
    //  p.angle = fan::random::value_f32(-1, 1);
    //  particle.shape = p;
    //  //particle.velocity = fan::math::direction_vector<fan::vec2>(fan::math::aim_angle(*(fan::vec2*)&p.position, mp) + fan::math::pi / 2).normalize();
    //  particle.velocity = fan::vec2(cos(i), sin(i)) / 10;
    //  fan::vec2 mp = gloco->get_mouse_position(pile->camera, pile->viewport);
    //  p.position = mp + particle.velocity;
    //  particle.c.start(fan::time::nanoseconds(2e+9));
    //  particles.push_back(particle);
    //  //count++;
    //  //spawn_rate.restart();
    //}
  });

  uint32_t count = 0;
  fan::time::clock spawn_rate;
  spawn_rate.start(fan::time::nanoseconds(0.3e+8));

  pile->loco.loop([&] {
    if (left_click && count < 1000 && spawn_rate.finished()) {
      particle_t particle;
      fan::vec2 mp = gloco->get_mouse_position(pile->camera, pile->viewport);
      p.position = fan::vec2(-0.8, 0);
      p.position.z = depth++;
      p.angle = fan::random::value_f32(-fan::math::pi, fan::math::pi);
      particle.shape = p;
      particle.velocity = fan::math::direction_vector<fan::vec2>(fan::math::aim_angle(*(fan::vec2*)&p.position, mp) + fan::math::pi / 2).normalize();
      particle.c.start(fan::time::nanoseconds(2e+9));
      particles.push_back(particle);
      count++;
      spawn_rate.restart();
    }
    for (uint32_t i = 0; i < particles.size(); ++i) {
      if (particles[i].c.finished()) {
        particles.erase(particles.begin() + i);
        count--;
        continue;
      }
      particles[i].shape.set_position(particles[i].shape.get_position() + fan::vec3(particles[i].velocity * gloco->get_delta_time(), 0));
      fan::color c = particles[i].shape.get_color();
      c.a -= gloco->get_delta_time() / 2;
      particles[i].shape.set_color(c);
      fan::vec2 size = particles[i].shape.get_size();
      f32_t multipler = (particles[i].c.elapsed() / particles[i].c.m_time);
      fan::vec2 new_size =  (fan::vec2(0.05) / p.size);
      new_size.x = pow(new_size.x, multipler);
      new_size.y = pow(new_size.y, multipler);
      size = new_size / 10;
      particles[i].shape.set_size(size);
    }
  });

  return 0;
}