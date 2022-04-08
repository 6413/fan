// Creates window and opengl context

#include <fan/graphics/graphics.h>
#include <fan/physics/static_joint.h>

struct pile_t {
  fan::window_t w;
  fan::opengl::context_t c;
  fan_2d::graphics::rectangle_t r;
  fan_2d::graphics::sprite_t s;
};

void user_set_position_cb(void* userptr, uint8_t joint_type, uint32_t joint_id, const fan::vec2& position) {
  pile_t* pile = (pile_t*)userptr;
  switch (joint_type) {
    case 0: {
      pile->r.set_position(&pile->c, joint_id, position);
      break;
    }
    case 1: {
      pile->s.set_position(&pile->c, joint_id, position);
      break;
    }
  }
}

void user_set_angle_cb(void* userptr, uint8_t joint_type, uint32_t joint_id, const fan::vec2& position, f32_t angle) {
  pile_t* pile = (pile_t*)userptr;
  switch (joint_type) {
    case 0: {
      pile->r.set_position(&pile->c, joint_id, position);
      pile->r.set_angle(&pile->c, joint_id, angle);
      break;
    }
    case 1: {
      pile->s.set_position(&pile->c, joint_id, position);
      pile->s.set_angle(&pile->c, joint_id, angle);
      break;
    }
  }
}

int main() {

  pile_t pile;

  pile.w.open();


  pile.c.init();
  pile.c.bind_to_window(&pile.w);
  pile.c.set_viewport(0, pile.w.get_size());
  pile.w.add_resize_callback(&pile.c, [](fan::window_t*, const fan::vec2i& size, void* userptr) {
    ((fan::opengl::context_t*)userptr)->set_viewport(0, size);
  });


  pile.r.open(&pile.c);
  fan_2d::graphics::rectangle_t::properties_t rp;
  rp.position = pile.w.get_size() / 2;
  rp.size = 100;
  rp.color = fan::colors::red;
  pile.r.push_back(&pile.c, rp);
  pile.r.enable_draw(&pile.c);

  
  pile.s.open(&pile.c);
  fan_2d::graphics::sprite_t::properties_t sp;
  sp.position = pile.w.get_size() / 4;
  sp.size = 100;
  sp.image = fan::graphics::load_image(&pile.c, "images/block/Dirt.webp");
  pile.s.push_back(&pile.c, sp);
  pile.s.push_back(&pile.c, sp);
  pile.s.enable_draw(&pile.c);

  fan_2d::physics::joint_head_t joint_head;
  joint_head.joint_type = 0;
  joint_head.joint_id = 0;
  joint_head.rotation_point = fan::vec2(-100, -100);
  joint_head.joint_tail.open();
  fan_2d::physics::joint_tail_t joint_tail;
  joint_tail.rotation_point = 0;
  joint_tail.joint_id = 0;
  joint_tail.m_position = fan::vec2(-200, 0);
  joint_tail.joint_type = 1;
  joint_tail.joint_tail.open();
  joint_head.joint_tail.push_back(joint_tail);

  joint_tail.m_position = fan::vec2(-200, 0);
  joint_tail.joint_id = 1;

  joint_head.joint_tail[0].joint_tail.push_back(joint_tail);

  while(1) {

    joint_head.set_position(&pile, user_set_position_cb, pile.w.get_mouse_position());
    joint_head.set_angle(&pile, user_set_position_cb, user_set_angle_cb, pile.w.get_mouse_position(), fan::time::clock::now() / 1e+9);

    uint32_t window_event = pile.w.handle_events();
    if(window_event & fan::window_t::events::close){
      pile.w.close();
      break;
    }

    pile.c.process();
    pile.c.render(&pile.w);
  }

  return 0;
}