#include <fan/pch.h>

int main() {
  loco_t loco;

  fan::physics::entity_t entity0 = loco.physics_context.create_box(
    fan::vec2(500, 200),
    fan::vec2(100, 10),
    fan::physics::body_type_e::dynamic_body
  );
  fan::physics::entity_t entity1 = loco.physics_context.create_box(
    fan::vec2(400, 960),
    fan::vec2(100, 10)
  );
  
  fan::graphics::rectangle_t rf0{ {
    .size = fan::vec2(100, 10),
    .color = fan::colors::white
  }};
  
  fan::graphics::rectangle_t rf1{ {
      .size = fan::vec2(100, 10),
      .color = fan::colors::green
  }};


  loco.loop([&] {
    {
      fan::vec2 p = b2Body_GetWorldPoint(entity0.body_id, fan::vec2(0));
      b2Rot rotation = b2Body_GetRotation(entity0.body_id);
      f32_t radians = b2Rot_GetAngle(rotation);
      rf0.set_position(p);
      rf0.set_angle(fan::vec3(0, 0, radians));
    }
    {
      fan::vec2 p = b2Body_GetWorldPoint(entity1.body_id, fan::vec2(0));
      b2Rot rotation = b2Body_GetRotation(entity1.body_id);
      f32_t radians = b2Rot_GetAngle(rotation);
      rf1.set_position(p);
      rf1.set_angle(fan::vec3(0, 0, radians));
    }
    
    loco.physics_context.step(loco.delta_time);
  });
}