#include <fan/pch.h>
#include <fan/graphics/physics_shapes.hpp>

int main() {
  loco_t loco;

  std::vector<fan::graphics::physics_shapes::rectangle_t*> entities;
  entities.resize(100);
  for (int i = 0; i < 100; ++i) {
    entities[i] = new fan::graphics::physics_shapes::rectangle_t{{
      .position = fan::vec2(500, 200),
      .size = fan::vec2(100, 10),
      .body_type = fan::physics::body_type_e::dynamic_body
    }};
  }

  fan::graphics::physics_shapes::rectangle_t entity1{{
    .position = fan::vec2(800, 900),
    .size = fan::vec2(1000, 10),
    .color = fan::colors::green,
    .body_type = fan::physics::body_type_e::static_body
  }};

  fan::graphics::physics_shapes::rectangle_t entity2{{
    .position = fan::vec2(400, 400),
    .size = fan::vec2(100, 10),
    .color = fan::colors::red,
    .body_type = fan::physics::body_type_e::dynamic_body
  }};

  b2RevoluteJointDef joint;
  //joint.body

  loco.loop([&] {
    //b2Body_SetAngularVelocity(entity2.body_id, 2.f);
    //b2Body_SetLinearVelocity(entity2.body_id, fan::vec2(0));
    //b2Body_SetTransform(entity2.body_id, )
    loco.physics_context.step(loco.delta_time);
  });
}