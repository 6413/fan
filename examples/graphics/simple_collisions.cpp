#include <fan/pch.h>
#include <fan/graphics/physics_shapes.hpp>

int main() {
  loco_t loco;

  std::vector<fan::graphics::physics_shapes::circle_t> entities;
  for (int i = 0; i < 10; ++i) {
    entities.push_back(fan::graphics::physics_shapes::circle_t{{
      .position = fan::vec2(i * 50 + 200, 400),
      .radius = 50,
      .color = fan::random::color(),
      .body_type = fan::physics::body_type_e::dynamic_body,
      .mass_data{.mass = 0.01f}
    }});
  }

  fan::graphics::physics_shapes::rectangle_t walls[4];
  fan::vec2 window_size = loco.window.get_size();
      fan::vec2 platform_size{window_size.x / 3, 50};
  // floor
  walls[0] = fan::graphics::physics_shapes::rectangle_t{ {
      .position = fan::vec2(window_size.x / 2, window_size.y),
      .size = fan::vec2(window_size.x, platform_size.y / 2),
      .color = fan::colors::green
  } };
  // ceiling
  walls[1] = fan::graphics::physics_shapes::rectangle_t{ {
      .position = fan::vec2(window_size.x / 2, 0),
      .size = fan::vec2(window_size.x, platform_size.y / 2),
      .color = fan::colors::green
  } };
  // left
  walls[2] = fan::graphics::physics_shapes::rectangle_t{ {
    .position = fan::vec2(0, window_size.y / 2),
    .size = fan::vec2(platform_size.y, window_size.y),
    .color = fan::colors::green
  } };
  // right
  walls[3] = fan::graphics::physics_shapes::rectangle_t{ {
      .position = fan::vec2(window_size.x, window_size.y / 2),
      .size = fan::vec2(platform_size.y, window_size.y),
      .color = fan::colors::green
  } };

  fan::graphics::physics_shapes::rectangle_t spinner{{
    .position = fan::vec2(window_size.x / 4, 400),
    .size = fan::vec2(600, 10),
    .color = fan::colors::red,
    .body_type = fan::physics::body_type_e::dynamic_body,
    .mass_data{.mass=10000.f}
  }};

  fan::graphics::physics_shapes::rectangle_t spinner2{{
    .position = fan::vec2(window_size.x / 1.5, 400),
    .size = fan::vec2(300, 10),
    .color = fan::colors::yellow,
    .body_type = fan::physics::body_type_e::dynamic_body,
    .mass_data{.mass=10000.f}
  }};

  fan::graphics::physics_shapes::rectangle_t spinner3{{
    .position = fan::vec2(window_size.x / 1.5, 400),
    .size = fan::vec2(100, 10),
    .color = fan::colors::pink,
    .body_type = fan::physics::body_type_e::dynamic_body,
    .mass_data{.mass=10.f}
  }};


  struct filter_context_t {
    void* walls;
    void* spinner;
  }filter_context;
  filter_context.walls = &walls;
  filter_context.spinner = &spinner;

  
  auto ignore_walls_and_spinner = [](b2ShapeId a, b2ShapeId b, void* context) ->bool {
    filter_context_t* filter = (filter_context_t*)context;
    auto* walls = (fan::graphics::physics_shapes::rectangle_t*)filter->walls;
    auto* spinner = (fan::graphics::physics_shapes::rectangle_t*)filter->spinner;
    if (b2Body_GetType(b2Shape_GetBody(a)) == fan::physics::body_type_e::static_body &&
      b2Body_GetType(b2Shape_GetBody(b)) == fan::physics::body_type_e::dynamic_body &&
      B2_ID_EQUALS(b, spinner->body_id)
      ) {
      return false;
    }
    
    
    return true;
  };

  b2World_SetCustomFilterCallback(loco.physics_context.world_id, ignore_walls_and_spinner, &filter_context);
  
  fan::graphics::physics_shapes::rectangle_t anchor{{
    .position = fan::vec3(fan::vec2(spinner.get_position()), 10),
    .size = fan::vec2(10, 10), // Small size for the anchor
    .color = fan::colors::green,
    .body_type = fan::physics::body_type_e::static_body
  }};

  fan::graphics::physics_shapes::rectangle_t anchor2{{
    .position = fan::vec3(fan::vec2(spinner2.get_position()), 10),
    .size = fan::vec2(10, 10), // Small size for the anchor
    .color = fan::colors::green,
    .body_type = fan::physics::body_type_e::static_body
  }};

 
  b2RevoluteJointDef revoluteJointDef = b2DefaultRevoluteJointDef();
  revoluteJointDef.bodyIdA = anchor;
  revoluteJointDef.bodyIdB = spinner;

  fan::vec2 pivot = spinner.get_position();
  revoluteJointDef.localAnchorA = b2Body_GetLocalPoint( revoluteJointDef.bodyIdA, pivot );
	revoluteJointDef.localAnchorB = b2Body_GetLocalPoint( revoluteJointDef.bodyIdB, pivot );
  revoluteJointDef.enableMotor = true;
  revoluteJointDef.motorSpeed = 2.0f; // Set desired motor speed (radians per second)
  revoluteJointDef.maxMotorTorque = 100000000000.0f; // Set maximum motor torque
  revoluteJointDef.collideConnected = false;

  auto joint = b2CreateRevoluteJoint(loco.physics_context.world_id, &revoluteJointDef);

  revoluteJointDef.bodyIdA = anchor2;
  revoluteJointDef.bodyIdB = spinner2;

  pivot = spinner2.get_position();
  revoluteJointDef.localAnchorA = b2Body_GetLocalPoint( revoluteJointDef.bodyIdA, pivot );
	revoluteJointDef.localAnchorB = b2Body_GetLocalPoint( revoluteJointDef.bodyIdB, pivot );
  revoluteJointDef.enableMotor = true;
  revoluteJointDef.motorSpeed = -2.0f; // Set desired motor speed (radians per second)
  revoluteJointDef.maxMotorTorque = 100000000000.0f; // Set maximum motor torque
  revoluteJointDef.collideConnected = false;

  auto joint2 = b2CreateRevoluteJoint(loco.physics_context.world_id, &revoluteJointDef);

  f32_t angle = 0;
  loco.loop([&] {
    b2RevoluteJoint_SetMotorSpeed(joint, -2.f);
    b2RevoluteJoint_SetMotorSpeed(joint2, 20.f);
    b2Body_ApplyAngularImpulse(spinner3, 1000000, true);
    if (ImGui::IsMouseDown(0)) {
      for (int i = 0; i < 1; ++i)
      entities.push_back(fan::graphics::physics_shapes::circle_t{{
        .position = loco.get_mouse_position() / 1.28f,
        .radius = 50,
        .color = fan::random::color(),
        .body_type = fan::physics::body_type_e::dynamic_body,
        .mass_data{.mass=0.01f}
      }});
    }
    //static int x = 0;
    //if (x % 1000 == 0)
    //fan::printcl(entities.size());
    //x++;
    //b2Body_SetTransform(entity2.body_id, fan::vec2(800, 400), b2Body_GetRotation(entity2.body_id));
    //b2Body_SetLinearVelocity(entity2.body_id, b2Vec2_zero);
    //b2Body_ApplyTorque(entity2.body_id, 100000, true);
    //b2Body_SetAngularVelocity(entity2.body_id, 2.f);
    //b2Body_SetLinearVelocity(entity2.body_id, fan::vec2(0));
    //b2Body_SetTransform(entity2.body_id, )
    loco.physics_context.step(loco.delta_time);
  });
}