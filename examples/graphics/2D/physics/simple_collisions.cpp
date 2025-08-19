// old, newer lib makes code much smaller
#include <fan/pch.h>

int main() {
  loco_t loco;
  std::vector<fan::graphics::physics::circle_t> entities;
  for (int i = 0; i < 10; ++i) {
    entities.push_back(fan::graphics::physics::circle_t{{
      .position = fan::vec2(i * 50 + 200, 400),
      .radius = 10,
      .color = fan::random::color(),
      .body_type = fan::physics::body_type_e::dynamic_body,
      .shape_properties{.friction=0.001f, .density= 1.f}
    }});
  }

  fan::vec2 window_size = loco.window.get_size();
  f32_t wall_thickness = 50.f;
  auto walls = fan::graphics::physics::create_stroked_rectangle(window_size / 2, window_size / 2, wall_thickness);

  fan::graphics::physics::rectangle_t spinner{{
    .position = fan::vec2(window_size.x / 4, 400),
    .size = fan::vec2(600, 10),
    .color = fan::colors::red,
    .body_type = fan::physics::body_type_e::dynamic_body,
    .mass_data{.mass=1.f}
  }};

  fan::graphics::physics::rectangle_t spinner2{{
    .position = fan::vec2(window_size.x / 1.5, 400),
    .size = fan::vec2(300, 10),
    .color = fan::colors::yellow,
    .body_type = fan::physics::body_type_e::dynamic_body,
  }};

  fan::graphics::physics::rectangle_t spinner3{{
    .position = fan::vec2(window_size.x / 1.5, 400),
    .size = fan::vec2(100, 10),
    .color = fan::colors::pink,
    .body_type = fan::physics::body_type_e::dynamic_body,
  }};


  struct filter_context_t {
    void* walls;
    void* spinner;
  }filter_context;
  filter_context.walls = &walls;
  filter_context.spinner = &spinner;

  
  auto ignore_walls_and_spinner = [](b2ShapeId a, b2ShapeId b, void* context) ->bool {
    filter_context_t* filter = (filter_context_t*)context;
    auto* walls = (fan::graphics::physics::rectangle_t*)filter->walls;
    auto* spinner = (fan::graphics::physics::rectangle_t*)filter->spinner;
    if (b2Body_GetType(b2Shape_GetBody(a)) == fan::physics::body_type_e::static_body &&
      b2Body_GetType(b2Shape_GetBody(b)) == fan::physics::body_type_e::dynamic_body &&
      (B2_ID_EQUALS(b, (*spinner))) ||
      (
        b2Body_GetType(b2Shape_GetBody(b)) == fan::physics::body_type_e::static_body &&
      b2Body_GetType(b2Shape_GetBody(a)) == fan::physics::body_type_e::dynamic_body &&
      (B2_ID_EQUALS(a, (*spinner))))
      ) {
      return false;
    }
    
    
    return true;
  };

  b2World_SetCustomFilterCallback(loco.physics_context.world_id, ignore_walls_and_spinner, &filter_context);
  
  fan::graphics::physics::rectangle_t anchor{{
    .position = fan::vec3(fan::vec2(spinner.get_position()), 10),
    .size = fan::vec2(10, 10),
    .color = fan::colors::green,
    .body_type = fan::physics::body_type_e::static_body
  }};

  fan::graphics::physics::rectangle_t anchor2{{
    .position = fan::vec3(fan::vec2(spinner2.get_position()), 10),
    .size = fan::vec2(10, 10),
    .color = fan::colors::green,
    .body_type = fan::physics::body_type_e::static_body
  }};

 
  b2RevoluteJointDef revoluteJointDef = b2DefaultRevoluteJointDef();
  revoluteJointDef.bodyIdA = anchor;
  revoluteJointDef.bodyIdB = spinner;

  fan::vec2 pivot = spinner.get_physics_position();
  revoluteJointDef.localAnchorA = b2Body_GetLocalPoint(revoluteJointDef.bodyIdA, pivot);
	revoluteJointDef.localAnchorB = b2Body_GetLocalPoint(revoluteJointDef.bodyIdB, pivot);
  revoluteJointDef.enableMotor = true;
  revoluteJointDef.motorSpeed = 2.0f;
  revoluteJointDef.maxMotorTorque = 1000000000000000.0f;
  revoluteJointDef.collideConnected = false;

  auto joint = b2CreateRevoluteJoint(loco.physics_context.world_id, &revoluteJointDef);

  revoluteJointDef.bodyIdA = anchor2;
  revoluteJointDef.bodyIdB = spinner2;

  pivot = spinner2.get_physics_position();
  revoluteJointDef.localAnchorA = b2Body_GetLocalPoint(revoluteJointDef.bodyIdA, pivot);
	revoluteJointDef.localAnchorB = b2Body_GetLocalPoint(revoluteJointDef.bodyIdB, pivot);
  revoluteJointDef.enableMotor = true;
  revoluteJointDef.motorSpeed = -2.0f;
  revoluteJointDef.maxMotorTorque = 100.0f;
  revoluteJointDef.collideConnected = false;

  auto joint2 = b2CreateRevoluteJoint(loco.physics_context.world_id, &revoluteJointDef);
  
  f32_t angle = 0;
  fan::time::timer c;
  uint64_t physics_time = 0;
  loco.loop([&] {
    b2RevoluteJoint_SetMotorSpeed(joint, -2.f);
    b2RevoluteJoint_SetMotorSpeed(joint2, 20.f);

   spinner3.apply_angular_impulse(100);
    if (ImGui::IsMouseDown(0) && !gloco->console.input.IsFocused()) {
      for (int i = 0; i < 10; ++i)
      entities.push_back(fan::graphics::physics::circle_t{{
        .position = loco.get_mouse_position() / 1.28f,
        .radius = 10,
        .color = fan::random::color(),
        .body_type = fan::physics::body_type_e::dynamic_body,
        .shape_properties{.friction=0.001f, .density= 0.001f, .fixed_rotation=true}
      }});
    }
    //

    //fan::printcl(
    //  "physics:", physics_time / 1e9, 
    //  "render:", loco.delta_time - physics_time / 1e9,
    //  "delta_time:", loco.delta_time,
    //  "elapsed:", physics_time / 1e9
    //);

    c.start();
    loco.physics_context.step(loco.delta_time);
    physics_time = c.elapsed();
  });
}