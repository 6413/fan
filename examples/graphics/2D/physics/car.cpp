#include <box2d/box2d.h>
import fan;

#include <fan/graphics/types.h>

int main() {
  using namespace fan::graphics;
  engine_t engine;
  

  f32_t scale = 0.2;

  fan::vec2 position = fan::physics::render_to_physics(500.0);

  b2Vec2 b2_vertices[6] = {
    { -1.5f, 0.5f }, { 1.5f, 0.5f },
    { 1.5f, -0.0f }, { 0.0f, -0.9f },
    { -1.15f, -0.9f }, { -1.5f, -0.2f },
  };


	for ( int i = 0; i < 6; ++i )
	{
		b2_vertices[i].x *= 0.85f * scale;
		b2_vertices[i].y *= 0.85f * scale;
	}

  std::vector<fan::graphics::vertex_t> vertices;
  for (uint32_t i = 0; i < std::size(b2_vertices); ++i) {
    fan::graphics::vertex_t v;
    v.position = fan::physics::physics_to_render(b2_vertices[i]);
    v.color = fan::colors::white;
    vertices.emplace_back(v);
  }
  fan::graphics::physics::polygon_t chassis{{
    .position = fan::physics::physics_to_render(position + fan::vec2(0, -1.0f * scale)),
    .radius = 0.15f * scale,
    .vertices = vertices,
    .draw_mode=fan::graphics::primitive_topology_t::triangle_fan,
    .body_type=fan::physics::body_type_e::dynamic_body,
  }};

	b2ShapeDef shapeDef = b2DefaultShapeDef();
	shapeDef.density = 1.0f / scale;
	shapeDef.friction = 0.2f;

  static auto image_tire = engine.image_load("images/tire.webp");

  fan::vec2 pos = fan::vec2(b2Add( { -1.0f * scale, -0.35f * scale }, position ));
  fan::graphics::physics::circle_sprite_t rear_wheel{{
    .position = fan::vec3(fan::physics::physics_to_render(pos), 1),
    .radius = 25,
    .size = 25,
    .image = image_tire,
    .body_type = fan::physics::body_type_e::dynamic_body,
    .shape_properties{.friction=1.5f, .density= 2.0f / scale, .fast_rotation=true}
  }};
  pos = b2Add( { 1.0f * scale, -0.4f * scale }, position );
  fan::graphics::physics::circle_sprite_t front_wheel{{
    .position = fan::vec3(fan::physics::physics_to_render(pos), 1),
    .radius = 25,
    .size = 25,
    .image = image_tire,
    .body_type = fan::physics::body_type_e::dynamic_body,
    .shape_properties{.friction=1.5f, .density= 2.0f / scale, .fast_rotation=true}
  }};


	b2Vec2 axis = { 0.0f, -1.0f };
	b2Vec2 pivot = b2Body_GetPosition( rear_wheel );

	float throttle = 0.0f;
	float speed = 50.0f;
	float torque = 2.5f * scale;
	float hertz = 5.0f;
	float dampingRatio = 0.7f;

	b2WheelJointDef jointDef = b2DefaultWheelJointDef();
	jointDef.bodyIdA = chassis;
	jointDef.bodyIdB = rear_wheel;
	jointDef.localAxisA = b2Body_GetLocalVector( jointDef.bodyIdA, axis );
	jointDef.localAnchorA = b2Body_GetLocalPoint( jointDef.bodyIdA, pivot );
	jointDef.localAnchorB = b2Body_GetLocalPoint( jointDef.bodyIdB, pivot );
	jointDef.motorSpeed = 0.0f;
	jointDef.maxMotorTorque = torque;
	jointDef.enableMotor = true;
	jointDef.hertz = hertz;
	jointDef.dampingRatio = dampingRatio;
	jointDef.lowerTranslation = -0.25f * scale;
	jointDef.upperTranslation = 0.25f * scale;
	jointDef.enableLimit = true;
	auto m_rearAxleId = b2CreateWheelJoint( engine.physics_context.world_id, &jointDef );

	pivot = b2Body_GetPosition(front_wheel);
	jointDef.bodyIdA = chassis;
	jointDef.bodyIdB = front_wheel;
	jointDef.localAxisA = b2Body_GetLocalVector(jointDef.bodyIdA, axis);
	jointDef.localAnchorA = b2Body_GetLocalPoint(jointDef.bodyIdA, pivot);
	jointDef.localAnchorB = b2Body_GetLocalPoint(jointDef.bodyIdB, pivot);
	jointDef.motorSpeed = 0.0f;
	jointDef.maxMotorTorque = torque;
	jointDef.enableMotor = true;
	jointDef.hertz = hertz;
	jointDef.dampingRatio = dampingRatio;
	jointDef.lowerTranslation = -0.25f * scale;
	jointDef.upperTranslation = 0.25f * scale;
	jointDef.enableLimit = true;
	auto m_frontAxleId = b2CreateWheelJoint(engine.physics_context.world_id, &jointDef);

  fan::vec2 window_size = engine.window.get_size();
  f32_t wall_thickness = 50.f;
  auto walls = fan::graphics::physics::create_stroked_rectangle(window_size / 2, window_size / 2, wall_thickness);

	b2Joint_WakeBodies(m_rearAxleId);

  fan::graphics::physics::mouse_joint_t mouse_joint;

  fan::graphics::physics::debug_draw(false);

  static constexpr f32_t amplitude = 150.0f;
  static constexpr f32_t frequency = 0.15f;
  static constexpr f32_t width = 40.0f;
  static constexpr f32_t ground_width = 2560;

  auto pp = fan::graphics::create_sine_ground(fan::vec2(0, 1200), amplitude, frequency, width, ground_width);
  fan::graphics::polygon_t ground{{
    .vertices = pp.vertices,
  }};

  auto points = fan::graphics::ground_points(fan::vec2(0, 1200), amplitude, frequency, width, ground_width);
  engine.physics_context.create_segment(0, points, b2_staticBody, {});

  fan_window_loop{
    if (engine.window.key_state(fan::key_a) != -1) {
      b2WheelJoint_SetMotorSpeed(m_rearAxleId,  -speed);
	    b2WheelJoint_SetMotorSpeed(m_frontAxleId, -speed);
    }
    else if (engine.window.key_state(fan::key_d) != -1) {
      b2WheelJoint_SetMotorSpeed(m_rearAxleId, speed);
	    b2WheelJoint_SetMotorSpeed(m_frontAxleId, speed);
    }
    else {
      b2WheelJoint_SetMotorSpeed(m_rearAxleId, 0);
	    b2WheelJoint_SetMotorSpeed(m_frontAxleId, 0);
    }
    

    //fan::print(chassis.get_physics_position());
    engine.physics_context.step(engine.delta_time);
  };
}