#include <fan/pch.h>

fan::graphics::physics_shapes::polygon_t* draw_p = 0;

/// Draw a closed polygon provided in CCW order.
void DrawPolygon(const b2Vec2* vertices, int vertexCount, b2HexColor color, void* context) {
  //printf("DrawPolygon\n");
  if (draw_p == nullptr) {
    draw_p = new std::remove_pointer_t<decltype(draw_p)>();
  }
  std::vector<fan::graphics::vertex_t> vs(vertexCount + 1);
  for (auto [i, v] : fan::enumerate(vs)) {
    v.position = fan::vec2(vertices[i]) * fan::physics::length_units_per_meter;
    v.color = fan::random::color();
  }
  vs.back() = vs.front();
  *draw_p = fan::graphics::physics_shapes::polygon_t{ {
    .vertices = vs
  }};
}

/// Draw a solid closed polygon provided in CCW order.
void DrawSolidPolygon(b2Transform transform, const b2Vec2* vertices, int vertexCount, float radius, b2HexColor color,
  void* context) {
    printf("DrawSolidPolygon\n");
}

/// Draw a circle.
void DrawCircle(b2Vec2 center, float radius, b2HexColor color, void* context) {
    printf("DrawCircle\n");
}

/// Draw a solid circle.
void DrawSolidCircle(b2Transform transform, float radius, b2HexColor color, void* context) {
    printf("DrawSolidCircle\n");
}

/// Draw a capsule.
void DrawCapsule(b2Vec2 p1, b2Vec2 p2, float radius, b2HexColor color, void* context) {
    printf("DrawCapsule\n");
}

/// Draw a solid capsule.
void DrawSolidCapsule(b2Vec2 p1, b2Vec2 p2, float radius, b2HexColor color, void* context) {
    printf("DrawSolidCapsule\n");
}

/// Draw a line segment.
void DrawSegment(b2Vec2 p1, b2Vec2 p2, b2HexColor color, void* context) {
    printf("DrawSegment\n");
}

/// Draw a transform. Choose your own length scale.
void DrawTransform(b2Transform transform, void* context) {
    printf("DrawTransform\n");
}

/// Draw a point.
void DrawPoint(b2Vec2 p, float size, b2HexColor color, void* context) {
    printf("DrawPoint\n");
}

/// Draw a string.
void DrawString(b2Vec2 p, const char* s, void* context) {
  fan::graphics::text(s, fan::vec2(p) * fan::physics::length_units_per_meter);
    //printf("DrawString\n");
}


int main() {
  using namespace fan::graphics;
  engine_t engine;

  f32_t scale = 0.1;

  fan::vec2 position = 500.0 / fan::physics::length_units_per_meter;

  b2Vec2 b2_vertices[6] = {
		{ -1.5f, -0.5f }, { 1.5f, -0.5f }, 
    { 1.5f, 0.0f }, { 0.0f, 0.9f }, 
    { -1.15f, 0.9f }, { -1.5f, 0.2f },
	};

	for ( int i = 0; i < 6; ++i )
	{
		b2_vertices[i].x *= 0.85f * scale;
		b2_vertices[i].y *= 0.85f * scale;
	}

  std::vector<fan::graphics::vertex_t> vertices;
  for (uint32_t i = 0; i < std::size(b2_vertices); ++i) {
    fan::graphics::vertex_t v;
    v.position = fan::vec2(b2_vertices[i]) * fan::physics::length_units_per_meter;
    v.color = fan::colors::red;
    vertices.emplace_back(v);
  }
  fan::graphics::physics_shapes::polygon_t chassis{{
    .position = fan::vec2(500) + fan::vec2(0, 1.0f * scale) * fan::physics::length_units_per_meter,
    .vertices = vertices,
    .body_type=fan::physics::body_type_e::dynamic_body
  }};

	b2ShapeDef shapeDef = b2DefaultShapeDef();
	shapeDef.density = 1.0f / scale;
	shapeDef.friction = 0.2f;

  fan::vec2 pos = fan::vec2(b2Add( { -1.0f * scale, 0.35f * scale }, position ));
  fan::graphics::physics_shapes::circle_t m_rearWheelId{{
    .position = pos * fan::physics::length_units_per_meter,
    .radius = 50,
    .color = fan::colors::gray,
    .body_type = fan::physics::body_type_e::dynamic_body,
    .shape_properties{.friction=1.5f, .density= 2.0f / scale, .allow_fast_rotation=true}
  }};
  pos = b2Add( { 1.0f * scale, 0.4f * scale }, position );
  fan::graphics::physics_shapes::circle_t m_frontWheelId{{
    .position = pos * fan::physics::length_units_per_meter,
    .radius = 50,
    .color = fan::colors::gray,
    .body_type = fan::physics::body_type_e::dynamic_body,
    .shape_properties{.friction=1.5f, .density= 2.0f / scale, .allow_fast_rotation=true}
  }};



	b2Vec2 axis = { 0.0f, 1.0f };
	b2Vec2 pivot = b2Body_GetPosition( m_rearWheelId );

	float throttle = 0.0f;
	float speed = 35.0f;
	float torque = 2.5f * scale;
	float hertz = 5.0f;
	float dampingRatio = 0.7f;

	b2WheelJointDef jointDef = b2DefaultWheelJointDef();

	jointDef.bodyIdA = chassis;
	jointDef.bodyIdB = m_rearWheelId;
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

	pivot = b2Body_GetPosition( m_frontWheelId );
	jointDef.bodyIdA = chassis;
	jointDef.bodyIdB = m_frontWheelId;
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
	auto m_frontAxleId = b2CreateWheelJoint( engine.physics_context.world_id, &jointDef );

  fan::vec2 window_size = engine.window.get_size();
  f32_t wall_thickness = 50.f;
  auto walls = fan::graphics::physics_shapes::create_stroked_rectangle(window_size / 2, window_size / 2, wall_thickness);
  b2WheelJoint_SetMotorSpeed( m_rearAxleId, speed );
	b2WheelJoint_SetMotorSpeed( m_frontAxleId, speed );
	b2Joint_WakeBodies( m_rearAxleId );
  fan::graphics::mouse_joint_t mouse_joint(chassis);
  b2DebugDraw b{};
  b.drawAABBs = true;

  b.DrawPolygon = DrawPolygon;
  b.DrawSolidPolygon = DrawSolidPolygon;
  b.DrawCircle = DrawCircle;
  b.DrawSolidCircle = DrawSolidCircle;
  b.DrawCapsule = DrawCapsule;
  b.DrawSolidCapsule = DrawSolidCapsule;
  b.DrawSegment = DrawSegment;
  b.DrawTransform = DrawTransform;
  b.DrawPoint = DrawPoint;
  b.DrawString = DrawString;

  fan_window_loop{
    if (is_mouse_clicked()) {
      engine.window.close();
      return;
    }
    mouse_joint.update_mouse(engine.physics_context.world_id, get_mouse_position());
    b2World_Draw(engine.physics_context.world_id, &b);
    engine.physics_context.step(30);
  };
}