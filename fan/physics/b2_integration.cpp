#include "b2_integration.hpp"

#include <cassert>

fan::physics::context_t::context_t(const properties_t& properties) {
  b2SetLengthUnitsPerMeter(properties.length_units_per_meter);
  b2WorldDef world_def = b2DefaultWorldDef();
  world_def.gravity = properties.gravity;
  world_id = b2CreateWorld(&world_def);
}

fan::physics::entity_t fan::physics::context_t::create_box(const fan::vec2& position, const fan::vec2& size, uint8_t body_type, const shape_properties_t& shape_properties) {
  polygon_t shape = b2MakeBox(size.x, size.y);
  entity_t entity;
  b2BodyDef body_def = b2DefaultBodyDef();
  body_def.position = position;
  body_def.type = (b2BodyType)body_type;
  body_def.fixedRotation = shape_properties.fixed_rotation;
  body_def.linearDamping = shape_properties.linear_damping;
  entity.body_id = b2CreateBody(world_id, &body_def);
  b2ShapeDef shape_def = b2DefaultShapeDef();
  shape_def.enablePreSolveEvents = shape_properties.enable_presolve_events;
  shape_def.density = shape_properties.density;
  shape_def.friction = shape_properties.friction;
  shape_def.restitution = shape_properties.restitution;
  shape_def.isSensor = shape_properties.is_sensor;
  shape_def.filter = shape_properties.filter;
  //shape_def.rollingResistance = shape_properties.rolling_resistance;
  b2CreatePolygonShape(entity.body_id, &shape_def, &shape);
  return entity;
}

fan::physics::entity_t fan::physics::context_t::create_circle(const fan::vec2& position, f32_t radius, uint8_t body_type, const shape_properties_t& shape_properties) {
  circle_t shape;
  shape.center = fan::vec2(0);
  shape.radius = radius;

  entity_t entity;
  b2BodyDef body_def = b2DefaultBodyDef();
  body_def.position = position;
  body_def.type = (b2BodyType)body_type;
  body_def.fixedRotation = shape_properties.fixed_rotation;
  body_def.linearDamping = shape_properties.linear_damping;
  entity.body_id = b2CreateBody(world_id, &body_def);
  b2ShapeDef shape_def = b2DefaultShapeDef();
  shape_def.enablePreSolveEvents = shape_properties.enable_presolve_events;
  shape_def.density = shape_properties.density;
  shape_def.friction = shape_properties.friction;
  shape_def.restitution = shape_properties.restitution;
  shape_def.isSensor = shape_properties.is_sensor;
  shape_def.filter = shape_properties.filter;
  //shape_def.rollingResistance = shape_properties.rolling_resistance;
  b2CreateCircleShape(entity.body_id, &shape_def, &shape);
  return entity;
}

fan::physics::entity_t fan::physics::context_t::create_capsule(const fan::vec2& position, const b2Capsule& info, uint8_t body_type, const shape_properties_t& shape_properties) {
  capsule_t shape = info;

  entity_t entity;
  b2BodyDef body_def = b2DefaultBodyDef();
  body_def.position = position;
  body_def.type = (b2BodyType)body_type;
  body_def.fixedRotation = shape_properties.fixed_rotation;
  body_def.linearDamping = shape_properties.linear_damping;
  entity.body_id = b2CreateBody(world_id, &body_def);
  b2ShapeDef shape_def = b2DefaultShapeDef();
  shape_def.enablePreSolveEvents = shape_properties.enable_presolve_events;
  shape_def.density = shape_properties.density;
  shape_def.friction = shape_properties.friction;
  shape_def.restitution = shape_properties.restitution;
  shape_def.isSensor = shape_properties.is_sensor;
  shape_def.filter = shape_properties.filter;
  //shape_def.rollingResistance = shape_properties.rolling_resistance;
  b2CreateCapsuleShape(entity.body_id, &shape_def, &shape);
  return entity;
}

void fan::physics::context_t::step(f32_t dt) {
  b2World_Step(world_id, dt, 4);
  sensor_events.update(world_id);
}

bool fan::physics::context_t::is_on_sensor(fan::physics::body_id_t test_id, fan::physics::body_id_t sensor_id) const {
  return sensor_events.is_on_sensor(test_id, sensor_id);
}

bool fan::physics::presolve_oneway_collision(b2ShapeId shapeIdA, b2ShapeId shapeIdB, b2Manifold* manifold, fan::physics::body_id_t character_body) {
    assert(b2Shape_IsValid(shapeIdA));
    assert(b2Shape_IsValid(shapeIdB));

    float sign = 0.0f;
    if (B2_ID_EQUALS(shapeIdA, character_body)) {
        sign = 1.0f;
    }
    else if (B2_ID_EQUALS(shapeIdB, character_body)) {
        sign = -1.0f;
    }
    else {
        // not colliding with the player, enable contact
        return true;
    }

    b2Vec2 normal = manifold->normal;
    if (sign * normal.y > 0.95f) {
        return true;
    }

    float separation = 0.0f;
    for (int i = 0; i < manifold->pointCount; ++i) {
        float s = manifold->points[i].separation;
        separation = separation < s ? separation : s;
    }

    if (separation > 0.1f * 64.f) {
        // shallow overlap
        return true;
    }

    // normal points down, disable contact
    return false;
}

void fan::physics::sensor_events_t::update(b2WorldId world_id) {
  sensor_events = b2World_GetSensorEvents(world_id);
  for (int i = 0; i < sensor_events.beginCount; ++i) {
    b2SensorBeginTouchEvent ev = sensor_events.beginEvents[i];
    update_contact(b2Shape_GetBody(ev.sensorShapeId), b2Shape_GetBody(ev.visitorShapeId), true);
  }
  for (int i = 0; i < sensor_events.endCount; ++i) {
    b2SensorBeginTouchEvent ev = sensor_events.beginEvents[i];
    update_contact(b2Shape_GetBody(ev.sensorShapeId), b2Shape_GetBody(ev.visitorShapeId), false);
  }
}

void fan::physics::sensor_events_t::update_contact(b2BodyId sensor_id, b2BodyId object_id, bool is_in_contact) {
  for (auto& contact : contacts) {
    if (B2_ID_EQUALS(contact.sensor_id, sensor_id) && B2_ID_EQUALS(contact.object_id, object_id)) {
      contact.is_in_contact = is_in_contact;
      return;
    }
  }
  contacts.push_back({
    .sensor_id = sensor_id,
    .object_id = object_id,
    .is_in_contact = is_in_contact
    });
}

bool fan::physics::sensor_events_t::is_on_sensor(fan::physics::body_id_t test_id, fan::physics::body_id_t sensor_id) const {
  for (const auto& contact : contacts) {
    if (B2_ID_EQUALS(contact.sensor_id, sensor_id) && B2_ID_EQUALS(contact.object_id, test_id)) {
      return contact.is_in_contact;
    }
  }
  return false;
}
