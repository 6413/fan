#include "b2_integration.hpp"

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
  entity.body_id = b2CreateBody(world_id, &body_def);
  b2ShapeDef shape_def = b2DefaultShapeDef();
  shape_def.enablePreSolveEvents = shape_properties.enable_presolve_events;
  shape_def.density = shape_properties.density;
  shape_def.friction = shape_properties.friction;
  shape_def.restitution = shape_properties.restitution;
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
  entity.body_id = b2CreateBody(world_id, &body_def);
  b2ShapeDef shape_def = b2DefaultShapeDef();
  shape_def.enablePreSolveEvents = shape_properties.enable_presolve_events;
  shape_def.density = shape_properties.density;
  shape_def.friction = shape_properties.friction;
  shape_def.restitution = shape_properties.restitution;
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
  entity.body_id = b2CreateBody(world_id, &body_def);
  b2ShapeDef shape_def = b2DefaultShapeDef();
  shape_def.enablePreSolveEvents = shape_properties.enable_presolve_events;
  shape_def.density = shape_properties.density;
  shape_def.friction = shape_properties.friction;
  shape_def.restitution = shape_properties.restitution;
  //shape_def.rollingResistance = shape_properties.rolling_resistance;
  b2CreateCapsuleShape(entity.body_id, &shape_def, &shape);
  return entity;
}

void fan::physics::context_t::step(f32_t dt) {
  b2World_Step(world_id, dt, 4);
}