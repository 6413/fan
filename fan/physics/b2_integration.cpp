#include "b2_integration.hpp"

fan::physics::context_t::context_t(const properties_t& properties) {
  b2SetLengthUnitsPerMeter(properties.length_units_per_meter);
  b2WorldDef world_def = b2DefaultWorldDef();
  world_def.gravity = properties.gravity;
  world_id = b2CreateWorld(&world_def);
}

fan::physics::entity_t fan::physics::context_t::create_box(const fan::vec2& position, const fan::vec2& size, uint8_t body_type) {
  polygon_t shape = b2MakeBox(size.x, size.y);
  entity_t entity;
  b2BodyDef body_def = b2DefaultBodyDef();
  body_def.position = position;
  body_def.type = (b2BodyType)body_type;
  entity.body_id = b2CreateBody(world_id, &body_def);
  b2ShapeDef shape_def = b2DefaultShapeDef();
  shape_def.density = 0.0001;
  shape_def.friction = 0.0001;
  shape_def.restitution = 0.0001;
  shape_def.rollingResistance = 0.0001;
  b2CreatePolygonShape(entity.body_id, &shape_def, &shape);
  return entity;
}

fan::physics::entity_t fan::physics::context_t::create_circle(const fan::vec2& position, f32_t radius, uint8_t body_type) {
  circle_t shape;
  shape.center = fan::vec2(0);
  shape.radius = radius;

  entity_t entity;
  b2BodyDef body_def = b2DefaultBodyDef();
  body_def.position = position;
  body_def.type = (b2BodyType)body_type;
  entity.body_id = b2CreateBody(world_id, &body_def);
  b2ShapeDef shape_def = b2DefaultShapeDef();
    shape_def.density = 0.0001;
  shape_def.friction = 0.0001;
  shape_def.restitution = 0.0001;
  shape_def.rollingResistance = 0.0001;
  b2CreateCircleShape(entity.body_id, &shape_def, &shape);
  return entity;
}


void fan::physics::context_t::step(f32_t dt) {
  b2World_Step(world_id, dt, 4);
}