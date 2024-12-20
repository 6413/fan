#include "b2_integration.hpp"

// half size
fan::physics::polygon_t fan::physics::make_box(const fan::vec2& size) {
  return b2MakeBox(size.x, size.y);
}

fan::physics::context_t::context_t(const properties_t& properties) {
  b2SetLengthUnitsPerMeter(properties.length_units_per_meter);
  b2WorldDef world_def = b2DefaultWorldDef();
  world_def.gravity = properties.gravity;
  world_id = b2CreateWorld(&world_def);
}

fan::physics::entity_t fan::physics::context_t::create_box(const fan::vec2& position, const fan::vec2& size, uint8_t body_type) {
  polygon_t shape = fan::physics::make_box(size);
  entity_t entity;
  b2BodyDef body_def = b2DefaultBodyDef();
  body_def.position = position;
  body_def.type = (b2BodyType)body_type;
  entity.body_id = b2CreateBody(world_id, &body_def);
  b2ShapeDef shape_def = b2DefaultShapeDef();
  b2CreatePolygonShape(entity.body_id, &shape_def, &shape);
  return entity;
}

void fan::physics::context_t::step(f32_t dt) {
  b2World_Step(world_id, dt, 4);
}