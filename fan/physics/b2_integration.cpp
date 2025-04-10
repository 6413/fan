#include "b2_integration.hpp"

#include <cassert>

fan::physics::context_t::operator b2WorldId& () {
  return world_id;
}

fan::physics::context_t::context_t(const properties_t& properties) {
  //b2SetLengthUnitsPerMeter(properties.length_units_per_meter);
  b2WorldDef world_def = b2DefaultWorldDef();
  world_def.gravity = properties.gravity * length_units_per_meter;
  b2SetLengthUnitsPerMeter(1.f / 512.f);
  world_id = b2CreateWorld(&world_def);
}

void fan::physics::context_t::set_gravity(const fan::vec2& gravity){
  b2World_SetGravity(world_id, gravity);
}

fan::vec2 fan::physics::context_t::get_gravity() const {
  return b2World_GetGravity(world_id);
}

fan::physics::entity_t fan::physics::context_t::create_box(const fan::vec2& position, const fan::vec2& size, f32_t angle, uint8_t body_type, const shape_properties_t& shape_properties) {
  polygon_t shape = b2MakeBox(size.x / length_units_per_meter * shape_properties.collision_multiplier.x, size.y/ length_units_per_meter * shape_properties.collision_multiplier.y);
  entity_t entity;
  b2BodyDef body_def = b2DefaultBodyDef();
  body_def.position = position / length_units_per_meter;
  body_def.rotation = b2MakeRot(angle);
  body_def.type = (b2BodyType)body_type;
  body_def.fixedRotation = shape_properties.fixed_rotation;
  body_def.linearDamping = shape_properties.linear_damping;
  body_def.allowFastRotation = shape_properties.fast_rotation;
  entity = b2CreateBody(world_id, &body_def);
#if fan_debug >= fan_debug_medium
  if (entity.is_valid() == false) {
    fan::throw_error_impl();
  }
#endif
  b2ShapeDef shape_def = b2DefaultShapeDef();
  shape_def.enablePreSolveEvents = shape_properties.presolve_events;
  shape_def.density = shape_properties.density;
  shape_def.friction = shape_properties.friction;
  shape_def.restitution = shape_properties.restitution;
  shape_def.isSensor = shape_properties.is_sensor;
  shape_def.filter = shape_properties.filter;
  //shape_def.rollingResistance = shape_properties.rolling_resistance;
  b2CreatePolygonShape(entity, &shape_def, &shape);
  return entity;
}

fan::physics::entity_t fan::physics::context_t::create_circle(const fan::vec2& position, f32_t radius, f32_t angle, uint8_t body_type, const shape_properties_t& shape_properties) {
  circle_t shape;
  shape.center = fan::vec2(0);
  shape.radius = radius / length_units_per_meter * shape_properties.collision_multiplier.x;

  entity_t entity;
  b2BodyDef body_def = b2DefaultBodyDef();
  body_def.position = position / length_units_per_meter;
  body_def.rotation = b2MakeRot(angle);
  body_def.type = (b2BodyType)body_type;
  body_def.fixedRotation = shape_properties.fixed_rotation;
  body_def.linearDamping = shape_properties.linear_damping;
  body_def.allowFastRotation = shape_properties.fast_rotation;

  entity = b2CreateBody(world_id, &body_def);
#if fan_debug >= fan_debug_medium
  if (entity.is_valid() == false) {
    fan::throw_error_impl();
  }
#endif
  b2ShapeDef shape_def = b2DefaultShapeDef();
  shape_def.enablePreSolveEvents = shape_properties.presolve_events;
  shape_def.density = shape_properties.density;
  shape_def.friction = shape_properties.friction;
  shape_def.restitution = shape_properties.restitution;
  shape_def.isSensor = shape_properties.is_sensor;
  shape_def.filter = shape_properties.filter;

  //shape_def.rollingResistance = shape_properties.rolling_resistance;
  b2CreateCircleShape(entity, &shape_def, &shape);
  return entity;
}

fan::physics::entity_t fan::physics::context_t::create_capsule(const fan::vec2& position, const b2Capsule& info, uint8_t body_type, const shape_properties_t& shape_properties) {
  capsule_t shape = info;
  shape.center1.x /= length_units_per_meter / shape_properties.collision_multiplier.x;
  shape.center1.y /= length_units_per_meter / shape_properties.collision_multiplier.y;
  shape.center2.x /= length_units_per_meter / shape_properties.collision_multiplier.x;
  shape.center2.y /= length_units_per_meter / shape_properties.collision_multiplier.y;
  shape.radius /= length_units_per_meter / shape_properties.collision_multiplier.x;

  entity_t entity;
  b2BodyDef body_def = b2DefaultBodyDef();
  body_def.position = position / length_units_per_meter;
  body_def.type = (b2BodyType)body_type;
  body_def.fixedRotation = shape_properties.fixed_rotation;
  body_def.linearDamping = shape_properties.linear_damping;
  body_def.allowFastRotation = shape_properties.fast_rotation;
  entity = b2CreateBody(world_id, &body_def);
#if fan_debug >= fan_debug_medium
  if (entity.is_valid() == false) {
    fan::throw_error_impl();
  }
#endif
  b2ShapeDef shape_def = b2DefaultShapeDef();
  shape_def.enablePreSolveEvents = shape_properties.presolve_events;
  shape_def.density = shape_properties.density;
  shape_def.friction = shape_properties.friction;
  shape_def.restitution = shape_properties.restitution;
  shape_def.isSensor = shape_properties.is_sensor;
  shape_def.filter = shape_properties.filter;
  //shape_def.rollingResistance = shape_properties.rolling_resistance;
  b2CreateCapsuleShape(entity, &shape_def, &shape);
  return entity;
}

fan::physics::entity_t fan::physics::context_t::create_segment(const fan::vec2& position, const std::vector<fan::vec2>& points, uint8_t body_type, const shape_properties_t& shape_properties) {
  entity_t entity;
  b2BodyDef body_def = b2DefaultBodyDef();
  body_def.position = position / length_units_per_meter;
  body_def.type = (b2BodyType)body_type;
  body_def.fixedRotation = shape_properties.fixed_rotation;
  body_def.linearDamping = shape_properties.linear_damping;
  body_def.allowFastRotation = shape_properties.fast_rotation;
  entity = b2CreateBody(world_id, &body_def);
#if fan_debug >= fan_debug_medium
  if (entity.is_valid() == false) {
    fan::throw_error_impl();
  }
#endif
  b2ShapeDef shape_def = b2DefaultShapeDef();
  shape_def.enablePreSolveEvents = shape_properties.presolve_events;
  shape_def.density = shape_properties.density;
  shape_def.friction = shape_properties.friction;
  shape_def.restitution = shape_properties.restitution;
  shape_def.isSensor = shape_properties.is_sensor;
  shape_def.filter = shape_properties.filter;

  for (std::size_t i = 0; i < points.size() - 1; ++i) {
    segment_t shape;
    shape.point1 = points[i] / length_units_per_meter;
    shape.point2 = points[i + 1] / length_units_per_meter;
    b2CreateSegmentShape(entity, &shape_def, &shape);
  }
  // connnect last to first
  if (points.size() > 2) {
    segment_t shape;
    shape.point1 = points.back() / length_units_per_meter;
    shape.point2 = points.front() / length_units_per_meter;
    b2CreateSegmentShape(entity, &shape_def, &shape);
  }
  return entity;
}

fan::physics::entity_t fan::physics::context_t::create_polygon(const fan::vec2& position, f32_t radius, const std::vector<fan::vec2>& points, uint8_t body_type, const shape_properties_t& shape_properties) {
  entity_t entity;
  b2BodyDef body_def = b2DefaultBodyDef();
  body_def.position = position / length_units_per_meter;
  body_def.type = (b2BodyType)body_type;
  body_def.fixedRotation = shape_properties.fixed_rotation;
  body_def.linearDamping = shape_properties.linear_damping;
  body_def.allowFastRotation = shape_properties.fast_rotation;
  entity = b2CreateBody(world_id, &body_def);

#if fan_debug >= fan_debug_medium
  // world probably locked
  if (entity.is_valid() == false) {
    fan::throw_error_impl();
  }
#endif
  b2ShapeDef shape_def = b2DefaultShapeDef();
  shape_def.enablePreSolveEvents = shape_properties.presolve_events;
  shape_def.density = shape_properties.density;
  shape_def.friction = shape_properties.friction;
  shape_def.restitution = shape_properties.restitution;
  shape_def.isSensor = shape_properties.is_sensor;
  shape_def.filter = shape_properties.filter;

  std::vector<b2Vec2> b2_points(points.size());
  for (std::size_t i = 0; i < b2_points.size(); ++i) {
    b2_points[i] = points[i] / length_units_per_meter;
  }

  b2Hull hull = b2ComputeHull(b2_points.data(), b2_points.size());
  b2Polygon polygon = b2MakePolygon(&hull, radius);

  b2CreatePolygonShape(entity, &shape_def, &polygon);
  return entity;
}


void fan::physics::context_t::step(f32_t dt) {
  static f32_t skip = 0;
  static f32_t x = 0;
  x += dt;
  skip += dt;
  if (skip == 0) {
    b2World_Step(world_id, 1.0 / 30, 4);
    return;
  }
  f32_t timestep = 1.0/256.0;
  while (x > timestep) {
    {
      auto it = body_updates.GetNodeFirst();
      while (it != body_updates.dst) {
        body_updates.StartSafeNext(it);
        auto& node = body_updates[it];
        node.cb();
        it = body_updates.EndSafeNext();
      }
    }
    b2World_Step(world_id, timestep, 4);
    sensor_events.update(world_id);
    x -= timestep;
  }
  body_updates.Clear();
}

bool fan::physics::context_t::is_on_sensor(fan::physics::body_id_t test_id, fan::physics::body_id_t sensor_id) const {
  return sensor_events.is_on_sensor(test_id, sensor_id);
}


// screen coordinates
fan::physics::ray_result_t fan::physics::context_t::raycast(const fan::vec2& src_, const fan::vec2& dst_) {
  fan::vec2 src = src_ / fan::physics::length_units_per_meter;
  fan::vec2 dst = dst_ / fan::physics::length_units_per_meter;
  b2QueryFilter qf = b2DefaultQueryFilter();

  b2Vec2 translation = dst - src;

  b2RayResult b2result = b2World_CastRayClosest(world_id, src, translation, qf);
  ray_result_t result;
  result.shapeId = b2result.shapeId;
  result.point = b2result.point;
  result.normal = b2result.normal;
  result.fraction = b2result.fraction;
  result.hit = b2result.hit;
  result.point *= fan::physics::length_units_per_meter;
  return result;
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


fan::physics::body_id_t fan::physics::deep_copy_body(b2WorldId worldId, fan::physics::body_id_t sourceBodyId) {
  if (!b2Body_IsValid(sourceBodyId)) {
    return b2_nullBodyId;
  }

  b2BodyDef bodyDef = b2DefaultBodyDef();
  bodyDef.type = b2Body_GetType(sourceBodyId);
  bodyDef.position = b2Body_GetPosition(sourceBodyId);
  bodyDef.rotation = b2Body_GetRotation(sourceBodyId);
  bodyDef.linearVelocity = b2Body_GetLinearVelocity(sourceBodyId);
  bodyDef.angularVelocity = b2Body_GetAngularVelocity(sourceBodyId);
  bodyDef.linearDamping = b2Body_GetLinearDamping(sourceBodyId);
  bodyDef.angularDamping = b2Body_GetAngularDamping(sourceBodyId);
  bodyDef.gravityScale = b2Body_GetGravityScale(sourceBodyId);
  bodyDef.sleepThreshold = b2Body_GetSleepThreshold(sourceBodyId);
  bodyDef.enableSleep = b2Body_IsSleepEnabled(sourceBodyId);
  bodyDef.isAwake = b2Body_IsAwake(sourceBodyId);
  bodyDef.fixedRotation = b2Body_IsFixedRotation(sourceBodyId);
  bodyDef.isBullet = b2Body_IsBullet(sourceBodyId);
  bodyDef.isEnabled = b2Body_IsEnabled(sourceBodyId);
  bodyDef.userData = b2Body_GetUserData(sourceBodyId);

  b2BodyId newBodyId = b2CreateBody(worldId, &bodyDef);
  if (!b2Body_IsValid(newBodyId)) {
    return b2_nullBodyId;
  }

  b2MassData massData = b2Body_GetMassData(sourceBodyId);
  b2Body_SetMassData(newBodyId, massData);

  const int shapeCount = b2Body_GetShapeCount(sourceBodyId);
  if (shapeCount > 0) {
    std::vector<b2ShapeId> shapes(shapeCount);
    b2Body_GetShapes(sourceBodyId, shapes.data(), shapeCount);

    for (b2ShapeId sourceShapeId : shapes) {
      b2ShapeDef shapeDef = b2DefaultShapeDef();

      shapeDef.density = b2Shape_GetDensity(sourceShapeId);
      shapeDef.friction = b2Shape_GetFriction(sourceShapeId);
      shapeDef.restitution = b2Shape_GetRestitution(sourceShapeId);
      shapeDef.filter = b2Shape_GetFilter(sourceShapeId);
      shapeDef.isSensor = b2Shape_IsSensor(sourceShapeId);
      shapeDef.userData = b2Shape_GetUserData(sourceShapeId);

      b2ShapeId newShapeId;
      b2ShapeType shapeType = b2Shape_GetType(sourceShapeId);

      switch (shapeType) {
      case b2_circleShape: {
        b2Circle circle = b2Shape_GetCircle(sourceShapeId);
        newShapeId = b2CreateCircleShape(newBodyId, &shapeDef, &circle);
        break;
      }
      case b2_capsuleShape: {
        b2Capsule capsule = b2Shape_GetCapsule(sourceShapeId);
        newShapeId = b2CreateCapsuleShape(newBodyId, &shapeDef, &capsule);
        break;
      }
      case b2_segmentShape: {
        b2Segment segment = b2Shape_GetSegment(sourceShapeId);
        newShapeId = b2CreateSegmentShape(newBodyId, &shapeDef, &segment);
        break;
      }
      case b2_polygonShape: {
        b2Polygon polygon = b2Shape_GetPolygon(sourceShapeId);
        newShapeId = b2CreatePolygonShape(newBodyId, &shapeDef, &polygon);
        break;
      }
      default:
        continue;
      }

      if (b2Shape_IsValid(newShapeId)) {
        b2Shape_EnableSensorEvents(newShapeId, b2Shape_AreSensorEventsEnabled(sourceShapeId));
        b2Shape_EnableContactEvents(newShapeId, b2Shape_AreContactEventsEnabled(sourceShapeId));
        b2Shape_EnablePreSolveEvents(newShapeId, b2Shape_ArePreSolveEventsEnabled(sourceShapeId));
        b2Shape_EnableHitEvents(newShapeId, b2Shape_AreHitEventsEnabled(sourceShapeId));
      }
    }
  }
  return newBodyId;
}

fan::vec2 fan::physics::physics_to_render(const fan::vec2& p){
  return p * fan::physics::length_units_per_meter;
}

fan::vec2 fan::physics::render_to_physics(const fan::vec2& p){
  return p / fan::physics::length_units_per_meter;
}

