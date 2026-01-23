module;

#if defined(FAN_PHYSICS_2D)
#include <fan/utility.h>
#include <box2d/box2d.h>
#endif

#include <utility>
#include <functional>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <array>
#include <sstream>

module fan.physics.b2_integration;

import fan.utility;

#if defined(FAN_PHYSICS_2D)
namespace fan::physics {
  circle_t::circle_t(const b2Circle& circle) : b2Circle(circle) {}

  joint_id_t::joint_id_t() : b2JointId(b2_nullJointId) {}

  joint_id_t::joint_id_t(const b2JointId& body_id) : b2JointId(body_id) {}

  void joint_id_t::set_joint(const joint_id_t& b) {
    *this = b;
  }

  bool joint_id_t::operator==(const joint_id_t& b) const {
    b2JointId a = *this;
    return B2_ID_EQUALS(a, b);
  }

  bool joint_id_t::operator!=(const joint_id_t& b) const {
    return !this->operator==(b);
  }

  bool joint_id_t::is_valid() {
    return *this != b2_nullJointId;
  }

  void joint_id_t::invalidate() {
    *this = b2_nullJointId;
  }

  void joint_id_t::destroy() {
    if (is_valid() == false) {
      return;
    }
    b2JointId id = *this;
    b2DestroyJoint(id);
    invalidate();
  }

  void joint_id_t::revolute_joint_set_motor_speed(f32_t v) {
    queue_one_time_command([id = *this, v]() {
      b2RevoluteJoint_SetMotorSpeed(id, v);
    });
  }

  ray_result_t::operator bool() {
    return hit;
  }

  shape_id_t::shape_id_t() : b2ShapeId(b2_nullShapeId) {}

  shape_id_t::shape_id_t(const b2ShapeId& shape_id) : b2ShapeId(shape_id) {}

  void shape_id_t::set_friction(f32_t friction) {
    queue_one_time_command([id = *this, friction]() {
      b2Shape_SetFriction(id, friction);
    });
  }

  bool shape_id_t::operator==(const shape_id_t& shape) const {
    return B2_ID_EQUALS(static_cast<b2ShapeId>(*this), shape);
  }

  bool shape_id_t::operator!=(const shape_id_t& shape) const {
    return !operator==(shape);
  }

  bool shape_id_t::is_valid() const {
    return b2Shape_IsValid(static_cast<const b2ShapeId&>(*this));
  }

  shape_id_t::operator bool() const {
    return is_valid();
  }

  body_id_t::body_id_t() : b2BodyId(b2_nullBodyId) {}

  body_id_t::body_id_t(const b2BodyId& body_id) : b2BodyId(body_id) {}

  void body_id_t::set_body(const body_id_t& b) {
    *this = b;
  }

  bool body_id_t::operator==(const body_id_t& b) const {
    return B2_ID_EQUALS(static_cast<const b2BodyId&>(*this),
      static_cast<const b2BodyId&>(b));
  }

  bool body_id_t::operator!=(const body_id_t& b) const {
    return !(*this == b);
  }

  body_id_t::operator bool() const {
    return is_valid();
  }

  body_id_t::operator b2ShapeId() const {
    return get_shape_id();
  }

  bool body_id_t::is_valid() const {
    return b2Body_IsValid(static_cast<const b2BodyId&>(*this));
  }

  void body_id_t::invalidate() {
    *this = b2_nullBodyId;
  }

  void body_id_t::destroy() {
    if (!is_valid()) return;
    b2BodyId id = *this;

    gphysics()->sensor_events.remove_body_contacts(id);
    b2DestroyBody(id);
    invalidate();
  }

  void body_id_t::erase() {
    destroy();
  }

  fan::vec2 body_id_t::get_linear_velocity() const {
    return fan::vec2(b2Body_GetLinearVelocity(*this)) * length_units_per_meter;
  }

  void body_id_t::set_linear_velocity(const fan::vec2& v) {
    b2Body_SetLinearVelocity(*this, v / length_units_per_meter);
  }

  void body_id_t::set_linear_damping(f32_t v) {
    b2Body_SetLinearDamping(*this, v);
  }

  f32_t body_id_t::get_angular_velocity() const {
    return b2Body_GetAngularVelocity(*this);
  }

  void body_id_t::set_angular_velocity(f32_t v) {
    b2Body_SetAngularVelocity(*this, v);
  }

  void body_id_t::apply_force_center(const fan::vec2& v) {
    b2Body_ApplyForceToCenter(*this, v / length_units_per_meter, true);
  }

  void body_id_t::apply_linear_impulse_center(const fan::vec2& v) {
    b2Body_ApplyLinearImpulseToCenter(*this, v / length_units_per_meter, true);
  }

  void body_id_t::zero_linear_impulse_center() {
    b2Body_SetLinearVelocity(*this, {0, 0});
  }

  void body_id_t::apply_angular_impulse(f32_t v) {
    b2Body_ApplyAngularImpulse(*this, v / (length_units_per_meter * length_units_per_meter), true);
  }

  fan::vec2 body_id_t::get_physics_position() const {
    return physics_to_render(b2Body_GetPosition(*this));
  }

  fan::vec2 body_id_t::get_position() const {
    return get_physics_position();
  }

  void body_id_t::set_physics_position(const fan::vec2& p) {
    b2Rot rotation = b2Body_GetRotation(*this);
    b2Body_SetTransform(*this, p / length_units_per_meter, rotation);
  }

  shape_id_t body_id_t::get_shape_id() const {
    b2ShapeId shape_id = b2_nullShapeId;
  #if FAN_DEBUG >= fan_debug_medium
    if (!b2Body_GetShapes(*this, &shape_id, 1)) {
      fan::throw_error();
    }
  #else
    b2Body_GetShapes(*this, &shape_id, 1);
  #endif
    return shape_id;
  }

  f32_t body_id_t::get_density() const {
    return b2Shape_GetDensity(get_shape_id());
  }

  f32_t body_id_t::get_friction() const {
    return b2Shape_GetFriction(get_shape_id());
  }

  f32_t body_id_t::get_mass() const {
    return b2Shape_GetMassData(get_shape_id()).mass * length_units_per_meter;
  }
  void body_id_t::set_mass(f32_t mass) {
    b2MassData md = b2Body_GetMassData(*this);
    md.mass = mass / length_units_per_meter;
    b2Body_SetMassData(*this, md);
  }

  f32_t body_id_t::get_restitution() const {
    return b2Shape_GetRestitution(get_shape_id());
  }

  void body_id_t::set_restitution(f32_t restitution) {
    b2Shape_SetRestitution(*this, restitution);
  }

  aabb_t body_id_t::get_aabb() const {
    b2AABB aabb = b2Shape_GetAABB(get_shape_id());
    return {
      physics_to_render(fan::vec2(aabb.lowerBound.x, aabb.lowerBound.y)),
      physics_to_render(fan::vec2(aabb.upperBound.x, aabb.upperBound.y))
    };
  }

  fan::vec2 body_id_t::get_size() const {
    aabb_t aabb = get_aabb();
    fan::vec2 size = aabb.max - aabb.min;
    return size;
  }

  void body_id_t::set_body_type(uint8_t body_type) {
    b2Body_SetType(*this, (b2BodyType)body_type);
  }

  bool body_id_t::test_overlap(const body_id_t& other) const {
    return physics::test_overlap(*this, other);
  }

  void body_id_t::on_overlap(const body_id_t& other, std::function<void()> callback) {
    physics::on_overlap(*this, other, std::move(callback));
  }

  f32_t body_id_t::get_gravity_scale() const {
    return b2Body_GetGravityScale(*this);
  }
  void body_id_t::set_gravity_scale(f32_t scale) {
    b2Body_SetGravityScale(*this, scale);
  }

  capsule_t::capsule_t(const b2Capsule& capsule) : b2Capsule(capsule) {}

  segment_t::segment_t(const b2Segment& segment) : b2Segment(segment) {}

  polygon_t::polygon_t(const b2Polygon& polygon) : b2Polygon(polygon) {}

  std::size_t b2_body_id_hash_t::operator()(const b2BodyId& id) const {
    return std::hash<uint64_t>{}(
      (uint64_t(id.index1) << 32) | (uint64_t(id.world0) << 16) | id.generation
      );
  }

  context_t::operator b2WorldId& () {
    return world_id;
  }

  context_t::context_t(const properties_t& properties) {
  #if defined(fan_std23)
    b2SetAllocator(
      [](unsigned int size, int a) { return fan::heap_profiler_t::instance().allocate_memory(size); },
      [](void* mem) { fan::heap_profiler_t::instance().deallocate_memory(mem); }
    );
  #endif
    gphysics() = this;
    b2WorldDef world_def = b2DefaultWorldDef();
    world_def.gravity = properties.gravity;

    b2SetLengthUnitsPerMeter(1.f / length_units_per_meter);
    world_id = b2CreateWorld(&world_def);

    gphysics().get_gravity = [this]() -> fan::vec2 {
      return get_gravity();
    };
    gphysics().set_gravity = [this](const fan::vec2& new_gravity) -> void {
      return set_gravity(new_gravity);
    };
  }

  void context_t::set_gravity(const fan::vec2& gravity) {
    b2World_SetGravity(world_id, gravity / length_units_per_meter);
  }

  fan::vec2 context_t::get_gravity() const {
    return b2World_GetGravity(world_id) * length_units_per_meter;
  }

  void context_t::begin_frame(f32_t dt) {
    delta_time = dt;
  }

  entity_t context_t::create_box(const fan::vec2& position, const fan::vec2& size, f32_t angle, uint8_t body_type, const shape_properties_t& shape_properties) {
    polygon_t shape = b2MakeBox(size.x / length_units_per_meter * shape_properties.collision_multiplier.x, size.y / length_units_per_meter * shape_properties.collision_multiplier.y);
    entity_t entity;
    b2BodyDef body_def = b2DefaultBodyDef();
    body_def.position = position / length_units_per_meter;
    body_def.rotation = b2MakeRot(angle);
    body_def.type = (b2BodyType)body_type;
    body_def.fixedRotation = shape_properties.fixed_rotation;
    body_def.linearDamping = shape_properties.linear_damping;
    body_def.angularDamping = shape_properties.angular_damping;
    body_def.allowFastRotation = shape_properties.fast_rotation;
    entity = b2CreateBody(world_id, &body_def);
  #if FAN_DEBUG >= fan_debug_medium
    if (entity.is_valid() == false) {
      fan::throw_error();
    }
  #endif
    b2ShapeDef shape_def = b2DefaultShapeDef();
    shape_def.enablePreSolveEvents = shape_properties.presolve_events;
    shape_def.enableContactEvents = shape_properties.contact_events;
    shape_def.density = shape_properties.density;
    shape_def.material.friction = shape_properties.friction;
    shape_def.material.restitution = shape_properties.restitution;
    shape_def.isSensor = shape_properties.is_sensor;
    shape_def.enableSensorEvents = true;
    shape_def.filter = shape_properties.filter;
    b2CreatePolygonShape(entity, &shape_def, &shape);
    return entity;
  }

  entity_t context_t::create_rectangle(const fan::vec2& position, const fan::vec2& size, f32_t angle, uint8_t body_type, const shape_properties_t& shape_properties) {
    return create_box(position, size, angle, body_type, shape_properties);
  }

  entity_t context_t::create_circle(const fan::vec2& position, f32_t radius, f32_t angle, uint8_t body_type, const shape_properties_t& shape_properties) {
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
    body_def.angularDamping = shape_properties.angular_damping;
    body_def.allowFastRotation = shape_properties.fast_rotation;

    entity = b2CreateBody(world_id, &body_def);
  #if FAN_DEBUG >= fan_debug_medium
    if (entity.is_valid() == false) {
      fan::throw_error();
    }
  #endif

    b2ShapeDef shape_def = b2DefaultShapeDef();
    shape_def.enablePreSolveEvents = shape_properties.presolve_events;
    shape_def.enableContactEvents = shape_properties.contact_events;
    shape_def.density = shape_properties.density;
    shape_def.material.friction = shape_properties.friction;
    shape_def.material.restitution = shape_properties.restitution;
    shape_def.isSensor = shape_properties.is_sensor;
    shape_def.enableSensorEvents = true;
    shape_def.filter = shape_properties.filter;

    b2CreateCircleShape(entity, &shape_def, &shape);
    return entity;
  }

  entity_t context_t::create_capsule(const fan::vec2& position, f32_t angle, const capsule_t& info, uint8_t body_type, const shape_properties_t& shape_properties) {
    capsule_t shape = info;
    shape.center1.x /= length_units_per_meter / shape_properties.collision_multiplier.x;
    shape.center1.y /= length_units_per_meter / shape_properties.collision_multiplier.y;
    shape.center2.x /= length_units_per_meter / shape_properties.collision_multiplier.x;
    shape.center2.y /= length_units_per_meter / shape_properties.collision_multiplier.y;
    shape.radius /= length_units_per_meter / shape_properties.collision_multiplier.x;

    entity_t entity;
    b2BodyDef body_def = b2DefaultBodyDef();
    body_def.position = position / length_units_per_meter;
    body_def.rotation.c = std::cos(-angle);
    body_def.rotation.s = std::sin(-angle);
    body_def.type = (b2BodyType)body_type;
    body_def.fixedRotation = shape_properties.fixed_rotation;
    body_def.linearDamping = shape_properties.linear_damping;
    body_def.angularDamping = shape_properties.angular_damping;
    body_def.allowFastRotation = shape_properties.fast_rotation;
    entity = b2CreateBody(world_id, &body_def);
  #if FAN_DEBUG >= fan_debug_medium
    if (entity.is_valid() == false) {
      fan::throw_error();
    }
  #endif
    b2ShapeDef shape_def = b2DefaultShapeDef();
    shape_def.enablePreSolveEvents = shape_properties.presolve_events;
    shape_def.enableContactEvents = shape_properties.contact_events;
    shape_def.density = shape_properties.density;
    shape_def.material.friction = shape_properties.friction;
    shape_def.material.restitution = shape_properties.restitution;
    shape_def.isSensor = shape_properties.is_sensor;
    shape_def.enableSensorEvents = true;
    shape_def.filter = shape_properties.filter;
    b2CreateCapsuleShape(entity, &shape_def, &shape);
    return entity;
  }

  entity_t context_t::create_segment(const fan::vec2& position, const std::vector<fan::vec2>& points, uint8_t body_type, const shape_properties_t& shape_properties) {
    entity_t entity;
    b2BodyDef body_def = b2DefaultBodyDef();
    body_def.position = position / length_units_per_meter;
    body_def.type = (b2BodyType)body_type;
    body_def.fixedRotation = shape_properties.fixed_rotation;
    body_def.linearDamping = shape_properties.linear_damping;
    body_def.angularDamping = shape_properties.angular_damping;
    body_def.allowFastRotation = shape_properties.fast_rotation;
    entity = b2CreateBody(world_id, &body_def);
  #if FAN_DEBUG >= fan_debug_medium
    if (entity.is_valid() == false) {
      fan::throw_error();
    }
  #endif
    b2ShapeDef shape_def = b2DefaultShapeDef();
    shape_def.enablePreSolveEvents = shape_properties.presolve_events;
    shape_def.enableContactEvents = shape_properties.contact_events;
    shape_def.density = shape_properties.density;
    shape_def.material.friction = shape_properties.friction;
    shape_def.material.restitution = shape_properties.restitution;
    shape_def.isSensor = shape_properties.is_sensor;
    shape_def.enableSensorEvents = true;
    shape_def.filter = shape_properties.filter;

    for (std::size_t i = 0; i < points.size() - 1; ++i) {
      segment_t shape;
      shape.point1 = points[i] / length_units_per_meter;
      shape.point2 = points[i + 1] / length_units_per_meter;
      b2CreateSegmentShape(entity, &shape_def, &shape);
    }
    if (points.size() > 2) {
      segment_t shape;
      shape.point1 = points.back() / length_units_per_meter;
      shape.point2 = points.front() / length_units_per_meter;
      b2CreateSegmentShape(entity, &shape_def, &shape);
    }
    return entity;
  }

  entity_t context_t::create_polygon(const fan::vec2& position, f32_t radius, const fan::vec2* points, int count, uint8_t body_type, const shape_properties_t& shape_properties) {
    entity_t entity;
    b2BodyDef body_def = b2DefaultBodyDef();
    body_def.position = position / length_units_per_meter;
    body_def.type = (b2BodyType)body_type;
    body_def.fixedRotation = shape_properties.fixed_rotation;
    body_def.linearDamping = shape_properties.linear_damping;
    body_def.angularDamping = shape_properties.angular_damping;
    body_def.allowFastRotation = shape_properties.fast_rotation;
    entity = b2CreateBody(world_id, &body_def);

  #if FAN_DEBUG >= fan_debug_medium
    if (entity.is_valid() == false) {
      fan::throw_error();
    }
  #endif
    b2ShapeDef shape_def = b2DefaultShapeDef();
    shape_def.enablePreSolveEvents = shape_properties.presolve_events;
    shape_def.enableContactEvents = shape_properties.contact_events;
    shape_def.density = shape_properties.density;
    shape_def.material.friction = shape_properties.friction;
    shape_def.material.restitution = shape_properties.restitution;
    shape_def.isSensor = shape_properties.is_sensor;
    shape_def.enableSensorEvents = true;
    shape_def.filter = shape_properties.filter;

    b2Vec2 b2_points[B2_MAX_POLYGON_VERTICES];
    int n = count;

    for (int i = 0; i < n; i++) {
      b2_points[i] = points[i] / length_units_per_meter;
    }

    b2Hull hull = b2ComputeHull(b2_points, n);
    b2Polygon polygon = b2MakePolygon(&hull, radius);

    b2CreatePolygonShape(entity, &shape_def, &polygon);
    return entity;
  }


  // a, b and c are local offsets from 'position' (center)
  entity_t context_t::create_triangle(const fan::vec2& position, const fan::vec2& a, const fan::vec2& b, const fan::vec2& c, uint8_t body_type, const shape_properties_t& shape_properties) {
    fan::vec2 pts[] = {a, b, c};
    return create_polygon(position, 0.0f, pts, std::size(pts), body_type, shape_properties);
  }

  void context_t::step(f32_t dt) {
    static f32_t accumulator = 0.0f;
    static constexpr f32_t max_frame_time = 1.f;
    dt = std::min(dt, max_frame_time);
    accumulator += dt;

    f32_t physics_timestep = default_physics_timestep;

    while (accumulator >= physics_timestep) {
      {
        auto it = physics_step_callbacks.GetNodeFirst();
        while (it != physics_step_callbacks.dst) {
          physics_step_callbacks.StartSafeNext(it);
          physics_step_callbacks[it]();
          it = physics_step_callbacks.EndSafeNext();
        }
      }

      for (auto& command : one_time_commands) {
        command();
      }
      one_time_commands.clear();

      process_collision_events();
      b2World_Step(world_id, physics_timestep, 4);
      sensor_events.update(world_id);
      accumulator -= physics_timestep;
    }
    {
      auto it = physics_updates.GetNodeFirst();
      while (it != physics_updates.dst) {
        physics_updates.StartSafeNext(it);
        ((fan::physics::shape_physics_update_cb)physics_updates[it].cb)(physics_updates[it]);
        it = physics_updates.EndSafeNext();
      }
    }
  }

  bool context_t::is_on_sensor(body_id_t test_id, body_id_t sensor_id) const {
    return sensor_events.is_on_sensor(test_id, sensor_id);
  }

  ray_result_t context_t::raycast(const fan::vec2& src_, const fan::vec2& dst_) {
    fan::vec2 src = src_ / length_units_per_meter;
    fan::vec2 dst = dst_ / length_units_per_meter;
    b2QueryFilter qf = b2DefaultQueryFilter();

    b2Vec2 translation = dst - src;

    b2RayResult b2result = b2World_CastRayClosest(world_id, src, translation, qf);
    ray_result_t result;
    result.shapeId = b2result.shapeId;
    result.point = b2result.point;
    result.normal = b2result.normal;
    result.fraction = b2result.fraction;
    result.hit = b2result.hit;
    result.point *= length_units_per_meter;
    return result;
  }

  void context_t::on_begin_touch(b2ShapeId shape_a, b2ShapeId shape_b) {
    add_collision(shape_a, shape_b);
  }

  void context_t::on_end_touch(b2ShapeId shape_a, b2ShapeId shape_b) {
    remove_collision(shape_a, shape_b);
  }

  void context_t::on_hit(b2ShapeId shape_a, b2ShapeId shape_b, f32_t approach_speed) {}

  uint64_t context_t::get_shape_key(b2ShapeId shape) const {
    return (uint64_t(shape.index1) << 32) | (uint64_t(shape.world0) << 16) | uint64_t(shape.generation);
  }

  void context_t::add_collision(b2ShapeId a, b2ShapeId b) {
    auto pair = std::minmax(get_shape_key(a), get_shape_key(b));
    active_collisions.insert(pair);
  }

  void context_t::remove_collision(b2ShapeId a, b2ShapeId b) {
    auto pair = std::minmax(get_shape_key(a), get_shape_key(b));
    active_collisions.erase(pair);
  }

  void context_t::process_collision_events() {
    b2ContactEvents contact_events = b2World_GetContactEvents(world_id);

    for (int i = 0; i < contact_events.beginCount; ++i) {
      const b2ContactBeginTouchEvent& event = contact_events.beginEvents[i];
      on_begin_touch(event.shapeIdA, event.shapeIdB);
    }

    for (int i = 0; i < contact_events.endCount; ++i) {
      const b2ContactEndTouchEvent& event = contact_events.endEvents[i];
      on_end_touch(event.shapeIdA, event.shapeIdB);
    }

    for (int i = 0; i < contact_events.hitCount; ++i) {
      const b2ContactHitEvent& event = contact_events.hitEvents[i];
      on_hit(event.shapeIdA, event.shapeIdB, event.approachSpeed);
    }
  }

  bool context_t::is_colliding(b2ShapeId a, b2ShapeId b) const {
    auto pair = std::minmax(get_shape_key(a), get_shape_key(b));
    return active_collisions.count(pair) > 0;
  }

  entity_t context_t::create_sensor_circle(const fan::vec2& position, f32_t radius) {
    return create_circle(
      position,
      radius,
      0,
      body_type_e::static_body,
      shape_properties_t {.is_sensor = true}
    );
  }

  entity_t context_t::create_sensor_rectangle(const fan::vec2& position, const fan::vec2& size) {
    return create_box(
      position,
      size,
      0,
      body_type_e::static_body,
      shape_properties_t {.is_sensor = true}
    );
  }

  physics_update_cbs_t::nr_t context_t::add_physics_update(const physics_update_data_t& cb_data) {
    auto it = gphysics()->physics_updates.NewNodeLast();
    gphysics()->physics_updates[it] = (fan::physics::physics_update_data_t)cb_data;
    return it;
  }

  fan::physics::physics_update_cbs_t::nd_t& context_t::get_physics_update_data(fan::physics::physics_update_cbs_t::nr_t nr) {
    return gphysics()->physics_updates[nr];
  }

  void context_t::remove_physics_update(physics_update_cbs_t::nr_t nr) {
    gphysics()->physics_updates.unlrec(nr);
  }

  void sensor_events_t::update(b2WorldId world_id) {
    b2SensorEvents sensor_events = b2World_GetSensorEvents(world_id);

    for (int i = 0; i < sensor_events.beginCount; ++i) {
      b2SensorBeginTouchEvent ev = sensor_events.beginEvents[i];
      if (b2Shape_IsValid(ev.sensorShapeId) && b2Shape_IsValid(ev.visitorShapeId)) {
        begin_touch_event_cb(ev);
        update_contact(b2Shape_GetBody(ev.sensorShapeId), b2Shape_GetBody(ev.visitorShapeId), true);
      }
    }
    for (int i = 0; i < sensor_events.endCount; ++i) {
      b2SensorEndTouchEvent ev = sensor_events.endEvents[i];
      if (b2Shape_IsValid(ev.sensorShapeId) && b2Shape_IsValid(ev.visitorShapeId)) {
        end_touch_event_cb(ev);
        update_contact(b2Shape_GetBody(ev.sensorShapeId), b2Shape_GetBody(ev.visitorShapeId), false);
      }
    }
    contacts.erase(
      std::remove_if(contacts.begin(), contacts.end(),
        [](const sensor_contact_t& c) { return !c.is_in_contact; }),
      contacts.end()
    );
  }

  void sensor_events_t::update_contact(b2BodyId sensor_id, b2BodyId object_id, bool is_in_contact) {
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

  void sensor_events_t::remove_body_contacts(b2BodyId body_id) {
    contacts.erase(
      std::remove_if(contacts.begin(), contacts.end(),
        [body_id](const sensor_contact_t& c) {
      return B2_ID_EQUALS(c.object_id, body_id) || B2_ID_EQUALS(c.sensor_id, body_id);
    }),
      contacts.end()
    );
  }

  bool sensor_events_t::is_on_sensor(body_id_t test_id, body_id_t sensor_id) const {
    for (const auto& contact : contacts) {
      if (B2_ID_EQUALS(contact.sensor_id, sensor_id) && B2_ID_EQUALS(contact.object_id, test_id)) {
        return contact.is_in_contact;
      }
    }
    return false;
  }

  bool b2_body_id_equal_t::operator()(const b2BodyId& a, const b2BodyId& b) const {
    return a.index1 == b.index1 && a.world0 == b.world0 && a.generation == b.generation;
  }

  context_t::properties_t::properties_t() {}

  size_t context_t::pair_hash_t::operator()(const std::pair<uint64_t, uint64_t>& p) const {
    return std::hash<uint64_t>{}(p.first) ^ (std::hash<uint64_t>{}(p.second) << 1);
  }

  void fill_shape_proxy(b2ShapeProxy& proxy, b2ShapeId shape_id, body_id_t body_id) {
    b2ShapeType shape_type = b2Shape_GetType(shape_id);
    b2Transform transform = b2Body_GetTransform(body_id);

    switch (shape_type) {
    case b2_circleShape:
    {
      b2Circle circle = b2Shape_GetCircle(shape_id);
      proxy.points[0] = b2TransformPoint(transform, circle.center);
      proxy.count = 1;
      proxy.radius = circle.radius;
      break;
    }
    case b2_capsuleShape:
    {
      b2Capsule capsule = b2Shape_GetCapsule(shape_id);
      proxy.points[0] = b2TransformPoint(transform, capsule.center1);
      proxy.points[1] = b2TransformPoint(transform, capsule.center2);
      proxy.count = 2;
      proxy.radius = capsule.radius;
      break;
    }
    case b2_polygonShape:
    {
      b2Polygon polygon = b2Shape_GetPolygon(shape_id);
      proxy.count = polygon.count;
      for (int i = 0; i < polygon.count; ++i) {
        proxy.points[i] = b2TransformPoint(transform, polygon.vertices[i]);
      }
      proxy.radius = polygon.radius;
      break;
    }
    case b2_segmentShape:
    {
      b2Segment segment = b2Shape_GetSegment(shape_id);
      proxy.points[0] = b2TransformPoint(transform, segment.point1);
      proxy.points[1] = b2TransformPoint(transform, segment.point2);
      proxy.count = 2;
      proxy.radius = 0.0f;
      break;
    }
    }
  }

  bool test_overlap(body_id_t body_a, body_id_t body_b) {
    if (!body_a.is_valid() || !body_b.is_valid()) {
      return false;
    }

    b2ShapeId shape_b = body_b.get_shape_id();
    if (!b2Shape_IsValid(shape_b)) {
      return false;
    }

    b2ShapeProxy proxy;
    fill_shape_proxy(proxy, shape_b, body_b);

    overlap_test_context_t context;
    context.target_shape = body_a.get_shape_id();
    context.found_overlap = false;

    b2QueryFilter filter = b2DefaultQueryFilter();
    b2World_OverlapShape(gphysics()->world_id, &proxy, filter, overlap_result_callback, &context);
    return context.found_overlap;
  }

  // point size = half size
  bool is_point_overlapping(const fan::vec2& position, const fan::vec2& point_size) {
    b2AABB aabb;
    aabb.lowerBound = fan::physics::render_to_physics(position) - fan::physics::render_to_physics(point_size);
    aabb.upperBound = fan::physics::render_to_physics(position) + fan::physics::render_to_physics(point_size);
    b2QueryFilter filter = b2DefaultQueryFilter();
    bool hit = false;
    b2World_OverlapAABB(gphysics()->world_id, aabb, filter, [](b2ShapeId shapeId, void* ctx) { return *(bool*)ctx = true; }, &hit);
    return hit;
  }

  void on_overlap(body_id_t body_a, body_id_t body_b, std::function<void()> callback) {
    if (!body_a.is_valid() || !body_b.is_valid()) {
      return;
    }

    b2ShapeId shape_b = body_b.get_shape_id();
    if (!b2Shape_IsValid(shape_b)) {
      return;
    }

    b2ShapeProxy proxy;
    fill_shape_proxy(proxy, shape_b, body_b);

    overlap_callback_context_t context;
    context.callback = std::move(callback);
    context.target_shape = body_a.get_shape_id();

    b2QueryFilter filter = b2DefaultQueryFilter();
    b2World_OverlapShape(gphysics()->world_id, &proxy, filter, overlap_callback_fcn, &context);
  }
  void queue_one_time_command(std::function<void()> callback) {
    one_time_commands.push_back(std::move(callback));
  }
  fan::vec2 check_wall_contact(body_id_t body_id, shape_id_t* colliding_wall) {
    if (!body_id.is_valid()) {
      return {0, 0};
    }

    b2ContactData contacts[16];
    int contact_count = b2Body_GetContactData(body_id, contacts, 16);

    for (int i = 0; i < contact_count; ++i) {
      const b2ContactData& contact = contacts[i];
      fan::vec2 normal = contact.manifold.normal;

      b2BodyId body_a = b2Shape_GetBody(contact.shapeIdA);
      f32_t sign = B2_ID_EQUALS(body_a, body_id) ? 1.0f : -1.0f;
      normal = normal * sign;

      if (colliding_wall) {
        *colliding_wall = contact.shapeIdA;
      }
      if (std::abs(normal.x) > 0.7f && contact.manifold.pointCount > 0) {
        return normal;
      }
    }

    return {0, 0};
  }
 bool is_on_ground(fan::physics::body_id_t main, bool jumping, fan::physics::body_id_t* feet) {
  for (int i = 0; i < 2; ++i) {
    fan::physics::body_id_t body_id = feet && feet[i].is_valid() ? feet[i] : main;
    b2Vec2 velocity = b2Body_GetLinearVelocity(body_id);
    
    if (velocity.y > -5.0f) {
      int capacity = b2Body_GetContactCapacity(body_id);
      capacity = b2MinInt(capacity, 4);
      b2ContactData contactData[4];
      int count = b2Body_GetContactData(body_id, contactData, capacity);
      
      for (int i = 0; i < count; ++i) {
        b2BodyId bodyIdA = b2Shape_GetBody(contactData[i].shapeIdA);
        f32_t sign = 0.0f;
        if (B2_ID_EQUALS(bodyIdA, body_id)) {
          sign = -1.0f;
        }
        else {
          sign = 1.0f;
        }
        if (sign * contactData[i].manifold.normal.y < -0.9f) {
          return true;
        }
      }
    }
  }
  return false;
}
  void apply_wall_slide(body_id_t body_id, const fan::vec2& wall_normal, f32_t slide_speed) {
    if (!wall_normal) {
      return;
    }

    fan::vec2 velocity = body_id.get_linear_velocity();
    f32_t mass = body_id.get_mass();

    if (velocity.y > slide_speed) {
      f32_t delta_v = slide_speed - velocity.y;
      delta_v = std::max(delta_v, -velocity.y);
      fan::vec2 impulse = fan::vec2(0, delta_v / mass);
      body_id.apply_linear_impulse_center(impulse);
    }
  }
  bool wall_jump(body_id_t body_id, const fan::vec2& wall_normal, f32_t push_x, f32_t max_up_speed) {
    if (!wall_normal) return false;

    f32_t desired_y_physics = -max_up_speed;
    fan::vec2 vel = body_id.get_linear_velocity();
    body_id.set_linear_velocity(fan::vec2(vel.x, vel.y >= 0 ? 0 : vel.y));
    body_id.apply_linear_impulse_center({push_x, desired_y_physics});
    return true;
  }
  bool overlap_callback_fcn(b2ShapeId shape_id, void* context) {
    overlap_callback_context_t* ctx = static_cast<overlap_callback_context_t*>(context);
    if (B2_ID_EQUALS(shape_id, ctx->target_shape)) {
      ctx->callback();
      return false;
    }
    return true;
  }
  bool is_colliding(const b2ShapeId& a, const b2ShapeId& b) {
    return gphysics()->is_colliding(a, b);
  }
  fan::physics::entity_t create_sensor_circle(const fan::vec2& position, f32_t radius) {
    return gphysics()->create_sensor_circle(position, radius);
  }
  fan::physics::entity_t create_sensor_rectangle(const fan::vec2& position, const fan::vec2& size) {
    return gphysics()->create_sensor_rectangle(position, size);
  }
  step_callback_nr_t add_physics_step_callback(std::function<void()> callback) {
    return fan::add_bll_raii_struct_cb<
      fan::physics::context_t
    >(gphysics(), &fan::physics::context_t::physics_step_callbacks, std::move(callback));
  }
  void remove_physics_step_callback(step_callback_nr_t nr) {
    gphysics()->physics_step_callbacks.unlrec(nr);
  }
  bool presolve_oneway_collision(b2ShapeId shapeIdA, b2ShapeId shapeIdB, b2Manifold* manifold, fan::physics::body_id_t character_body) {
    if (!b2Shape_IsValid(shapeIdA)) {
      fan::throw_error("Shape invalid");
    }
    if (!b2Shape_IsValid(shapeIdB)) {
      fan::throw_error("Shape invalid");
    }

    f32_t sign = 0.0f;
    if (B2_ID_EQUALS(shapeIdA, character_body)) {
      sign = 1.0f;
    }
    else if (B2_ID_EQUALS(shapeIdB, character_body)) {
      sign = -1.0f;
    }
    else {
      return true;
    }

    b2Vec2 normal = manifold->normal;
    if (sign * normal.y > 0.95f) {
      return true;
    }

    f32_t separation = 0.0f;
    for (int i = 0; i < manifold->pointCount; ++i) {
      f32_t s = manifold->points[i].separation;
      separation = separation < s ? separation : s;
    }

    if (separation > 0.1f * 64.f) {
      return true;
    }

    return false;
  }
  fan::physics::body_id_t deep_copy_body(b2WorldId worldId, fan::physics::body_id_t sourceBodyId) {
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

    const int shapeCount = b2Body_GetShapeCount(sourceBodyId);
    if (shapeCount > 0) {
      std::vector<b2ShapeId> shapes(shapeCount);
      b2Body_GetShapes(sourceBodyId, shapes.data(), shapeCount);

      for (b2ShapeId sourceShapeId : shapes) {
        b2ShapeDef shape_def = b2DefaultShapeDef();

        shape_def.density = b2Shape_GetDensity(sourceShapeId);
        shape_def.material.friction = b2Shape_GetFriction(sourceShapeId);
        shape_def.material.restitution = b2Shape_GetRestitution(sourceShapeId);
        shape_def.filter = b2Shape_GetFilter(sourceShapeId);
        shape_def.isSensor = b2Shape_IsSensor(sourceShapeId);
        shape_def.enableSensorEvents = true;
        shape_def.userData = b2Shape_GetUserData(sourceShapeId);

        b2ShapeId newShapeId;
        b2ShapeType shapeType = b2Shape_GetType(sourceShapeId);

        switch (shapeType) {
        case b2_circleShape:
        {
          b2Circle circle = b2Shape_GetCircle(sourceShapeId);
          newShapeId = b2CreateCircleShape(newBodyId, &shape_def, &circle);
          break;
        }
        case b2_capsuleShape:
        {
          b2Capsule capsule = b2Shape_GetCapsule(sourceShapeId);
          newShapeId = b2CreateCapsuleShape(newBodyId, &shape_def, &capsule);
          break;
        }
        case b2_segmentShape:
        {
          b2Segment segment = b2Shape_GetSegment(sourceShapeId);
          newShapeId = b2CreateSegmentShape(newBodyId, &shape_def, &segment);
          break;
        }
        case b2_polygonShape:
        {
          b2Polygon polygon = b2Shape_GetPolygon(sourceShapeId);
          newShapeId = b2CreatePolygonShape(newBodyId, &shape_def, &polygon);
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
    b2MassData massData = b2Body_GetMassData(sourceBodyId);
    b2Body_SetMassData(newBodyId, massData);
    return newBodyId;
  }

  void set_pre_solve_callback(b2WorldId world_id, b2PreSolveFcn* fcn, void* context) {
    b2World_SetPreSolveCallback(world_id, fcn, context);
  }

  // for drawing physics shapes
  fan::physics::physics_update_cbs_t::nr_t add_physics_update(const fan::physics::physics_update_data_t& cb_data) {
    return gphysics()->add_physics_update(cb_data);
  }
  fan::physics::physics_update_cbs_t::nd_t& get_physics_update_data(fan::physics::physics_update_cbs_t::nr_t nr) {
    return gphysics()->get_physics_update_data(nr);
  }
  void remove_physics_update(fan::physics::physics_update_cbs_t::nr_t nr) {
    gphysics()->remove_physics_update(nr);
  }
  bool overlap_result_callback(b2ShapeId shape_id, void* context) {
    overlap_test_context_t* ctx = static_cast<overlap_test_context_t*>(context);
    if (B2_ID_EQUALS(shape_id, ctx->target_shape)) {
      ctx->found_overlap = true;
      return false;
    }
    return true;
  }
  std::array<fan::physics::entity_t, 4> create_stroked_rectangle(
    const fan::vec2& center_position,
    const fan::vec2& half_size,
    f32_t thickness,
    uint8_t body_type,
    std::array<fan::physics::shape_properties_t, 4> shape_properties
  ){
    std::array<fan::physics::entity_t, 4> walls;

    std::array<fan::vec2, 4> positions {{
        fan::vec2(center_position.x, center_position.y - half_size.y),
        fan::vec2(center_position.x, center_position.y + half_size.y),
        fan::vec2(center_position.x - half_size.x, center_position.y),
        fan::vec2(center_position.x + half_size.x, center_position.y)
      }};

    std::array<fan::vec2, 4> sizes {{
        fan::vec2(half_size.x, thickness),
        fan::vec2(half_size.x, thickness),
        fan::vec2(thickness, half_size.y),
        fan::vec2(thickness, half_size.y)
      }};

    for (uint32_t i = 0; i < 4; i++) {
      walls[i] = fan::physics::gphysics()->create_rectangle(
        positions[i],
        sizes[i],
        0.0f,
        body_type,
        shape_properties[i]
      );
    }
    return walls;
  }

  void b2_to_fan_vertices(
    const b2Transform& xf,
    const b2Vec2* b2_vertices,
    int vertex_count,
    std::vector<fan::vec2>& out
  ) {
    out.resize(vertex_count);

    for (auto [i, v] : fan::enumerate(out)) {
      b2Vec2 w = b2TransformPoint(xf, b2_vertices[i]);
      v = fan::physics::physics_to_render(w);
    }
  }

  bool is_rectangle(const std::vector<fan::vec2>& v){
    if (v.size() != 4) {
      return false;
    }

    auto ab = v[1] - v[0];
    auto bc = v[2] - v[1];
    auto cd = v[3] - v[2];
    auto da = v[0] - v[3];

    constexpr f32_t eps = 1e-4f;

    bool right_angles =
      std::abs(ab.dot(bc)) < eps &&
      std::abs(bc.dot(cd)) < eps &&
      std::abs(cd.dot(da)) < eps &&
      std::abs(da.dot(ab)) < eps;

    if (!right_angles) {
      return false;
    }

    bool equal_sides = 
      std::abs(ab.length() - cd.length()) < eps &&
      std::abs(bc.length() - da.length()) < eps;

    return equal_sides;
  }
}
#endif