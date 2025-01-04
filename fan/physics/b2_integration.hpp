#pragma once

#include <box2d/box2d.h>

#include <fan/types/vector.h>

namespace fan {
  namespace physics {
    struct shapes_e {
      enum {
        capsule,
        polygon,
        circle,
        box,
      };
    };

    inline static f32_t length_units_per_meter = 1024.f;

    struct capsule_t : b2Capsule {
      using b2Capsule::b2Capsule;
      capsule_t(const b2Capsule& capsule) : b2Capsule(capsule) {}
    };
    struct polygon_t : b2Polygon {
      using b2Polygon::b2Polygon;
      polygon_t(const b2Polygon& polygon) : b2Polygon(polygon) {}
    };
    struct circle_t : b2Circle {
      using b2Circle::b2Circle;
      circle_t(const b2Circle& circle) : b2Circle(circle) {}
    };
    /// A line segment with two-sided collision.
    struct segment_t : b2Segment {
      using b2Segment::b2Segment;
      segment_t(const b2Segment& segment) : b2Segment(segment) {}
    };
    // struct chain_segment_t : b2ChainSegment {
    //   using b2ChainSegment::b2ChainSegment;
    //   chain_segment_t(const b2ChainSegment& segment) : b2ChainSegment(segment) {}
    // };

    struct body_id_t : b2BodyId {
      using b2BodyId::b2BodyId;
      body_id_t() : b2BodyId(b2_nullBodyId){}
      body_id_t(const b2BodyId& body_id) : b2BodyId(body_id) {

      }
      void set_body(const body_id_t& b) {
        *this = b;
      }
      bool operator==(const body_id_t& b) const {
        b2BodyId a = *this;
        return B2_ID_EQUALS(a, b);
      }
      bool operator!=(const body_id_t& b) const {
        return !this->operator==(b);
      }
      bool is_valid() {
        return *this != b2_nullBodyId;
      }
      void destroy() {
        if (is_valid() == false) {
          return;
        }
        b2DestroyBody(*this);
        *this = b2_nullBodyId;
      }

      fan::vec2 get_linear_velocity() const {
        return fan::vec2(b2Body_GetLinearVelocity(*this)) * length_units_per_meter;
      }
      void set_linear_velocity(const fan::vec2& v) {
        b2Body_SetLinearVelocity(*this, v / length_units_per_meter);
      }
      void apply_force_center(const fan::vec2& v) {
        b2Body_ApplyForceToCenter(*this, v / length_units_per_meter, true);
      }
      void apply_linear_impulse_center(const fan::vec2& v) {
        b2Body_ApplyLinearImpulseToCenter(*this, v / length_units_per_meter, true);
      }
      void apply_angular_impulse(f32_t impulse) {
        b2Body_ApplyAngularImpulse(*this, impulse / length_units_per_meter, true);
      }
      fan::vec2 get_physics_position() const {
        return b2Body_GetPosition(*this);
      }
    };

    struct shape_properties_t {
      f32_t friction = 0.6f;
	    f32_t density = 1.0f;
      f32_t restitution = 0.0f;
      f32_t rolling_resistance = 0.f;
      bool fixed_rotation = false;
      bool enable_presolve_events = false;
      bool is_sensor = false;
      f32_t linear_damping = 0.0f;
      b2Filter filter = b2DefaultFilter();
    };

    struct entity_t : body_id_t {
      using body_id_t::body_id_t;

    };

    struct body_type_e {
      enum : uint8_t{
        static_body = b2_staticBody,
        kinematic_body = b2_kinematicBody,
        dynamic_body = b2_dynamicBody,
        count = b2_bodyTypeCount
      };
    };

    using body_type = b2BodyType;

    struct sensor_events_t {
      struct sensor_contact_t {
        fan::physics::body_id_t sensor_id;
        fan::physics::body_id_t object_id;
        bool is_in_contact = 0;
      };
      void update(b2WorldId world_id);

      void update_contact(b2BodyId sensor_id, b2BodyId object_id, bool is_in_contact);

      bool is_on_sensor(fan::physics::body_id_t test_id, fan::physics::body_id_t sensor_id) const;
      b2SensorEvents sensor_events;
      std::vector<sensor_contact_t> contacts;
    };

    struct context_t {

      struct properties_t {
        // clang
        properties_t() {};
        fan::vec2 gravity{0, 9.8f/length_units_per_meter};
      };
      context_t(const properties_t& properties = properties_t());
      
      entity_t create_box(const fan::vec2& position, const fan::vec2& size, uint8_t body_type, const shape_properties_t& shape_properties);
      entity_t create_circle(const fan::vec2& position, f32_t radius, uint8_t body_type, const shape_properties_t& shape_properties);
      fan::physics::entity_t create_capsule(const fan::vec2& position, const b2Capsule& info, uint8_t body_type, const shape_properties_t& shape_properties);

      void step(f32_t dt);

      bool is_on_sensor(fan::physics::body_id_t test_id, fan::physics::body_id_t sensor_id) const;

      b2WorldId world_id;
      sensor_events_t sensor_events;
    };

    // This callback must be thread-safe. It may be called multiple times simultaneously.
// Notice how this method is constant and doesn't change any data. It also
// does not try to access any values in the world that may be changing, such as contact data.
    bool presolve_oneway_collision(b2ShapeId shapeIdA, b2ShapeId shapeIdB, b2Manifold* manifold, fan::physics::body_id_t character_body);

    fan::physics::body_id_t deep_copy_body(b2WorldId worldId, fan::physics::body_id_t sourceBodyId);
  }
}