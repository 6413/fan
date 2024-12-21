#pragma once

#include <box2d/box2d.h>

#include <fan/types/vector.h>

namespace fan {
  namespace physics {
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
    struct chain_segment_t : b2ChainSegment {
      using b2ChainSegment::b2ChainSegment;
      chain_segment_t(const b2ChainSegment& segment) : b2ChainSegment(segment) {}
    };

    struct body_id_t : b2BodyId {
      using b2BodyId::b2BodyId;
      operator b2BodyId() const {
        return *this;
      }
      body_id_t(const b2BodyId& body_id) : b2BodyId(body_id) {}
    };

    struct entity_t {
      body_id_t body_id;
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

    struct context_t {

      struct properties_t {
        // clang
        properties_t() = default;
        f32_t length_units_per_meter{128.f};
        fan::vec2 gravity{0, 9.8f * length_units_per_meter};
      };
      context_t(const properties_t& properties = properties_t());
      
      entity_t create_box(const fan::vec2& position, const fan::vec2& size, uint8_t body_type);
      entity_t create_circle(const fan::vec2& position, f32_t radius, uint8_t body_type);

      void step(f32_t dt);

      b2WorldId world_id;
    };
  }
}