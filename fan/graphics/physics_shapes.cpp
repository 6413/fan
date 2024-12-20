#include "physics_shapes.hpp"

void fan::graphics::physics_shapes::shape_physics_update(const loco_t::physics_update_data_t& data) {
  rectangle_t& shape = *((rectangle_t*)data.shape);
  fan::vec2 p = b2Body_GetWorldPoint(shape.body_id, fan::vec2(0));
  b2Rot rotation = b2Body_GetRotation(shape.body_id);
  f32_t radians = b2Rot_GetAngle(rotation);
  shape.set_position(p);
  shape.set_angle(fan::vec3(0, 0, radians));
}