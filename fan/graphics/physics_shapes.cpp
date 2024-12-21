#include "physics_shapes.hpp"

void fan::graphics::physics_shapes::shape_physics_update(const loco_t::physics_update_data_t& data) {
  fan::vec2 p = b2Body_GetWorldPoint(data.body_id, fan::vec2(0));
  b2Rot rotation = b2Body_GetRotation(data.body_id);
  f32_t radians = b2Rot_GetAngle(rotation);
  loco_t::shape_t& shape = *(loco_t::shape_t*)&data.shape_id;
  shape.set_position(p);
  shape.set_angle(fan::vec3(0, 0, radians));
}