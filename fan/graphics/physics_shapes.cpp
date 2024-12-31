#include "physics_shapes.hpp"

void fan::graphics::physics_shapes::shape_physics_update(const loco_t::physics_update_data_t& data) {
  fan::vec2 p = b2Body_GetWorldPoint(data.body_id, fan::vec2(0));
  b2Rot rotation = b2Body_GetRotation(data.body_id);
  f32_t radians = b2Rot_GetAngle(rotation);
  loco_t::shape_t& shape = *(loco_t::shape_t*)&data.shape_id;
  shape.set_position(p);
  shape.set_angle(fan::vec3(0, 0, radians));
}

std::array<fan::graphics::physics_shapes::rectangle_t, 4> fan::graphics::physics_shapes::create_stroked_rectangle(
  const fan::vec2& center_position, 
  const fan::vec2& half_size,
  f32_t thickness,
  const fan::color& wall_color, 
  std::array<fan::physics::shape_properties_t, 4> shape_properties
) {
  std::array<fan::graphics::physics_shapes::rectangle_t, 4> walls;
  const fan::color wall_outline = wall_color * 2;
  // top
  walls[0] = fan::graphics::physics_shapes::rectangle_t{ {
      .position = fan::vec2(center_position.x, center_position.y - half_size.y),
      .size = fan::vec2(half_size.x * 2, thickness),
      .color = wall_color,
      .outline_color = wall_outline,
      .shape_properties = shape_properties[0]
    } };
  // bottom
  walls[1] = fan::graphics::physics_shapes::rectangle_t{ {
      .position = fan::vec2(center_position.x, center_position.y + half_size.y),
      .size = fan::vec2(half_size.x * 2, thickness),
      .color = wall_color,
      .outline_color = wall_color,
      .shape_properties = shape_properties[1]
    } };
  // left
  walls[2] = fan::graphics::physics_shapes::rectangle_t{ {
      .position = fan::vec2(center_position.x - half_size.x, center_position.y),
      .size = fan::vec2(thickness, half_size.y * 2),
      .color = wall_color,
      .outline_color = wall_outline,
      .shape_properties = shape_properties[2]
    } };
  // right
  walls[3] = fan::graphics::physics_shapes::rectangle_t{ {
      .position = fan::vec2(center_position.x + half_size.x, center_position.y),
      .size = fan::vec2(thickness, half_size.y * 2),
      .color = wall_color,
      .outline_color = wall_outline,
      .shape_properties = shape_properties[3]
    } };
  return walls;
}

void fan::graphics::character2d_t::process_movement(f32_t friction) {
  bool can_jump = false;

  b2Vec2 velocity = b2Body_GetLinearVelocity(character);
  if (jump_delay == 0.0f && jumping == false && velocity.y < 0.01f) {
    int capacity = b2Body_GetContactCapacity(character);
    capacity = b2MinInt(capacity, 4);
    b2ContactData contactData[4];
    int count = b2Body_GetContactData(character, contactData, capacity);
    for (int i = 0; i < count; ++i) {
      b2BodyId bodyIdA = b2Shape_GetBody(contactData[i].shapeIdA);
      float sign = 0.0f;
      if (B2_ID_EQUALS(bodyIdA, character.body_id)) {
        // normal points from A to B
        sign = -1.0f;
      }
      else {
        sign = 1.0f;
      }
      if (sign * contactData[i].manifold.normal.y < -0.9f) {
        can_jump = true;
        break;
      }
    }
  }

  if (gloco->input_action.is_action_down("move_left")) {
    b2Body_ApplyForceToCenter(character, { -force, 0 }, true);
  }
  if (gloco->input_action.is_action_down("move_right")) {
    b2Body_ApplyForceToCenter(character, { force, 0 }, true);
  }
  bool move_up = gloco->input_action.is_action_down("move_up");
  if (move_up) {
    if (can_jump) {
      b2Body_ApplyLinearImpulseToCenter(character, { 0, -impulse }, true);
      jump_delay = 0.5f;
      jumping = true;
    }
  }
  else {
    jumping = false;
  }
  jump_delay = 0;
}