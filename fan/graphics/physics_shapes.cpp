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
  for (int i = 0; i < 2; ++i) {
    b2Vec2 velocity = b2Body_GetLinearVelocity(feet[i]);
    if (jump_delay == 0.0f && jumping == false && velocity.y < 0.01f) {
      int capacity = b2Body_GetContactCapacity(feet[i]);
      capacity = b2MinInt(capacity, 4);
      b2ContactData contactData[4];
      int count = b2Body_GetContactData(feet[i], contactData, capacity);
      for (int i = 0; i < count; ++i) {
        b2BodyId bodyIdA = b2Shape_GetBody(contactData[i].shapeIdA);
        float sign = 0.0f;
        if (B2_ID_EQUALS(bodyIdA, feet[i])) {
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
  }

  walk_force = 0;
  if (gloco->input_action.is_action_down("move_left")) {
    if (std::abs(velocity.x) <= max_speed) {
      b2Body_ApplyForceToCenter(character, { -force, 0 }, true);
      walk_force = -force;
    }
  }
  if (gloco->input_action.is_action_down("move_right")) {
    if (std::abs(velocity.x) <= max_speed) {
      b2Body_ApplyForceToCenter(character, { force, 0 }, true);
      walk_force = force;
    }
  }
  //if (gloco->input_action.is_action_down("move_down")) {
  //  b2Body_ApplyForceToCenter(character, { 0, impulse*10 }, true);
  //}

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

void fan::graphics::UpdateReferenceAngle(b2WorldId world, b2JointId& joint_id, float new_reference_angle) {

  b2BodyId bodyIdA = b2Joint_GetBodyA(joint_id);
  b2BodyId bodyIdB = b2Joint_GetBodyB(joint_id);

  b2Vec2 anchor = b2Joint_GetLocalAnchorA(joint_id);
  bool enableLimit = b2RevoluteJoint_IsLimitEnabled(joint_id);
  float lowerAngle = b2RevoluteJoint_GetLowerLimit(joint_id);
  float upperAngle = b2RevoluteJoint_GetUpperLimit(joint_id);
  bool enableMotor = b2RevoluteJoint_IsMotorEnabled(joint_id);
  float motorSpeed = b2RevoluteJoint_GetMotorSpeed(joint_id);
  float maxMotorTorque = b2RevoluteJoint_GetMaxMotorTorque(joint_id);
  float hertz = b2RevoluteJoint_GetSpringHertz(joint_id);
  float damping_ratio = b2RevoluteJoint_GetSpringDampingRatio(joint_id);

  b2DestroyJoint(joint_id);

  b2RevoluteJointDef jointDef;
  jointDef.bodyIdA = bodyIdA;
  jointDef.bodyIdB = bodyIdB;
  jointDef.localAnchorA = b2Vec2{ anchor.x, anchor.y };
  jointDef.localAnchorB = b2Vec2{ anchor.x, anchor.y };
  jointDef.referenceAngle = new_reference_angle;
  jointDef.enableLimit = enableLimit;
  jointDef.lowerAngle = lowerAngle;
  jointDef.upperAngle = upperAngle;
  jointDef.enableMotor = enableMotor;
  jointDef.motorSpeed = motorSpeed;
  jointDef.maxMotorTorque = maxMotorTorque;
  jointDef.enableSpring = hertz > 0.0f;
  jointDef.hertz = hertz;
  jointDef.dampingRatio = damping_ratio;

  joint_id = b2CreateRevoluteJoint(world, &jointDef);
}

void fan::graphics::CreateHuman(Human* human, b2WorldId worldId, b2Vec2 position, f32_t scale, f32_t frictionTorque, f32_t hertz, f32_t dampingRatio, int groupIndex, void* userData, bool colorize)
{
  assert(human->is_spawned == false);

  position.y = gloco->window.get_size().y - position.y;

  for (int i = 0; i < bone_e::bone_count; ++i)
  {
    human->bones[i].joint_id = b2_nullJointId;
    human->bones[i].friction_scale = 1.0f;
    human->bones[i].parent_index = -1;
  }

  human->scale = scale;

  b2BodyDef bodyDef = b2DefaultBodyDef();
  bodyDef.type = b2_dynamicBody;
  bodyDef.sleepThreshold = 0.1f;
  bodyDef.userData = userData;

  b2ShapeDef shapeDef = b2DefaultShapeDef();
  shapeDef.friction = 0.2f;
  shapeDef.filter.groupIndex = -groupIndex;
  shapeDef.filter.categoryBits = 2;
  shapeDef.filter.maskBits = (1 | 2);

  b2ShapeDef footShapeDef = shapeDef;
  footShapeDef.friction = 0.001f;
  footShapeDef.filter.categoryBits = 2;
  footShapeDef.filter.maskBits = 1;

  if (colorize)
  {
    footShapeDef.customColor = b2_colorSaddleBrown;
  }

  f32_t s = scale;
  f32_t maxTorque = frictionTorque * s;
  bool enableMotor = true;
  bool enableLimit = true;
  f32_t drawSize = 0.05f;

  b2HexColor shirtColor = b2_colorMediumTurquoise;
  b2HexColor pantColor = b2_colorDodgerBlue;
  b2HexColor skinColors[4] = { b2_colorNavajoWhite, b2_colorLightYellow, b2_colorPeru, b2_colorTan };
  b2HexColor skinColor = skinColors[groupIndex % 4];

  // hip
  {
    Bone* bone = human->bones + bone_e::hip;
    bone->parent_index = -1;
    bodyDef.position = b2Add(b2Vec2{ 0.0f, -0.95f * s }, position);
    bodyDef.linearDamping = 0.0f;

    bone->visual = fan::graphics::physics_shapes::capsule_t{ {
        .position = fan::vec3(fan::vec2(bodyDef.position), 100),
        .center0 = { 0.0f, 0.02f * s },
        .center1 = { 0.0f, -0.02f * s },
        .radius = 0.095f * s,
        .color = fan::color::hex(pantColor),
        .body_type = fan::physics::body_type_e::dynamic_body,
        .shape_properties{
        .fixed_rotation = true,
        .linear_damping = 0.0,
        .filter = shapeDef.filter
    }
      } };
  }

  // torso
  {
    Bone* bone = human->bones + bone_e::torso;
    bone->parent_index = bone_e::hip;

    bodyDef.position = b2Add(b2Vec2{ 0.0f, -1.2f * s }, position);
    bone->visual = fan::graphics::physics_shapes::capsule_t{ {
        .position = fan::vec3(fan::vec2(bodyDef.position), 50),
        .center0 = { 0.0f, 0.135f * s },
        .center1 = { 0.0f, -0.135f * s },
        .radius = 0.09f * s,
        .color = fan::color::hex(shirtColor),
        .body_type = fan::physics::body_type_e::dynamic_body,
        .shape_properties{
        .fixed_rotation = true,
        .linear_damping = 0.0,
        .filter = shapeDef.filter
    }
      } };
    bone->friction_scale = 0.5f;

    b2Vec2 pivot = b2Add(b2Vec2{ 0.0f, -1.0f * s }, position);
    b2RevoluteJointDef jointDef = b2DefaultRevoluteJointDef();
    jointDef.bodyIdA = human->bones[bone->parent_index].visual;
    jointDef.bodyIdB = bone->visual;
    jointDef.localAnchorA = b2Body_GetLocalPoint(jointDef.bodyIdA, pivot);
    jointDef.localAnchorB = b2Body_GetLocalPoint(jointDef.bodyIdB, pivot);
    jointDef.enableLimit = enableLimit;
    jointDef.lowerAngle = -0.25f * fan::math::pi;
    jointDef.upperAngle = 0.0f;
    jointDef.enableMotor = enableMotor;
    jointDef.maxMotorTorque = bone->friction_scale * maxTorque * scale;
    jointDef.enableSpring = hertz > 0.0f;
    jointDef.hertz = hertz;
    jointDef.dampingRatio = dampingRatio;
    jointDef.drawSize = drawSize;

    bone->joint_id = b2CreateRevoluteJoint(worldId, &jointDef);
  }

  // head
  {
    Bone* bone = human->bones + bone_e::head;
    bone->parent_index = bone_e::torso;

    bodyDef.position = b2Add(b2Vec2{ 0.0f, -1.475f * s }, position);
    bone->friction_scale = 0.25f;

    if (colorize)
    {
      shapeDef.customColor = skinColor;
    }

    bone->visual = fan::graphics::physics_shapes::capsule_t{ {
        .position = fan::vec3(fan::vec2(bodyDef.position), 100),
        .center0 = { 0.0f, 0.038f * s },
        .center1 = { 0.0f, -0.039f * s },
        .radius = 0.075f * s,
        .color = fan::color::hex(skinColor),
        .body_type = fan::physics::body_type_e::dynamic_body,
        .shape_properties{ .fixed_rotation = false, .linear_damping = 0.1f,
        .filter = shapeDef.filter },
      } };

    b2Vec2 pivot = b2Add(b2Vec2{ 0.0f, -1.4f * s }, position);
    b2RevoluteJointDef jointDef = b2DefaultRevoluteJointDef();
    jointDef.bodyIdA = human->bones[bone->parent_index].visual;
    jointDef.bodyIdB = bone->visual;
    jointDef.localAnchorA = b2Body_GetLocalPoint(jointDef.bodyIdA, pivot);
    jointDef.localAnchorB = b2Body_GetLocalPoint(jointDef.bodyIdB, pivot);
    jointDef.enableLimit = enableLimit;
    jointDef.lowerAngle = -0.3f * fan::math::pi;
    jointDef.upperAngle = 0.1f * fan::math::pi;
    jointDef.enableMotor = enableMotor;
    jointDef.maxMotorTorque = bone->friction_scale * maxTorque * scale;
    jointDef.enableSpring = hertz > 0.0f;
    jointDef.hertz = hertz;
    jointDef.dampingRatio = dampingRatio;
    jointDef.drawSize = drawSize;

    bone->joint_id = b2CreateRevoluteJoint(worldId, &jointDef);
  }

  // upper left leg
  {
    Bone* bone = human->bones + bone_e::upper_left_leg;
    bone->parent_index = bone_e::hip;

    bodyDef.position = b2Add(b2Vec2{ 0.0f, -0.775f * s }, position);
    bodyDef.linearDamping = 0.0f;

    bone->visual = fan::graphics::physics_shapes::capsule_t{ {
        .position = fan::vec3(fan::vec2(bodyDef.position), 100),
        .center0 = { 0.0f, 0.125f * s },
        .center1 = { 0.0f, -0.125f * s },
        .radius = 0.06f * s,
        .color = fan::color::hex(pantColor) * 1.5,
        .body_type = fan::physics::body_type_e::dynamic_body,
        .shape_properties{ .fixed_rotation = false, .linear_damping = 0.0f,
        .filter = shapeDef.filter },
      } };

    bone->friction_scale = 1.0f;

    b2Vec2 pivot = b2Add(b2Vec2{ 0.0f, -0.9f * s }, position);
    b2RevoluteJointDef jointDef = b2DefaultRevoluteJointDef();
    jointDef.bodyIdA = human->bones[bone->parent_index].visual;
    jointDef.bodyIdB = bone->visual;
    jointDef.localAnchorA = b2Body_GetLocalPoint(jointDef.bodyIdA, pivot);
    jointDef.localAnchorB = b2Body_GetLocalPoint(jointDef.bodyIdB, pivot);
    jointDef.enableLimit = enableLimit;
    jointDef.lowerAngle = -fan::math::pi / 10;
    jointDef.upperAngle = fan::math::pi / 10;
    jointDef.enableMotor = enableMotor;
    jointDef.maxMotorTorque = bone->friction_scale * maxTorque * scale * 100000000.f;
    jointDef.enableSpring = hertz > 0.0f;
    jointDef.hertz = hertz;
    jointDef.dampingRatio = dampingRatio;
    jointDef.drawSize = drawSize;

    bone->joint_id = b2CreateRevoluteJoint(worldId, &jointDef);
  }

  b2Vec2 points[4] = {
    { -0.03f * s, 0.185f * s },
    { 0.11f * s, 0.185f * s },
    { 0.11f * s, 0.16f * s },
    { -0.03f * s, 0.14f * s },
  };

  b2Hull footHull = b2ComputeHull(points, 4);
  b2Polygon footPolygon = b2MakePolygon(&footHull, 0.015f * s);

  // lower left leg
  {
    Bone* bone = human->bones + bone_e::lower_left_leg;
    bone->parent_index = bone_e::upper_left_leg;

    bodyDef.position = b2Add(b2Vec2{ 0.0f, -0.475f * s }, position);
    bodyDef.linearDamping = 0.0f;
    bone->friction_scale = 0.5f;

    bone->visual = fan::graphics::physics_shapes::capsule_t{ {
        .position = fan::vec3(fan::vec2(bodyDef.position), 100),
        .center0 = { 0.0f, 0.155f * s },
        .center1 = { 0.0f, -0.125f * s },
        .radius = 0.045f * s,
        .color = fan::color::hex(pantColor) * 1.5,
        .body_type = fan::physics::body_type_e::dynamic_body,
        .shape_properties{
        .fixed_rotation = true,
        .linear_damping = 0.0f,
        .filter = shapeDef.filter
    },
      } };
    b2CreatePolygonShape(bone->visual, &footShapeDef, &footPolygon);

    b2Vec2 pivot = b2Add(b2Vec2{ 0.0f, -0.625f * s }, position);
    b2RevoluteJointDef jointDef = b2DefaultRevoluteJointDef();
    jointDef.bodyIdA = human->bones[bone->parent_index].visual;
    jointDef.bodyIdB = bone->visual;
    jointDef.localAnchorA = b2Body_GetLocalPoint(jointDef.bodyIdA, pivot);
    jointDef.localAnchorB = b2Body_GetLocalPoint(jointDef.bodyIdB, pivot);
    jointDef.enableLimit = enableLimit;
    //jointDef.lowerAngle = -fan::math::pi / 10;
    //jointDef.upperAngle = -fan::math::pi / 10;
    jointDef.enableMotor = enableMotor;
    jointDef.maxMotorTorque = bone->friction_scale * maxTorque * scale;
    jointDef.enableSpring = hertz > 0.0f;
    jointDef.hertz = hertz;
    jointDef.dampingRatio = dampingRatio;
    jointDef.drawSize = drawSize;

    bone->joint_id = b2CreateRevoluteJoint(worldId, &jointDef);
  }

  //// upper right leg
  {
    Bone* bone = human->bones + bone_e::upper_right_leg;
    bone->parent_index = bone_e::hip;

    bodyDef.position = b2Add(b2Vec2{ 0.0f, -0.775f * s }, position);
    bone->visual = fan::graphics::physics_shapes::capsule_t{ {
        .position = fan::vec3(fan::vec2(bodyDef.position), 100),
        .center0 = { 0.0f, 0.125f * s },
        .center1 = { 0.0f, -0.125f * s },
        .radius = 0.06f * s,
        .color = fan::color::hex(pantColor),
        .body_type = fan::physics::body_type_e::dynamic_body,
        .shape_properties{ .fixed_rotation = false, .linear_damping = 0.0f,
        .filter = shapeDef.filter },
      } };
    bone->friction_scale = 1.0f;

    b2Vec2 pivot = b2Add(b2Vec2{ 0.0f, -0.9f * s }, position);
    b2RevoluteJointDef jointDef = b2DefaultRevoluteJointDef();
    jointDef.bodyIdA = human->bones[bone->parent_index].visual;
    jointDef.bodyIdB = bone->visual;
    jointDef.localAnchorA = b2Body_GetLocalPoint(jointDef.bodyIdA, pivot);
    jointDef.localAnchorB = b2Body_GetLocalPoint(jointDef.bodyIdB, pivot);
    jointDef.enableLimit = enableLimit;
    jointDef.lowerAngle = -fan::math::pi / 10;
    jointDef.upperAngle = fan::math::pi / 10;
    jointDef.enableMotor = enableMotor;
    jointDef.maxMotorTorque = bone->friction_scale * maxTorque * scale * 100000000.f;
    jointDef.enableSpring = hertz > 0.0f;
    jointDef.hertz = hertz;
    jointDef.dampingRatio = dampingRatio;
    jointDef.drawSize = drawSize;

    bone->joint_id = b2CreateRevoluteJoint(worldId, &jointDef);
  }

  //// lower right leg
  {
    Bone* bone = human->bones + bone_e::lower_right_leg;
    bone->parent_index = bone_e::upper_right_leg;

    bodyDef.position = b2Add(b2Vec2{ 0.0f, -0.475f * s }, position);
    bone->visual = fan::graphics::physics_shapes::capsule_t{ {
        .position = fan::vec3(fan::vec2(bodyDef.position), 100),
        .center0 = { 0.0f, 0.155f * s },
        .center1 = { 0.0f, -0.125f * s },
        .radius = 0.045f * s,
        .color = fan::color::hex(pantColor),
        .body_type = fan::physics::body_type_e::dynamic_body,
        .shape_properties{
        .fixed_rotation = true,
        .linear_damping = 0.0f,
        .filter = shapeDef.filter
    },
      } };
    bone->friction_scale = 0.5f;

    // b2Polygon box = b2MakeOffsetBox(0.1f * s, 0.03f * s, {0.05f * s, -0.175f * s}, 0.0f);
    // b2CreatePolygonShape(bone->visual, &shapeDef, &box);

    // capsule = { { -0.02f * s, -0.175f * s }, { 0.13f * s, -0.175f * s }, 0.03f * s };
    // b2CreateCapsuleShape( bone->visual, &footShapeDef, &capsule );

    b2CreatePolygonShape(bone->visual, &footShapeDef, &footPolygon);

    b2Vec2 pivot = b2Add(b2Vec2{ 0.0f, -0.625f * s }, position);
    b2RevoluteJointDef jointDef = b2DefaultRevoluteJointDef();
    jointDef.bodyIdA = human->bones[bone->parent_index].visual;
    jointDef.bodyIdB = bone->visual;
    jointDef.localAnchorA = b2Body_GetLocalPoint(jointDef.bodyIdA, pivot);
    jointDef.localAnchorB = b2Body_GetLocalPoint(jointDef.bodyIdB, pivot);
    jointDef.enableLimit = enableLimit;
    //  jointDef.lowerAngle = -fan::math::pi / 10;
    //jointDef.upperAngle = -fan::math::pi / 10;
    jointDef.enableMotor = enableMotor;
    jointDef.maxMotorTorque = bone->friction_scale * maxTorque * scale;
    jointDef.enableSpring = hertz > 0.0f;
    jointDef.hertz = hertz;
    jointDef.dampingRatio = dampingRatio;
    jointDef.drawSize = drawSize;

    bone->joint_id = b2CreateRevoluteJoint(worldId, &jointDef);
  }

  // upper left arm
  {
    Bone* bone = human->bones + bone_e::upper_left_arm;
    bone->parent_index = bone_e::torso;
    bone->friction_scale = 0.5f;

    bodyDef.position = b2Add(b2Vec2{ 0.0f, -1.225f * s }, position);
    bodyDef.linearDamping = 0.0f;
    bone->visual = fan::graphics::physics_shapes::capsule_t{ {
        .position = fan::vec3(fan::vec2(bodyDef.position), 100),
        .center0 = { 0.0f, 0.125f * s },
        .center1 = { 0.0f, -0.125f * s },
        .radius = 0.035f * s,
        .color = fan::color::hex(shirtColor),
        .body_type = fan::physics::body_type_e::dynamic_body,
        .shape_properties{ .fixed_rotation = false, .linear_damping = 0.0f,
        .filter = shapeDef.filter },
      } };

    b2Vec2 pivot = b2Add(b2Vec2{ 0.0f, -1.35f * s }, position);
    b2RevoluteJointDef jointDef = b2DefaultRevoluteJointDef();
    jointDef.bodyIdA = human->bones[bone->parent_index].visual;
    jointDef.bodyIdB = bone->visual;
    jointDef.localAnchorA = b2Body_GetLocalPoint(jointDef.bodyIdA, pivot);
    jointDef.localAnchorB = b2Body_GetLocalPoint(jointDef.bodyIdB, pivot);
    jointDef.enableLimit = enableLimit;
    jointDef.lowerAngle = -0.1f * fan::math::pi;
    jointDef.upperAngle = 0.8f * fan::math::pi;
    jointDef.enableMotor = enableMotor;
    jointDef.maxMotorTorque = bone->friction_scale * maxTorque * scale;
    jointDef.enableSpring = hertz > 0.0f;
    jointDef.hertz = hertz;
    jointDef.dampingRatio = dampingRatio;
    jointDef.drawSize = drawSize;

    bone->joint_id = b2CreateRevoluteJoint(worldId, &jointDef);
  }

  // lower left arm
  {
    Bone* bone = human->bones + bone_e::lower_left_arm;
    bone->parent_index = bone_e::upper_left_arm;

    bodyDef.position = b2Add(b2Vec2{ 0.0f, -0.975f * s }, position);
    bodyDef.linearDamping = 0.1f;
    bone->visual = fan::graphics::physics_shapes::capsule_t{ {
        .position = fan::vec3(fan::vec2(bodyDef.position), 100),
        .center0 = { 0.0f, 0.125f * s },
        .center1 = { 0.0f, -0.125f * s },
        .radius = 0.03f * s,
        .color = fan::color::hex(skinColor),
        .body_type = fan::physics::body_type_e::dynamic_body,
        .shape_properties{ .fixed_rotation = false, .linear_damping = 0.1f,
        .filter = shapeDef.filter },
      } };
    bone->friction_scale = 0.1f;

    b2Vec2 pivot = b2Add(b2Vec2{ 0.0f, -1.1f * s }, position);
    b2RevoluteJointDef jointDef = b2DefaultRevoluteJointDef();
    jointDef.bodyIdA = human->bones[bone->parent_index].visual;
    jointDef.bodyIdB = bone->visual;
    jointDef.localAnchorA = b2Body_GetLocalPoint(jointDef.bodyIdA, pivot);
    jointDef.localAnchorB = b2Body_GetLocalPoint(jointDef.bodyIdB, pivot);
    jointDef.referenceAngle = -0.25f * fan::math::pi;
    jointDef.enableLimit = enableLimit;
    jointDef.lowerAngle = -0.2f * fan::math::pi;
    jointDef.upperAngle = 0.3f * fan::math::pi;
    jointDef.enableMotor = enableMotor;
    jointDef.maxMotorTorque = bone->friction_scale * maxTorque * scale;
    jointDef.enableSpring = hertz > 0.0f;
    jointDef.hertz = hertz;
    jointDef.dampingRatio = dampingRatio;
    jointDef.drawSize = drawSize;

    bone->joint_id = b2CreateRevoluteJoint(worldId, &jointDef);
  }

  // upper right arm
  {
    Bone* bone = human->bones + bone_e::upper_right_arm;
    bone->parent_index = bone_e::torso;

    bodyDef.position = b2Add(b2Vec2{ 0.0f, -1.225f * s }, position);
    bodyDef.linearDamping = 0.0f;
    bone->visual = fan::graphics::physics_shapes::capsule_t{ {
        .position = fan::vec3(fan::vec2(bodyDef.position), 100),
        .center0 = { 0.0f, 0.125f * s },
        .center1 = { 0.0f, -0.125f * s },
        .radius = 0.035f * s,
        .color = fan::color::hex(skinColor),
        .body_type = fan::physics::body_type_e::dynamic_body,
        .shape_properties{ .fixed_rotation = false, .linear_damping = 0.0f,
        .filter = shapeDef.filter },
      } };
    bone->friction_scale = 0.5f;

    b2Vec2 pivot = b2Add(b2Vec2{ 0.0f, -1.35f * s }, position);
    b2RevoluteJointDef jointDef = b2DefaultRevoluteJointDef();
    jointDef.bodyIdA = human->bones[bone->parent_index].visual;
    jointDef.bodyIdB = bone->visual;
    jointDef.localAnchorA = b2Body_GetLocalPoint(jointDef.bodyIdA, pivot);
    jointDef.localAnchorB = b2Body_GetLocalPoint(jointDef.bodyIdB, pivot);
    jointDef.enableLimit = enableLimit;
    jointDef.lowerAngle = -0.1f * fan::math::pi;
    jointDef.upperAngle = 0.8f * fan::math::pi;
    jointDef.enableMotor = enableMotor;
    jointDef.maxMotorTorque = bone->friction_scale * maxTorque * scale;
    jointDef.enableSpring = hertz > 0.0f;
    jointDef.hertz = hertz;
    jointDef.dampingRatio = dampingRatio;
    jointDef.drawSize = drawSize;

    bone->joint_id = b2CreateRevoluteJoint(worldId, &jointDef);
  }

  // lower right arm
  {
    Bone* bone = human->bones + bone_e::lower_right_arm;
    bone->parent_index = bone_e::upper_right_arm;

    bodyDef.position = b2Add(b2Vec2{ 0.0f, -0.975f * s }, position);
    bodyDef.linearDamping = 0.1f;
    bone->visual = fan::graphics::physics_shapes::capsule_t{ {
        .position = fan::vec3(fan::vec2(bodyDef.position), 100),
        .center0 = { 0.0f, 0.125f * s },
        .center1 = { 0.0f, -0.125f * s },
        .radius = 0.03f * s,
        .color = fan::color::hex(skinColor),
        .body_type = fan::physics::body_type_e::dynamic_body,
        .shape_properties{ .fixed_rotation = false, .linear_damping = 0.1f,
        .filter = shapeDef.filter },
      } };
    bone->friction_scale = 0.1f;

    b2Vec2 pivot = b2Add(b2Vec2{ 0.0f, -1.1f * s }, position);
    b2RevoluteJointDef jointDef = b2DefaultRevoluteJointDef();
    jointDef.bodyIdA = human->bones[bone->parent_index].visual;
    jointDef.bodyIdB = bone->visual;
    jointDef.localAnchorA = b2Body_GetLocalPoint(jointDef.bodyIdA, pivot);
    jointDef.localAnchorB = b2Body_GetLocalPoint(jointDef.bodyIdB, pivot);
    jointDef.referenceAngle = -0.25f * fan::math::pi;
    jointDef.enableLimit = enableLimit;
    jointDef.lowerAngle = -0.2f * fan::math::pi;
    jointDef.upperAngle = 0.3f * fan::math::pi;
    jointDef.enableMotor = enableMotor;
    jointDef.maxMotorTorque = bone->friction_scale * maxTorque * scale;
    jointDef.enableSpring = hertz > 0.0f;
    jointDef.hertz = hertz;
    jointDef.dampingRatio = dampingRatio;
    jointDef.drawSize = drawSize;

    bone->joint_id = b2CreateRevoluteJoint(worldId, &jointDef);
  }

  human->is_spawned = true;
}

void fan::graphics::DestroyHuman(Human* human)
{
  assert(human->is_spawned == true);

  for (int i = 0; i < bone_e::bone_count; ++i)
  {
    if (B2_IS_NULL(human->bones[i].joint_id))
    {
      continue;
    }

    b2DestroyJoint(human->bones[i].joint_id);
    human->bones[i].joint_id = b2_nullJointId;
  }

  for (int i = 0; i < bone_e::bone_count; ++i)
  {
    if (B2_IS_NULL(human->bones[i].visual.body_id))
    {
      continue;
    }

    human->bones[i].visual.body_id.destroy();
    human->bones[i].visual.erase();
  }

  human->is_spawned = false;
}

void fan::graphics::Human_SetVelocity(Human* human, b2Vec2 velocity)
{
  for (int i = 0; i < bone_e::bone_count; ++i)
  {
    b2BodyId bodyId = human->bones[i].visual;

    if (B2_IS_NULL(bodyId))
    {
      continue;
    }

    b2Body_SetLinearVelocity(bodyId, velocity);
  }
}

void fan::graphics::Human_ApplyRandomAngularImpulse(Human* human, f32_t magnitude)
{
  assert(human->is_spawned == true);
  f32_t impulse = fan::random::f32(-magnitude, magnitude);
  b2Body_ApplyAngularImpulse(human->bones[bone_e::torso].visual, impulse, true);
}

void fan::graphics::Human_SetJointFrictionTorque(Human* human, f32_t torque)
{
  assert(human->is_spawned == true);
  if (torque == 0.0f)
  {
    for (int i = 1; i < bone_e::bone_count; ++i)
    {
      b2RevoluteJoint_EnableMotor(human->bones[i].joint_id, false);
    }
  }
  else
  {
    for (int i = 1; i < bone_e::bone_count; ++i)
    {
      b2RevoluteJoint_EnableMotor(human->bones[i].joint_id, true);
      f32_t scale = human->scale * human->bones[i].friction_scale;
      b2RevoluteJoint_SetMaxMotorTorque(human->bones[i].joint_id, scale * torque);
    }
  }
}

void fan::graphics::Human_SetJointSpringHertz(Human* human, f32_t hertz)
{
  assert(human->is_spawned == true);
  if (hertz == 0.0f)
  {
    for (int i = 1; i < bone_e::bone_count; ++i)
    {
      b2RevoluteJoint_EnableSpring(human->bones[i].joint_id, false);
    }
  }
  else
  {
    for (int i = 1; i < bone_e::bone_count; ++i)
    {
      b2RevoluteJoint_EnableSpring(human->bones[i].joint_id, true);
      b2RevoluteJoint_SetSpringHertz(human->bones[i].joint_id, hertz);
    }
  }
}

void fan::graphics::Human_SetJointDampingRatio(Human* human, f32_t dampingRatio)
{
  assert(human->is_spawned == true);
  for (int i = 1; i < bone_e::bone_count; ++i)
  {
    b2RevoluteJoint_SetSpringDampingRatio(human->bones[i].joint_id, dampingRatio);
  }
}

void fan::graphics::human_animate_walk(Human* human, f32_t force, f32_t dt) {
  f32_t massScale = 250.0f;

  b2BodyId torsoId = human->bones[bone_e::torso].visual;
  b2Vec2 force_ = { force * massScale, 0 };

  b2BodyId leftLegId = human->bones[bone_e::upper_left_leg].visual;
  b2BodyId rightLegId = human->bones[bone_e::upper_right_leg].visual;

  Bone& blower_left_arm = human->bones[bone_e::lower_left_arm];
  Bone& blower_right_arm = human->bones[bone_e::lower_right_arm];
  Bone& bupper_left_leg = human->bones[bone_e::upper_left_leg];
  Bone& bupper_right_leg = human->bones[bone_e::upper_right_leg];
  Bone& blower_left_leg = human->bones[bone_e::lower_left_leg];
  Bone& blower_right_leg = human->bones[bone_e::lower_right_leg];

  static f32_t counter = 0;
  static f32_t prev = force;

  f32_t torso_vel_x = b2Body_GetLinearVelocity(torsoId).x;
  f32_t torso_vel_y = b2Body_GetLinearVelocity(torsoId).y;
  int vel_sgn = fan::math::sgn(torso_vel_x);
  int force_sgn = fan::math::sgn(force);
  f32_t swing_speed = torso_vel_x ? (vel_sgn * 0.f + torso_vel_x / 15.f) : 0;

  f32_t ttransform = b2Rot_GetAngle(b2Body_GetRotation(human->bones[bone_e::torso].visual));
  f32_t ltransform = b2Rot_GetAngle(b2Body_GetRotation(leftLegId));
  f32_t rtransform = b2Rot_GetAngle(b2Body_GetRotation(rightLegId));

  static int last_valid_direction = 0;

  if (torso_vel_x || counter == 0) {
    counter += swing_speed * dt;

    f32_t quarter_pi = 0.25f * fan::math::pi;
    quarter_pi *= 3; // why this is required?

    if (std::abs(torso_vel_x) / 100.f > 1.f && torso_vel_x) {
      last_valid_direction = vel_sgn;
      UpdateReferenceAngle(gloco->physics_context.world_id, blower_left_arm.joint_id, last_valid_direction == 1 ? quarter_pi : -quarter_pi);
      UpdateReferenceAngle(gloco->physics_context.world_id, blower_right_arm.joint_id, last_valid_direction == 1 ? quarter_pi : -quarter_pi);
      
    }
    //}

    f32_t ang = fan::math::pi / 4.f;
    b2RevoluteJoint_SetLimits(blower_left_leg.joint_id, vel_sgn * -ang, vel_sgn * ang);
    b2RevoluteJoint_SetLimits(blower_right_leg.joint_id, vel_sgn * -ang, vel_sgn * ang);

    b2RevoluteJoint_SetLimits(bupper_left_leg.joint_id, vel_sgn * -ang, vel_sgn * ang);
    b2RevoluteJoint_SetLimits(bupper_right_leg.joint_id, vel_sgn * -ang, vel_sgn * ang);


    if (force || std::abs(torso_vel_x / 100.f) > 1.f) {
      if (sin(counter) < 0) {
        b2RevoluteJoint_SetMotorSpeed(bupper_left_leg.joint_id, (ttransform - ltransform) * std::abs(torso_vel_x / 25.f));
        b2RevoluteJoint_SetMotorSpeed(bupper_right_leg.joint_id, vel_sgn * -std::abs(sin(counter)) * 10);
      }
      else {
        b2RevoluteJoint_SetMotorSpeed(bupper_left_leg.joint_id, vel_sgn * -std::abs(sin(counter)) * 10);
        b2RevoluteJoint_SetMotorSpeed(bupper_right_leg.joint_id, (ttransform - rtransform) * std::abs(torso_vel_x / 25.f));
      }
    }
    else {
      b2RevoluteJoint_SetMotorSpeed(bupper_left_leg.joint_id, (ttransform - ltransform) * 5);
      b2RevoluteJoint_SetMotorSpeed(bupper_right_leg.joint_id, (ttransform - rtransform) * 5);
    }
  }
}
