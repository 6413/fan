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

fan::graphics::character2d_t::character2d_t() {
  add_inputs();
}

void fan::graphics::character2d_t::add_inputs() {
  gloco->input_action.add(fan::key_a, "move_left");
  gloco->input_action.add(fan::key_d, "move_right");
  gloco->input_action.add(fan::key_space, "move_up");
  gloco->input_action.add(fan::key_s, "move_down");
}

bool fan::graphics::character2d_t::is_on_ground(fan::physics::body_id_t main, std::array<fan::physics::body_id_t, 2> feet, bool jumping) {
  for (int i = 0; i < 2; ++i) {
    fan::physics::body_id_t body_id = feet[i];
    if (body_id.is_valid() == false) {
      body_id = main;
    }
    b2Vec2 velocity = b2Body_GetLinearVelocity(body_id);
    if (jumping == false && velocity.y < 0.01f) {
      int capacity = b2Body_GetContactCapacity(body_id);
      capacity = b2MinInt(capacity, 4);
      b2ContactData contactData[4];
      int count = b2Body_GetContactData(body_id, contactData, capacity);
      for (int i = 0; i < count; ++i) {
        b2BodyId bodyIdA = b2Shape_GetBody(contactData[i].shapeIdA);
        float sign = 0.0f;
        if (B2_ID_EQUALS(bodyIdA, body_id)) {
          // normal points from A to B
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

void fan::graphics::character2d_t::process_movement(f32_t friction) {
  bool can_jump = false;

  b2Vec2 velocity = b2Body_GetLinearVelocity(*this);

  can_jump = is_on_ground(*this, std::to_array(feet), jumping);

  walk_force = 0;
  if (gloco->input_action.is_action_down("move_left")) {
    if (velocity.x > -max_speed) {
      b2Body_ApplyForceToCenter(*this, { -force, 0 }, true);
      walk_force = -force;
    }
  }
  if (gloco->input_action.is_action_down("move_right")) {
    if (velocity.x <= max_speed) {
      b2Body_ApplyForceToCenter(*this, { force, 0 }, true);
      walk_force = force;
    }
  }
  //if (gloco->input_action.is_action_down("move_down")) {
  //  b2Body_ApplyForceToCenter(character, { 0, impulse*10 }, true);
  //}

  bool move_up = gloco->input_action.is_action_clicked("move_up");
  if (move_up) {
    if (can_jump) {
      //b2Body_ApplyLinearImpulseToCenter(*this, { 0, -impulse }, true);
      jump_delay = 0.f;
      jumping = true;
    }
    else {
      jumping = false;
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

    b2Vec2 localAnchorA = b2Joint_GetLocalAnchorA(joint_id);
    b2Vec2 localAnchorB = b2Joint_GetLocalAnchorB(joint_id);
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
    jointDef.localAnchorA = localAnchorA;
    jointDef.localAnchorB = localAnchorB;
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

void fan::graphics::CreateHuman(Human* human, b2WorldId worldId, b2Vec2 position, f32_t scale, f32_t frictionTorque, f32_t hertz, f32_t dampingRatio, int groupIndex, void* userData, const std::array<loco_t::image_t, bone_e::bone_count>& images, const fan::color& color) {
  auto verified_images = images;
  for (auto& image : verified_images) {
    if (image.iic()) {
      image = gloco->default_texture;
    }
  }

  assert(human->is_spawned == false);


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
  shapeDef.friction = 0.5f;
  shapeDef.filter.groupIndex = -groupIndex;
  shapeDef.filter.categoryBits = 2;
  shapeDef.filter.maskBits = (1 | 2);

  b2ShapeDef footShapeDef = shapeDef;
  footShapeDef.friction = 0.5f;
  footShapeDef.filter.categoryBits = 2;
  footShapeDef.filter.maskBits = 1;

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

    bone->visual = fan::graphics::physics_shapes::capsule_sprite_t{ {
        .position = fan::vec3(fan::vec2(bodyDef.position), 50),
        .center0 = { 0.0f, 0.02f * s },
        .center1 = { 0.0f, -0.02f * s },
        .radius = 0.064f * s,
        .color = color,
        .image = images[bone_e::hip],
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
    bone->visual = fan::graphics::physics_shapes::capsule_sprite_t{ {
        .position = fan::vec3(fan::vec2(bodyDef.position), 50),
        .center0 = { 0.0f, 0.135f * s },
        .center1 = { 0.0f, -0.135f * s },
        .radius = 0.085f * s,
        .color = color,
        .image = images[bone_e::torso],
        .body_type = fan::physics::body_type_e::dynamic_body,
        .shape_properties{
        .fixed_rotation = true,
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

    bodyDef.position = b2Add(b2Vec2{ 0.0f, -1.555f * s }, position);
    bone->friction_scale = 0.25f;

    bone->visual = fan::graphics::physics_shapes::capsule_sprite_t{ {
        .position = fan::vec3(fan::vec2(bodyDef.position), 100),
        .center0 = { 0.0f, 0.038f * s },
        .center1 = { 0.0f, -0.039f * s },
        .radius = 0.124f * s,
        .color = color,
        .image = images[bone_e::head],
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

    bone->visual = fan::graphics::physics_shapes::capsule_sprite_t{ {
        .position = fan::vec3(fan::vec2(bodyDef.position), 10),
        .center0 = { 0.0f, 0.125f * s },
        .center1 = { 0.0f, -0.125f * s},
        .radius = 0.064f * s,
        .color = color,
        .image = images[bone_e::upper_left_leg],
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
		jointDef.lowerAngle = -0.5f * fan::math::pi;
		jointDef.upperAngle = 0.5f * fan::math::pi;
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

    bone->visual = fan::graphics::physics_shapes::capsule_sprite_t{ {
        .position = fan::vec3(fan::vec2(bodyDef.position), 15),
        .center0 = { 0.0f, 0.125f * s },
        .center1 = { 0.0f, -0.125f * s},
        .radius = 0.094f * s,
        .color = color,
        .image = images[bone_e::lower_left_leg],
        .body_type = fan::physics::body_type_e::dynamic_body,
        .shape_properties{
        .friction=shapeDef.friction,
        .fixed_rotation = false,
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
		jointDef.lowerAngle = -0.5f * fan::math::pi;
		jointDef.upperAngle = 0.5f * fan::math::pi;
    jointDef.enableMotor = enableMotor;
    jointDef.maxMotorTorque = bone->friction_scale * maxTorque * scale * 100000000.f;
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
    bone->visual = fan::graphics::physics_shapes::capsule_sprite_t{ {
        .position = fan::vec3(fan::vec2(bodyDef.position), 100),
        .center0 = { 0.0f, 0.125f * s  },
        .center1 = { 0.0f, -0.125f * s },
        .radius = 0.064f * s,
        .color = color,
        .image = images[bone_e::upper_right_leg],
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
		jointDef.lowerAngle = -0.5f * fan::math::pi;
		jointDef.upperAngle = 0.5f * fan::math::pi;
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
    bone->friction_scale = 0.5f;

    bodyDef.position = b2Add(b2Vec2{ 0.0f, -0.475f * s }, position);
    bone->visual = fan::graphics::physics_shapes::capsule_sprite_t{ {
        .position = fan::vec3(fan::vec2(bodyDef.position), 150),
        .center0 = { 0.0f, 0.125f * s },
        .center1 = { 0.0f, -0.125f * s},
        .radius = 0.094f * s,
        .color = color,
        .image = images[bone_e::lower_right_leg],
        .body_type = fan::physics::body_type_e::dynamic_body,
        .shape_properties{
        .friction=shapeDef.friction,
        .fixed_rotation = false,
        .linear_damping = 0.0f,
        .filter = shapeDef.filter
    },
      } };

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
		jointDef.lowerAngle = -0.5f * fan::math::pi;
		jointDef.upperAngle = 0.5f * fan::math::pi;
    jointDef.enableMotor = enableMotor;
    jointDef.maxMotorTorque = bone->friction_scale * maxTorque * scale * 100000000.f;
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

    bodyDef.position = b2Add(b2Vec2{ 0.0f, -1.145f * s }, position);
    bodyDef.linearDamping = 0.0f;
    bone->visual = fan::graphics::physics_shapes::capsule_sprite_t{ {
        .position = fan::vec3(fan::vec2(bodyDef.position), 10),
        .center0 = { 0.0f, 0.125f * s },
        .center1 = { 0.0f, -0.125f * s},
        .radius = 0.064f * s,
        .color = color,
        .image = images[bone_e::upper_left_arm],
        .body_type = fan::physics::body_type_e::dynamic_body,
        .shape_properties{ .fixed_rotation = false, .linear_damping = 0.5f,
        .filter = shapeDef.filter },
      } };

    b2Vec2 pivot = b2Add(b2Vec2{ 0.0f, -1.35f * s }, position);
    b2RevoluteJointDef jointDef = b2DefaultRevoluteJointDef();
    jointDef.bodyIdA = human->bones[bone->parent_index].visual;
    jointDef.bodyIdB = bone->visual;
    jointDef.localAnchorA = b2Body_GetLocalPoint(jointDef.bodyIdA, pivot);
    jointDef.localAnchorB = b2Body_GetLocalPoint(jointDef.bodyIdB, pivot);
    jointDef.enableLimit = enableLimit;
    jointDef.lowerAngle = -0.5f * fan::math::pi;
    jointDef.upperAngle = 0.5f * fan::math::pi;
    jointDef.enableMotor = enableMotor;
    jointDef.maxMotorTorque = bone->friction_scale * maxTorque * scale;
    jointDef.enableSpring = hertz > 0.0f;
    jointDef.hertz = hertz*5;
    jointDef.dampingRatio = dampingRatio;
    jointDef.drawSize = drawSize;

    bone->joint_id = b2CreateRevoluteJoint(worldId, &jointDef);
  }

  // lower left arm
  {
    Bone* bone = human->bones + bone_e::lower_left_arm;
    bone->parent_index = bone_e::upper_left_arm;

    bodyDef.position = b2Add(b2Vec2{ -0.1f, -0.925f * s }, position);
    bodyDef.linearDamping = 0.1f;
    bone->visual = fan::graphics::physics_shapes::capsule_sprite_t{ {
        .position = fan::vec3(fan::vec2(bodyDef.position), 15),
        .center0 = { 0.0f, 0.125f * s },
        .center1 = { 0.0f, -0.125f * s},
        .radius = 0.064f * s,
        .color = color,
        .image = images[bone_e::lower_left_arm],
        .body_type = fan::physics::body_type_e::dynamic_body,
        .shape_properties{ .fixed_rotation = false, .linear_damping = 0.5f,
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
    jointDef.lowerAngle = -0.5f * fan::math::pi;
    jointDef.upperAngle = 0.5f * fan::math::pi;
    jointDef.enableMotor = enableMotor;
    jointDef.maxMotorTorque = bone->friction_scale * maxTorque * scale;
    jointDef.enableSpring = hertz > 0.0f;
    jointDef.hertz = hertz*5;
    jointDef.dampingRatio = dampingRatio;
    jointDef.drawSize = drawSize;

    bone->joint_id = b2CreateRevoluteJoint(worldId, &jointDef);
  }

  // upper right arm
  {
    Bone* bone = human->bones + bone_e::upper_right_arm;
    bone->parent_index = bone_e::torso;

    bodyDef.position = b2Add(b2Vec2{ 0.0f, -1.145f * s }, position);
    bodyDef.linearDamping = 0.0f;
    bone->visual = fan::graphics::physics_shapes::capsule_sprite_t{ {
        .position = fan::vec3(fan::vec2(bodyDef.position), 100),
        .center0 = { 0.0f, 0.125f * s },
        .center1 = { 0.0f, -0.125f * s},
        .radius = 0.064f * s,
        .color = color,
        .image = images[bone_e::upper_right_arm],
        .body_type = fan::physics::body_type_e::dynamic_body,
        .shape_properties{ .fixed_rotation = false, .linear_damping = 0.5f,
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
    jointDef.lowerAngle = -0.5f * fan::math::pi;
    jointDef.upperAngle = 0.5f * fan::math::pi;
    jointDef.enableMotor = enableMotor;
    jointDef.maxMotorTorque = bone->friction_scale * maxTorque * scale;
    jointDef.enableSpring = hertz > 0.0f;
    jointDef.hertz = hertz*5;
    jointDef.dampingRatio = dampingRatio;
    jointDef.drawSize = drawSize;

    bone->joint_id = b2CreateRevoluteJoint(worldId, &jointDef);
  }

  // lower right arm
  {
    Bone* bone = human->bones + bone_e::lower_right_arm;
    bone->parent_index = bone_e::upper_right_arm;

    bodyDef.position = b2Add(b2Vec2{ -0.1f, -0.975f * s }, position);
    bodyDef.linearDamping = 0.1f;
    bone->visual = fan::graphics::physics_shapes::capsule_sprite_t{ {
        .position = fan::vec3(fan::vec2(bodyDef.position), 110),
        .center0 = { 0.0f,  0.125f * s },
        .center1 = { 0.0f, -0.125f * s },
        .radius = 0.064f * s,
        .color = color,
        .image = images[bone_e::lower_right_arm],
        .body_type = fan::physics::body_type_e::dynamic_body,
        .shape_properties{ .fixed_rotation = false, .linear_damping = 0.5f,
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
    jointDef.lowerAngle = -0.5f * fan::math::pi;
    jointDef.upperAngle = 0.5f * fan::math::pi;
    jointDef.enableMotor = enableMotor;
    jointDef.maxMotorTorque = bone->friction_scale * maxTorque * scale;
    jointDef.enableSpring = hertz > 0.0f;
    jointDef.hertz = hertz*5;
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

  Bone& blower_left_arm = human->bones[bone_e::lower_left_arm];
  Bone& blower_right_arm = human->bones[bone_e::lower_right_arm];
  Bone& bupper_left_leg = human->bones[bone_e::upper_left_leg];
  Bone& bupper_right_leg = human->bones[bone_e::upper_right_leg];
  Bone& blower_left_leg = human->bones[bone_e::lower_left_leg];
  Bone& blower_right_leg = human->bones[bone_e::lower_right_leg];

  f32_t torso_vel_x = b2Body_GetLinearVelocity(torsoId).x;
  f32_t torso_vel_y = b2Body_GetLinearVelocity(torsoId).y;
  int vel_sgn = fan::math::sgn(torso_vel_x);
  int force_sgn = fan::math::sgn(force);
  f32_t swing_speed = torso_vel_x ? (vel_sgn * 0.f + torso_vel_x / 15.f) : 0;

  f32_t ttransform = b2Rot_GetAngle(b2Body_GetRotation(human->bones[bone_e::torso].visual));
  f32_t lutransform = b2Rot_GetAngle(b2Body_GetRotation(bupper_left_leg.visual));
  f32_t rutransform = b2Rot_GetAngle(b2Body_GetRotation(bupper_right_leg.visual));

  f32_t lltransform = b2Rot_GetAngle(b2Body_GetRotation(blower_left_leg.visual));
  f32_t rltransform = b2Rot_GetAngle(b2Body_GetRotation(blower_right_leg.visual));

  if (std::abs(torso_vel_x) / 130.f > 1.f && torso_vel_x) {
    for (int i = 0; i < bone_e::bone_count; ++i) {
      human->bones[i].visual.set_tc_size(fan::vec2(vel_sgn, 1));
    }
  }

  if (torso_vel_x) {
    if (!force) {
      b2Body_ApplyForceToCenter(torsoId, fan::vec2(-torso_vel_x*100000.f, 0), true);
    }

    f32_t quarter_pi = -0.25f * fan::math::pi;
    //quarter_pi *= 3; // why this is required?
    //quarter_pi += fan::math::pi;

    if (std::abs(torso_vel_x) / 130.f > 1.f && torso_vel_x) {
      UpdateReferenceAngle(gloco->physics_context.world_id, blower_left_arm.joint_id, vel_sgn == 1 ? quarter_pi : -quarter_pi);
      UpdateReferenceAngle(gloco->physics_context.world_id, blower_right_arm.joint_id, vel_sgn == 1 ? quarter_pi : -quarter_pi);
      human->look_direction = vel_sgn;
    }

    if (force || std::abs(torso_vel_x / 100.f) > 1.f) {
      f32_t leg_turn =  0.4;

      if (rutransform < (human->look_direction == 1 ? -leg_turn / 2 : -leg_turn)) {
        human->direction = 0;
      }
      if (rutransform > (human->look_direction == -1 ? leg_turn / 2 : leg_turn)) {
        human->direction = 1;
      }

      f32_t rotate_speed = 1.3 * std::abs(torso_vel_x) / 200.f;

      if (human->direction == 1) {
        b2RevoluteJoint_SetMotorSpeed(bupper_right_leg.joint_id, -rotate_speed);
        b2RevoluteJoint_SetMotorSpeed(bupper_left_leg.joint_id, rotate_speed);
        
      }
      else {
        b2RevoluteJoint_SetMotorSpeed(bupper_right_leg.joint_id, rotate_speed);
        b2RevoluteJoint_SetMotorSpeed(bupper_left_leg.joint_id, -rotate_speed);
        
      }
      b2RevoluteJoint_SetMotorSpeed(blower_right_leg.joint_id, (human->look_direction * leg_turn/2 - rltransform));
      b2RevoluteJoint_SetMotorSpeed(blower_left_leg.joint_id,(human->look_direction * leg_turn/2 - lltransform));
    }
    else {
      b2RevoluteJoint_SetMotorSpeed(bupper_left_leg.joint_id, (ttransform - lutransform) * 5);
      b2RevoluteJoint_SetMotorSpeed(bupper_right_leg.joint_id, (ttransform - rutransform) * 5);

      b2RevoluteJoint_SetMotorSpeed(blower_left_leg.joint_id, (ttransform - lltransform) * 5);
      b2RevoluteJoint_SetMotorSpeed(blower_right_leg.joint_id, (ttransform - rltransform) * 5);
    }
  }
}

void fan::graphics::human_animate_jump(Human* human, f32_t impulse, f32_t dt, bool is_jumping) {
  Bone& bupper_left_leg = human->bones[bone_e::upper_left_leg];
  Bone& bupper_right_leg = human->bones[bone_e::upper_right_leg];
  Bone& blower_left_leg = human->bones[bone_e::lower_left_leg];
  Bone& blower_right_leg = human->bones[bone_e::lower_right_leg];
  f32_t massScale = 250.0f;
  //impulse *= massScale;
  if (is_jumping) {
    human->go_up = 0;
  }
  if (human->go_up == 1 && !human->jump_animation_timer.finished()) {
    b2Body_ApplyLinearImpulseToCenter(human->bones[bone_e::torso].visual, fan::vec2(0, impulse/2.f), true);
  }
  else if (human->go_up == 1 && human->jump_animation_timer.finished()) {
    b2Body_ApplyLinearImpulseToCenter(human->bones[bone_e::torso].visual, fan::vec2(0, -impulse), true);
    human->go_up = 0;
  }
  if (human->go_up == 0 && is_jumping) {
    //f32_t torso_vel_x = b2Body_GetLinearVelocity(human->bones[bone_e::torso].visual).x;
    //b2RevoluteJoint_SetSpringHertz(blower_left_leg.joint_id, 1);
    //b2RevoluteJoint_SetSpringHertz(blower_right_leg.joint_id, 1);

    //b2RevoluteJoint_SetMotorSpeed(blower_left_leg.joint_id, fan::math::sgn(torso_vel_x) * 10.2 );
    //b2RevoluteJoint_SetMotorSpeed(blower_left_leg.joint_id, fan::math::sgn(torso_vel_x) * 10.2 );

    //b2RevoluteJoint_SetMotorSpeed(bupper_left_leg.joint_id,  fan::math::sgn(torso_vel_x) *  -10.2 );
    //b2RevoluteJoint_SetMotorSpeed(bupper_right_leg.joint_id, fan::math::sgn(torso_vel_x) *  -10.2);

    human->go_up = 1;
    human->jump_animation_timer.start(0.09e9);
  }
}

fan::graphics::human_t::human_t(const fan::vec2& position, const f32_t scale, const std::array<loco_t::image_t, bone_e::bone_count>& images, const fan::color& color) {
  load(position, scale, images, color);
}

void fan::graphics::human_t::load(const fan::vec2& position, const f32_t scale, const std::array<loco_t::image_t, bone_e::bone_count>& images, const fan::color& color) {
  f32_t m_jointFrictionTorque = 0.03f;
  f32_t _jointHertz = 3.0f;
  f32_t _jointDampingRatio = 0.5f;
  fan::graphics::CreateHuman(
    dynamic_cast<Human*>(this),
    gloco->physics_context.world_id,
    position,
    scale,
    m_jointFrictionTorque,
    _jointHertz,
    _jointDampingRatio,
    1,
    nullptr,
    images,
    color
  );
}
void fan::graphics::human_t::animate_walk(f32_t force, f32_t dt) {
  human_animate_walk(dynamic_cast<Human*>(this), force, dt);
}
void fan::graphics::human_t::animate_jump(f32_t force, f32_t dt, bool is_jumping) {
  human_animate_jump(dynamic_cast<Human*>(this), force, dt, is_jumping);
}

bool fan::graphics::mouse_joint_t::QueryCallback(b2ShapeId shapeId, void* context) {
  QueryContext* queryContext = static_cast<QueryContext*>(context);

  b2BodyId bodyId = b2Shape_GetBody(shapeId);
  b2BodyType bodyType = b2Body_GetType(bodyId);
  if (bodyType != b2_dynamicBody)
  {
    // continue query
    return true;
  }

  bool overlap = b2Shape_TestPoint(shapeId, queryContext->point);
  if (overlap)
  {
    // found shape
    queryContext->bodyId = bodyId;
    return false;
  }

  return true;
}

fan::graphics::mouse_joint_t::mouse_joint_t(fan::physics::body_id_t tb) {
  this->target_body = tb;
}

void fan::graphics::mouse_joint_t::update_mouse(b2WorldId world_id, const fan::vec2& position) {
  if (ImGui::IsMouseDown(0)) {
    fan::vec2 p = gloco->get_mouse_position();
    if (!B2_IS_NON_NULL(mouse_joint)) {
      b2AABB box;
      b2Vec2 d = { 0.001f, 0.001f };
      box.lowerBound = b2Sub(p, d);
      box.upperBound = b2Add(p, d);

      QueryContext queryContext = { p, b2_nullBodyId };
      b2World_OverlapAABB(world_id, box, b2DefaultQueryFilter(), QueryCallback, &queryContext);
      if (B2_IS_NON_NULL(queryContext.bodyId)) {

        b2MouseJointDef mouseDef = b2DefaultMouseJointDef();
        mouseDef.bodyIdA = target_body;
        mouseDef.bodyIdB = queryContext.bodyId;
        mouseDef.target = p;
        mouseDef.hertz = 5.0f;
        mouseDef.dampingRatio = 0.7f;
        mouseDef.maxForce = 100000.0f * b2Body_GetMass(queryContext.bodyId);
        mouse_joint = b2CreateMouseJoint(world_id, &mouseDef);
        b2Body_SetAwake(queryContext.bodyId, true);
      }
    }
    else {
      b2MouseJoint_SetTarget(mouse_joint, p);
      b2BodyId bodyIdB = b2Joint_GetBodyB(mouse_joint);
      b2Body_SetAwake(bodyIdB, true);
    }
  }
  else if (ImGui::IsMouseReleased(0)) {
    if (B2_IS_NON_NULL(mouse_joint)) {
      b2DestroyJoint(mouse_joint);
      mouse_joint = b2_nullJointId;
    }
  }
}
