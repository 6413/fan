#include "physics_shapes.hpp"


void fan::graphics::physics::shape_physics_update(const loco_t::physics_update_data_t& data) {
  if (!b2Body_IsValid(data.body_id)) {
    fan::print("invalid body data (corruption)");
    return;
  }
  if (b2Body_GetType(data.body_id) == b2_staticBody) {
    return;
  }

  fan::vec2 p = b2Body_GetWorldPoint(data.body_id, fan::vec2(0));
  b2Rot rotation = b2Body_GetRotation(data.body_id);
  f32_t radians = b2Rot_GetAngle(rotation);
   
  loco_t::shape_t& shape = *(loco_t::shape_t*)&data.shape_id;
  shape.set_position(p * fan::physics::length_units_per_meter);
  shape.set_angle(fan::vec3(0, 0, radians));
  b2ShapeId id[1];
  if (b2Body_GetShapes(data.body_id, id, 1)) {
    auto aabb = b2Shape_GetAABB(id[0]);
    fan::vec2 size = fan::vec2(aabb.upperBound - aabb.lowerBound) / 2;
    fan::graphics::physics::physics_update_cb(shape, shape.get_position(), size* fan::physics::length_units_per_meter/2, radians);
  }
  //hitbox_visualize[(void*) & data.body_id] = fan::graphics::rectangle_t{{
  //    .position = fan::vec3(p * fan::physics::length_units_per_meter, 0xffff-100),
  //    .size = 30,
  //    .color = fan::colors::green
  //}};


  // joint debug
  
 /*  int joint_count = b2Body_GetJointCount(data.body_id);
  if (joint_count > 0) {
    std::vector<fan::physics::joint_id_t> joints(joint_count);
    b2Body_GetJoints(data.body_id, joints.data(), joint_count);
    joint_visualize[&shape].clear();
    for (fan::physics::joint_id_t joint_id : joints) {
      fan::vec2 anchor_a = b2Joint_GetLocalAnchorA(joint_id);
      fan::vec2 anchor_b = b2Joint_GetLocalAnchorB(joint_id);
      b2BodyId body_a = b2Joint_GetBodyA(joint_id);
      b2BodyId body_b = b2Joint_GetBodyB(joint_id);
      fan::vec2 world_anchor_a = b2Body_GetWorldPoint(body_a, anchor_a);
      fan::vec2 world_anchor_b = b2Body_GetWorldPoint(body_b, anchor_b);

      joint_visualize[&shape].emplace_back(fan::graphics::circle_t{{
        .position = fan::vec3(world_anchor_a * fan::physics::length_units_per_meter, 60002),
        .radius = 3,
        .color = fan::color(0, 0, 1, 0.5),
        .blending = true
      }});

      joint_visualize[&shape].emplace_back(fan::graphics::line_t{{
        .src = fan::vec3(p * fan::physics::length_units_per_meter, 60001),
        .dst = fan::vec3(world_anchor_b * fan::physics::length_units_per_meter, 60001),
        .color = fan::color(1, 0, 0, 0.5),
        .blending = true
      }});
    }
  }*/
}

std::array<fan::graphics::physics::rectangle_t, 4> fan::graphics::physics::create_stroked_rectangle(
  const fan::vec2& center_position, 
  const fan::vec2& half_size,
  f32_t thickness,
  const fan::color& wall_color, 
  std::array<fan::physics::shape_properties_t, 4> shape_properties
) {
  std::array<fan::graphics::physics::rectangle_t, 4> walls;
  const fan::color wall_outline = wall_color * 2;
  // top
  walls[0] = fan::graphics::physics::rectangle_t{ {
      .position = fan::vec2(center_position.x, center_position.y - half_size.y),
      .size = fan::vec2(half_size.x * 2, thickness),
      .color = wall_color,
      .outline_color = wall_outline,
      .shape_properties = shape_properties[0]
    } };
  // bottom
  walls[1] = fan::graphics::physics::rectangle_t{ {
      .position = fan::vec2(center_position.x, center_position.y + half_size.y),
      .size = fan::vec2(half_size.x * 2, thickness),
      .color = wall_color,
      .outline_color = wall_color,
      .shape_properties = shape_properties[1]
    } };
  // left
  walls[2] = fan::graphics::physics::rectangle_t{ {
      .position = fan::vec2(center_position.x - half_size.x, center_position.y),
      .size = fan::vec2(thickness, half_size.y * 2),
      .color = wall_color,
      .outline_color = wall_outline,
      .shape_properties = shape_properties[2]
    } };
  // right
  walls[3] = fan::graphics::physics::rectangle_t{ {
      .position = fan::vec2(center_position.x + half_size.x, center_position.y),
      .size = fan::vec2(thickness, half_size.y * 2),
      .color = wall_color,
      .outline_color = wall_outline,
      .shape_properties = shape_properties[3]
    } };
  return walls;
}

fan::graphics::physics::character2d_t::character2d_t() {
  add_inputs();
}

void fan::graphics::physics::character2d_t::add_inputs() {
  gloco->input_action.add(fan::key_a, "move_left");
  gloco->input_action.add(fan::key_d, "move_right");
  gloco->input_action.add(fan::key_space, "move_up");
  gloco->input_action.add(fan::key_s, "move_down");
}

bool fan::graphics::physics::character2d_t::is_on_ground(fan::physics::body_id_t main, std::array<fan::physics::body_id_t, 2> feet, bool jumping) {
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

void fan::graphics::physics::character2d_t::process_movement(uint8_t movement, f32_t friction) {
  fan::vec2 velocity = get_linear_velocity();

  auto movement_left_right = [&] {
    walk_force = 0;
    if (gloco->input_action.is_action_down("move_left")) {
      move_to_direction(fan::vec2(-1, 0));
    }
    if (gloco->input_action.is_action_down("move_right")) {
      move_to_direction(fan::vec2(1, 0));
    }
  };
  auto movement_up_down = [&] {
    walk_force = 0;
    if (gloco->input_action.is_action_down("move_up")) {
      move_to_direction(fan::vec2(0, -1));
    }
    if (gloco->input_action.is_action_down("move_down")) {
      move_to_direction(fan::vec2(0, 1));
    }
  };
  switch (movement) {
  case movement_e::side_view: {

    movement_left_right();

    bool can_jump = false;

    can_jump = is_on_ground(*this, std::to_array(feet), jumping);

    bool move_up = gloco->input_action.is_action_clicked("move_up");
    if (move_up) {
      if (can_jump) {
        if (handle_jump) {
          set_linear_velocity(fan::vec2(get_linear_velocity().x, 0));
          apply_linear_impulse_center({ 0, -impulse });
        }
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
    break;
  }
  case movement_e::top_view: {
    movement_left_right();
    movement_up_down();
    break;
  }
  }
}

void fan::graphics::physics::character2d_t::move_to_direction(const fan::vec2& direction){
  fan::vec2 velocity = get_linear_velocity();
  if (direction.x < 0) {
    if (velocity.x > -max_speed) {
      apply_force_center({ -force, 0 });
    }
  }
  else if (direction.x > 0) {
    if (velocity.x <= max_speed) {
      apply_force_center({ force, 0 });
    }
  }
  if (direction.y < 0) {
    if (velocity.y > -max_speed) {
      apply_force_center({ 0, -force });
    }
  }
  if (direction.y > 0) {
    if (velocity.y <= max_speed) {
      apply_force_center({ 0, force });
    }
  }
}

void fan::graphics::physics::update_reference_angle(b2WorldId world, fan::physics::joint_id_t& joint_id, float new_reference_angle) {

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


void update_position(b2WorldId world, fan::physics::joint_id_t& joint_id, fan::vec2 position) {

    b2BodyId bodyIdA = b2Joint_GetBodyA(joint_id);
    b2BodyId bodyIdB = b2Joint_GetBodyB(joint_id);

    b2Vec2 localAnchorA = b2Body_GetLocalPoint(bodyIdA, position);
    b2Vec2 localAnchorB = b2Body_GetLocalPoint(bodyIdB, position);
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
    //jointDef.referenceAngle = new_reference_angle;
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

fan::graphics::physics::human_t::human_t(const fan::vec2& position, const f32_t scale, const std::array<loco_t::image_t, bone_e::bone_count>& images, const fan::color& color) {
  load(position, scale, images, color);
}

void fan::graphics::physics::human_t::load_bones(const fan::vec2& position, f32_t scale, std::array<fan::graphics::physics::bone_t, fan::graphics::physics::bone_e::bone_count>& bones) {
  for (int i = 0; i < fan::graphics::physics::bone_e::bone_count; ++i) {
    bones[i].joint_id = b2_nullJointId;
    bones[i].friction_scale = 1.0f;
    bones[i].parent_index = -1;
  }

  struct bone_data_t {
    int parent_index;
    fan::vec3 position;
    f32_t size;
    f32_t friction_scale;
    fan::vec2 pivot;
    f32_t lower_angle;
    f32_t upper_angle;
    f32_t reference_angle;
    fan::vec2 center0;
    fan::vec2 center1;
  };

   bone_data_t bone_data[] = {
    { // hip
      .parent_index = -1,
      .position = {0.0f, -0.95f, 55},
      .size = 0.095f,
      .friction_scale = 1.0f,
      .pivot = {0.0f, 0.0f},
      .lower_angle = 0.f,
      .upper_angle = 0.f,
      .reference_angle = 0.f,
      .center0 = {0.f, -0.02f},
      .center1 = {0.f, 0.02f}
    },
    { // torso
      .parent_index = fan::graphics::physics::bone_e::hip,
      .position = {0.0f, -1.2f, 60},
      .size = 0.09f,
      .friction_scale = 0.5f,
      .pivot = {0.0f, -1.0f},
      .lower_angle = -0.25f * fan::math::pi,
      .upper_angle = 0.f,
      .reference_angle = 0.f,
      .center0 = {0.f, -0.135f},
      .center1 = {0.f, 0.135f}
    },
    { // head
      .parent_index = fan::graphics::physics::bone_e::torso,
      .position = {0.0f, -1.475f, 44},
      .size = 0.075f,
      .friction_scale = 0.25f,
      .pivot = {0.0f, -1.4f},
      .lower_angle = -0.3f * fan::math::pi,
      .upper_angle = 0.1f * fan::math::pi,
      .reference_angle = 0.f,
      .center0 = {0.f, -0.038f},
      .center1 = {0.f, 0.039f}
    },
    { // upper left leg
      .parent_index = fan::graphics::physics::bone_e::hip,
      .position = {0.0f, -0.775f, 52},
      .size = 0.06f,
      .friction_scale = 1.0f,
      .pivot = {0.0f, -0.9f},
      .lower_angle = -0.5f * fan::math::pi,
      .upper_angle = 0.5f * fan::math::pi,
      .reference_angle = 0.f,
      .center0 = {0.f, -0.125f},
      .center1 = {0.f, 0.125f}
    },
    { // lower left leg
      .parent_index = fan::graphics::physics::bone_e::upper_left_leg,
      .position = {0.0f, -0.475f, 51},
      .size = 0.045f,
      .friction_scale = 0.5f,
      .pivot = {0.0f, -0.625f},
      .lower_angle = -0.5f * fan::math::pi,
      .upper_angle = 0.5f * fan::math::pi,
      .reference_angle = 0.f,
      .center0 = {0.f, -0.125f},
      .center1 = {0.f, 0.045f}
    },
    { // upper right leg
      .parent_index = fan::graphics::physics::bone_e::hip,
      .position = {0.0f, -0.775f, 54},
      .size = 0.06f,
      .friction_scale = 1.0f,
      .pivot = {0.0f, -0.9f},
      .lower_angle = -0.5f * fan::math::pi,
      .upper_angle = 0.5f * fan::math::pi,
      .reference_angle = 0.f,
      .center0 = {0.f, -0.125f},
      .center1 = {0.f, 0.125f}
    },
    { // lower right leg
      .parent_index = fan::graphics::physics::bone_e::upper_right_leg,
      .position = {0.0f, -0.475f, 53},
      .size = 0.045f,
      .friction_scale = 0.5f,
      .pivot = {0.0f, -0.625f},
      .lower_angle = -0.5f * fan::math::pi,
      .upper_angle = 0.5f * fan::math::pi,
      .reference_angle = 0.f,
      .center0 = {0.f, -0.155f},
      .center1 = {0.f, 0.125f}
    },
    { // upper left arm
      .parent_index = fan::graphics::physics::bone_e::torso,
      .position = {0.0f, -1.225f, 62},
      .size = 0.035f,
      .friction_scale = 0.5f,
      .pivot = {0.0f, -1.35f},
      .lower_angle = -0.5f * fan::math::pi,
      .upper_angle = 0.5f * fan::math::pi,
      .reference_angle = 0.f,
      .center0 = {0.f, -0.125f},
      .center1 = {0.f, 0.125f}
    },
    { // lower left arm
      .parent_index = fan::graphics::physics::bone_e::upper_left_arm,
      .position = {0.0f, -0.975f, 61},
      .size = 0.03f,
      .friction_scale = 0.1f,
      .pivot = {0.0f, -1.1f},
      .lower_angle = -0.5f * fan::math::pi,
      .upper_angle = 0.5f * fan::math::pi,
      .reference_angle = -0.25f * fan::math::pi,
      .center0 = {0.f, -0.125f},
      .center1 = {0.f, 0.125f}
    },
    { // upper right arm
      .parent_index = fan::graphics::physics::bone_e::torso,
      .position = {0.0f, -1.225f, 64},
      .size = 0.035f,
      .friction_scale = 0.5f,
      .pivot = {0.0f, -1.35f},
      .lower_angle = -0.5f * fan::math::pi,
      .upper_angle = 0.5f * fan::math::pi,
      .reference_angle = 0.f,
      .center0 = {0.f, -0.125f},
      .center1 = {0.f, 0.125f}
    },
    { // lower right arm
      .parent_index = fan::graphics::physics::bone_e::upper_right_arm,
      .position = {0.0f, -0.975f, 63},
      .size = 0.03f,
      .friction_scale = 0.1f,
      .pivot = {0.0f, -1.1f},
      .lower_angle = -0.5f * fan::math::pi,
      .upper_angle = 0.5f * fan::math::pi,
      .reference_angle = -0.25f * fan::math::pi,
      .center0 = {0.f, -0.125f},
      .center1 = {0.f, 0.125f}
    }
  };

  for (int i = 0; i < std::size(bone_data); ++i) {
    bones[i].parent_index = bone_data[i].parent_index;
    bones[i].position = fan::vec2(bone_data[i].position) * scale;
    bones[i].position.z = bone_data[i].position.z;
    bones[i].size = bone_data[i].size * scale;
    bones[i].friction_scale = bone_data[i].friction_scale;
    bones[i].pivot = bone_data[i].pivot * scale;
    bones[i].lower_angle = bone_data[i].lower_angle;
    bones[i].upper_angle = bone_data[i].upper_angle;
    bones[i].reference_angle = bone_data[i].reference_angle;
    bones[i].center0 = bone_data[i].center0 * scale;
    bones[i].center1 = bone_data[i].center1 * scale;
  }
}

void fan::graphics::physics::human_t::load_preset(const fan::vec2& position, const f32_t scale, const bone_images_t& images, std::array<bone_t, bone_e::bone_count>& bones, const fan::color& color) {
  this->scale = scale;
  int groupIndex = 1;
  f32_t frictionTorque = 0.03f;
  f32_t hertz = 5.0f;
  f32_t dampingRatio = 0.5f;
  b2WorldId worldId = gloco->physics_context.world_id;

  b2Filter filter = b2DefaultFilter();

  filter.groupIndex = -groupIndex;
  filter.categoryBits = 2;
  filter.maskBits = (1 | 2); 

  f32_t maxTorque = frictionTorque * scale*1000;
  bool enableMotor = true;
  bool enableLimit = true;

  for (int i = 0; i < std::size(bones); ++i) {
    auto& bone = bones[i];
    bone.visual = fan::graphics::physics::capsule_sprite_t{ {
      .position = fan::vec3(position + (fan::vec2(bone.position) * fan::physics::length_units_per_meter + bone.offset) * scale, bone.position.z),
      /*
        bone.center0 * fan::physics::length_units_per_meter * bone.scale * scale
        bone.center1 * fan::physics::length_units_per_meter * bone.scale * scale
      */
      .center0 =fan::vec2(0),
      .center1 =fan::vec2(0),
      .radius = fan::physics::length_units_per_meter * bone.size.y * bone.scale  * scale,
      .color = color,
      .image = images[i],
      .body_type = fan::physics::body_type_e::dynamic_body,
      .shape_properties{ 
        .friction=0.6,
        .fixed_rotation = i == bone_e::hip || i == bone_e::torso,
        .linear_damping = 0.0f,
        .filter = filter 
      },
    } };

    if (bone.parent_index == -1) {
      continue;
    }
    fan::vec2 physics_position = bone.visual.get_physics_position();
    fan::vec2 pivot = (position / fan::physics::length_units_per_meter) + bone.pivot * scale;
  //  fan::graphics::physics::hitbox_visualize[&bones[i]] = fan::graphics::rectangle_t{{
  //  .position = fan::vec3(position + bone.pivot * scale * fan::physics::length_units_per_meter, 60001),
  //  .size=5,
  //  .color = fan::color(0, 0, 1, 0.2),
  //  .outline_color=fan::color(0, 0, 1, 0.2)*2,
  //  .blending=true
  //}};
    b2RevoluteJointDef joint_def = b2DefaultRevoluteJointDef();
    joint_def.bodyIdA = bones[bone.parent_index].visual;
    joint_def.bodyIdB = bone.visual;
    joint_def.localAnchorA = b2Body_GetLocalPoint(joint_def.bodyIdA, pivot);
    joint_def.localAnchorB = b2Body_GetLocalPoint(joint_def.bodyIdB, pivot);
    joint_def.referenceAngle = bone.reference_angle;
    joint_def.enableLimit = enableLimit;
    joint_def.lowerAngle = bone.lower_angle;
    joint_def.upperAngle = bone.upper_angle;
    joint_def.enableMotor = enableMotor;
    joint_def.maxMotorTorque = bone.friction_scale * maxTorque;
    joint_def.enableSpring = hertz > 0.0f;
    joint_def.hertz = hertz;
    joint_def.dampingRatio = dampingRatio;
    
   bone.joint_id = b2CreateRevoluteJoint(worldId, &joint_def);
  }
  is_spawned = true;
}

void fan::graphics::physics::human_t::load(const fan::vec2& position, const f32_t scale, const std::array<loco_t::image_t, bone_e::bone_count>& images, const fan::color& color) {
  load_bones(position, scale, bones);
  load_preset(position, scale, images, bones, color);
}

fan::graphics::physics::human_t::bone_images_t fan::graphics::physics::human_t::load_character_images(
  const std::string& character_folder_path,
  const loco_t::image_load_properties_t& lp
){
  fan::graphics::physics::human_t::bone_images_t character_images;
  character_images[fan::graphics::physics::bone_e::head] = gloco->image_load(character_folder_path + "/head.webp", lp);
  character_images[fan::graphics::physics::bone_e::torso] = gloco->image_load(character_folder_path + "/torso.webp", lp);
  character_images[fan::graphics::physics::bone_e::hip] = gloco->image_load(character_folder_path + "/hip.webp", lp);
  character_images[fan::graphics::physics::bone_e::upper_left_leg] = gloco->image_load(character_folder_path + "/upper_leg.webp", lp);
  character_images[fan::graphics::physics::bone_e::lower_left_leg] = gloco->image_load(character_folder_path + "/lower_leg.webp", lp);
  character_images[fan::graphics::physics::bone_e::upper_right_leg] = character_images[fan::graphics::physics::bone_e::upper_left_leg];
  character_images[fan::graphics::physics::bone_e::lower_right_leg] = character_images[fan::graphics::physics::bone_e::lower_left_leg];
  character_images[fan::graphics::physics::bone_e::upper_left_arm] = gloco->image_load(character_folder_path + "/upper_arm.webp", lp);
  character_images[fan::graphics::physics::bone_e::lower_left_arm] = gloco->image_load(character_folder_path + "/lower_arm.webp", lp);
  character_images[fan::graphics::physics::bone_e::upper_right_arm] = character_images[fan::graphics::physics::bone_e::upper_left_arm];
  character_images[fan::graphics::physics::bone_e::lower_right_arm] = character_images[fan::graphics::physics::bone_e::lower_left_arm];
  return character_images;
}
void fan::graphics::physics::human_t::animate_walk(f32_t force, f32_t dt) {
  
  fan::physics::body_id_t torso_id = bones[bone_e::torso].visual;
  b2Vec2 force_ = { force, 0 };

  bone_t& blower_left_arm = bones[bone_e::lower_left_arm];
  bone_t& blower_right_arm = bones[bone_e::lower_right_arm];
  bone_t& bupper_left_leg = bones[bone_e::upper_left_leg];
  bone_t& bupper_right_leg = bones[bone_e::upper_right_leg];
  bone_t& blower_left_leg = bones[bone_e::lower_left_leg];
  bone_t& blower_right_leg = bones[bone_e::lower_right_leg];

  f32_t torso_vel_x = torso_id.get_linear_velocity().x;
  f32_t torso_vel_y = torso_id.get_linear_velocity().y;
  int vel_sgn = fan::math::sgn(torso_vel_x);

  int force_sgn = fan::math::sgn(force);
  f32_t swing_speed = torso_vel_x ? (vel_sgn * 0.f + torso_vel_x / 15.f) : 0;

  f32_t ttransform = b2Rot_GetAngle(b2Body_GetRotation(bones[bone_e::torso].visual));
  f32_t lutransform = b2Rot_GetAngle(b2Body_GetRotation(bupper_left_leg.visual));
  f32_t rutransform = b2Rot_GetAngle(b2Body_GetRotation(bupper_right_leg.visual));

  f32_t lltransform = b2Rot_GetAngle(b2Body_GetRotation(blower_left_leg.visual));
  f32_t rltransform = b2Rot_GetAngle(b2Body_GetRotation(blower_right_leg.visual));

  if (std::abs(torso_vel_x) / 130.f > 1.f && torso_vel_x) {
    for (int i = 0; i < bone_e::bone_count; ++i) {
      bones[i].visual.set_tc_size(fan::vec2(vel_sgn, 1));
      if (torso_vel_x < 0) {
        //b2Body_SetTransform(bones[i].visual,  bones[i].visual.get_physics_position() + fan::vec2(bones[bone_e::torso].visual.get_position().x - bones[i].visual.get_position().x, 0) / fan::physics::length_units_per_meter/2, b2Body_GetRotation(bones[i].visual));
        if (bones[i].joint_id.is_valid() == false) {
          continue;
        }
        static int x = 0;
        if (!x) {
          fan::vec2 pivot = fan::vec2(500, 300.f) / fan::physics::length_units_per_meter + bones[i].pivot * scale;
   //     update_position(gloco->physics_context.world_id, bones[i].joint_id, pivot);
        x++;
        }
        
      }
    }
  }

  if (torso_vel_x) {
    if (!force) {
   //   torsoId.apply_force_center(fan::vec2(-torso_vel_x, 0));
    }

    f32_t quarter_pi = -0.25f * fan::math::pi;
    //quarter_pi *= 3; // why this is required?
    //quarter_pi += fan::math::pi;
    if (std::abs(torso_vel_x) / 130.f > 1.f && torso_vel_x) {
   //   update_reference_angle(gloco->physics_context.world_id, blower_left_arm.joint_id, vel_sgn == 1 ? quarter_pi : -quarter_pi);
  //    update_reference_angle(gloco->physics_context.world_id, blower_right_arm.joint_id, vel_sgn == 1 ? quarter_pi : -quarter_pi);
      look_direction = vel_sgn;
    }

    if (force || std::abs(torso_vel_x / 10.f) > 1.f) {
      f32_t leg_turn =  0.4;

      if (rutransform < (look_direction == 1 ? -leg_turn / 2 : -leg_turn)) {
        direction = 0;
      }
      if (rutransform > (look_direction == -1 ? leg_turn / 2 : leg_turn)) {
        direction = 1;
      }

      f32_t rotate_speed = 1.3 * std::abs(torso_vel_x) / 200.f;

      if (direction == 1) {
        bupper_right_leg.joint_id.revolute_joint_set_motor_speed(-rotate_speed);
        bupper_left_leg.joint_id.revolute_joint_set_motor_speed(rotate_speed);
        
      }
      else {
        bupper_right_leg.joint_id.revolute_joint_set_motor_speed(rotate_speed);
        bupper_left_leg.joint_id.revolute_joint_set_motor_speed(-rotate_speed);
        
      }
      blower_right_leg.joint_id.revolute_joint_set_motor_speed(look_direction * leg_turn/4 - rltransform);
      blower_left_leg.joint_id.revolute_joint_set_motor_speed(look_direction * leg_turn/4 - lltransform);
    }
    else {
      bupper_left_leg.joint_id.revolute_joint_set_motor_speed((ttransform - lutransform) * 5);
      bupper_right_leg.joint_id.revolute_joint_set_motor_speed((ttransform - rutransform) * 5);

      blower_left_leg.joint_id.revolute_joint_set_motor_speed((ttransform - lltransform) * 5);
      blower_right_leg.joint_id.revolute_joint_set_motor_speed((ttransform - rltransform) * 5);
    }
  }

}
void fan::graphics::physics::human_t::animate_jump(f32_t impulse, f32_t dt, bool is_jumping) {
  bone_t& bupper_left_leg = bones[bone_e::upper_left_leg];
  bone_t& bupper_right_leg = bones[bone_e::upper_right_leg];
  bone_t& blower_left_leg = bones[bone_e::lower_left_leg];
  bone_t& blower_right_leg = bones[bone_e::lower_right_leg];
  if (is_jumping) {
    go_up = 0;
  }
  if (go_up == 1 && !jump_animation_timer.finished()) {
    bones[bone_e::torso].visual.apply_linear_impulse_center(fan::vec2(0, impulse/4));
  }
  else if (go_up == 1 && jump_animation_timer.finished()) {
    bones[bone_e::torso].visual.apply_linear_impulse_center(fan::vec2(0, -impulse));
    go_up = 0;
  }
  if (go_up == 0 && is_jumping) {
    //f32_t torso_vel_x = b2Body_GetLinearVelocity(bones[bone_e::torso].visual).x;
    //b2RevoluteJoint_SetSpringHertz(blower_left_leg.joint_id, 1);
    //b2RevoluteJoint_SetSpringHertz(blower_right_leg.joint_id, 1);

    //b2RevoluteJoint_SetMotorSpeed(blower_left_leg.joint_id, fan::math::sgn(torso_vel_x) * 10.2 );
    //b2RevoluteJoint_SetMotorSpeed(blower_left_leg.joint_id, fan::math::sgn(torso_vel_x) * 10.2 );

    //b2RevoluteJoint_SetMotorSpeed(bupper_left_leg.joint_id,  fan::math::sgn(torso_vel_x) *  -10.2 );
    //b2RevoluteJoint_SetMotorSpeed(bupper_right_leg.joint_id, fan::math::sgn(torso_vel_x) *  -10.2);

    go_up = 1;
    jump_animation_timer.start(0.09e9);
  }
}

void fan::graphics::physics::human_t::erase(){
  assert(is_spawned == true);

  for (int i = 0; i < bone_e::bone_count; ++i) {
    if (B2_IS_NULL(bones[i].joint_id))
    {
      continue;
    }

    if (b2Joint_IsValid(bones[i].joint_id)) {
      b2DestroyJoint(bones[i].joint_id);
      bones[i].joint_id = b2_nullJointId;
    }
  }
}

bool fan::graphics::physics::mouse_joint_t::QueryCallback(b2ShapeId shapeId, void* context) {
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

fan::graphics::physics::mouse_joint_t::mouse_joint_t() {
  
  auto default_body = b2DefaultBodyDef();
  dummy_body.set_body(b2CreateBody(gloco->physics_context, &default_body));
  nr = gloco->m_update_callback.NewNodeLast();
  // not copy safe
  gloco->m_update_callback[nr] = [this](loco_t* loco) {
#if defined(loco_imgui)
    if (ImGui::IsMouseDown(0)) {
      fan::vec2 p = gloco->get_mouse_position() / fan::physics::length_units_per_meter;
      if (!B2_IS_NON_NULL(mouse_joint)) {
        b2AABB box;
        b2Vec2 d = { 0.001f, 0.001f };
        box.lowerBound = b2Sub(p, d);
        box.upperBound = b2Add(p, d);

        QueryContext queryContext = { p, b2_nullBodyId };
        b2World_OverlapAABB(loco->physics_context, box, b2DefaultQueryFilter(), QueryCallback, &queryContext);
        if (B2_IS_NON_NULL(queryContext.bodyId)) {

          b2MouseJointDef mouseDef = b2DefaultMouseJointDef();
          mouseDef.bodyIdA = dummy_body;
          mouseDef.bodyIdB = queryContext.bodyId;
          mouseDef.target = p;
          mouseDef.hertz = 5.0f;
          mouseDef.dampingRatio = 0.7f;
          mouseDef.maxForce = 1000.0f * b2Body_GetMass(queryContext.bodyId);
          mouse_joint = b2CreateMouseJoint(loco->physics_context, &mouseDef);
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
#endif
  };
}

fan::graphics::physics::mouse_joint_t::~mouse_joint_t() {
  if (dummy_body.is_valid()) {
    dummy_body.destroy();
  }
  if (nr.iic() == false) {
    gloco->m_update_callback.unlrec(nr);
    nr.sic();
  }
}

int z_depth = 0;
std::vector<fan::graphics::line_t> debug_draw_polygon;
std::vector<fan::graphics::polygon_t> debug_draw_solid_polygon;
std::vector<fan::graphics::circle_t> debug_draw_circle;
std::vector<fan::graphics::line_t> debug_draw_line;
/// Draw a closed polygon provided in CCW order.
void DrawPolygon(const fan::vec2* vertices, int vertexCount, b2HexColor color, void* context) {
  for (int i = 0; i < vertexCount; i++) {
    int next_i = (i + 1) % vertexCount;
    
    debug_draw_polygon.emplace_back(fan::graphics::line_t{ {
      .src = fan::vec3(fan::physics::physics_to_render(vertices[i]), 0x1f00 + z_depth),
      .dst = fan::physics::physics_to_render(vertices[next_i]),
      .color = fan::color::hexa(color)
    }});
  }

  ++z_depth;
}

/// Draw a solid closed polygon provided in CCW order.
void DrawSolidPolygon(b2Transform transform, const b2Vec2* vertices, int vertexCount, float radius, b2HexColor color,
  void* context) {
  std::vector<fan::graphics::vertex_t> vs(vertexCount);
  for (auto [i, v] : fan::enumerate(vs)) {
    v.position = fan::physics::physics_to_render(vertices[i]);
    v.color = fan::color::hexa(color);
  }
  debug_draw_solid_polygon.emplace_back(fan::graphics::polygon_t{ {
    .position = fan::vec3(0, 0, 0x1f00 + z_depth),
    .vertices = vs,
    .draw_mode=fan::graphics::primitive_topology_t::triangle_fan,
  }});
  ++z_depth;
}

/// Draw a circle.
void DrawCircle(b2Vec2 center, float radius, b2HexColor color, void* context) {
  debug_draw_circle.emplace_back(fan::graphics::circle_t{ {
    .position = fan::vec3(fan::physics::physics_to_render(center), 0x1f00 + z_depth),
    .radius = fan::physics::physics_to_render(radius).x,
    .color = fan::color::hexa(color),
  }});
  ++z_depth;
}

/// Draw a solid circle.
void DrawSolidCircle(b2Transform transform, float radius, b2HexColor color, void* context) {
  debug_draw_circle.emplace_back(fan::graphics::circle_t{ {
    .position = fan::vec3(fan::physics::physics_to_render(transform.p), 0x1f00 + z_depth),
    .radius = fan::physics::physics_to_render(radius).x,
    .color = fan::color::hexa(color),
  }});
  ++z_depth;
}

/// Draw a capsule.
void DrawCapsule(b2Vec2 p1, b2Vec2 p2, float radius, b2HexColor color, void* context) {
  printf("DrawCapsule\n");
}

/// Draw a solid capsule.
void DrawSolidCapsule(b2Vec2 p1, b2Vec2 p2, float radius, b2HexColor color, void* context) {
  printf("DrawSolidCapsule\n");
}

/// Draw a line segment.
void DrawSegment(b2Vec2 p1, b2Vec2 p2, b2HexColor color, void* context) {
  debug_draw_line.emplace_back(fan::graphics::line_t{ {
    .src = fan::vec3(fan::physics::physics_to_render(p1), 0x1f00 + z_depth),
    .dst = fan::vec3(fan::physics::physics_to_render(p2), 0x1f00 + z_depth),
    .color = fan::color::hexa(color),
  }});
  ++z_depth;
}

/// Draw a transform. Choose your own length scale.
void DrawTransform(b2Transform transform, void* context) {
    printf("DrawTransform\n");
}

/// Draw a point.
void DrawPoint(b2Vec2 p, float size, b2HexColor color, void* context) {
  //vs.back() = vs.front();
  debug_draw_circle.emplace_back(fan::graphics::circle_t{{
    .position = fan::vec3(fan::physics::physics_to_render(p), 0x1f00 + z_depth),
    .radius = size,
    .color = fan::color::hexa(color)
  }});
  ++z_depth;
}

/// Draw a string.
void DrawString(b2Vec2 p, const char* s, void* context) {
  fan::graphics::text(s, fan::physics::physics_to_render(p));
}


b2DebugDraw initialize_debug(bool enabled) {
  return b2DebugDraw{
  .DrawPolygon = (decltype(b2DebugDraw::DrawPolygon))DrawPolygon,
  .DrawSolidPolygon = DrawSolidPolygon,
  .DrawCircle = DrawCircle,
  .DrawSolidCircle = DrawSolidCircle,
  .DrawCapsule = DrawCapsule,
  .DrawSolidCapsule = DrawSolidCapsule,
  .DrawSegment = DrawSegment,
  .DrawTransform = DrawTransform,
  .DrawPoint = DrawPoint,
  .DrawString = DrawString,
  .drawJoints = enabled,
  .drawAABBs = enabled,
  };
}

b2DebugDraw fan::graphics::physics::box2d_debug_draw = []{
  auto init_it = fan::graphics::engine_init_cbs.NewNodeLast();
  fan::graphics::engine_init_cbs[init_it] = [](loco_t* loco){
    auto it = loco->m_update_callback.NewNodeLast();
    loco->m_update_callback[it] = [](loco_t* loco) {
      z_depth = 0;
      debug_draw_polygon.clear();
      debug_draw_solid_polygon.clear();
      debug_draw_circle.clear();
      debug_draw_line.clear();
      b2World_Draw(loco->physics_context.world_id, &fan::graphics::physics::box2d_debug_draw);
    };
  };
  return initialize_debug(false);
}();

void fan::graphics::physics::step(f32_t dt) {
  gloco->physics_context.step(dt);
}

void fan::graphics::physics::debug_draw(bool enabled) {
  fan::graphics::physics::box2d_debug_draw = initialize_debug(enabled);
}