import fan;

struct pile_t {
  loco_t loco;
};

int main() {
  pile_t pile;
  fan::vec2 window_size = pile.loco.window.get_size();
  f32_t wall_thickness = 50.f;
  auto walls = fan::graphics::physics::create_stroked_rectangle(
    window_size / 2, window_size / 2, wall_thickness, fan::color::hex(0x6e8d6eff),
    {
      fan::physics::shape_properties_t{.friction=1.0}, 
      fan::physics::shape_properties_t{.friction=1.0}, 
      fan::physics::shape_properties_t{.friction=1.0}, 
      fan::physics::shape_properties_t{.friction=1.0}
    }
  );

  fan::graphics::physics::rectangle_t platforms[2];
  platforms[0] = fan::graphics::physics::rectangle_t{ {
    .position = fan::vec2(window_size.x / 5, window_size.y / 1.5),
    .size = fan::vec2(wall_thickness * 4, wall_thickness / 4),
    .color = fan::color::hex(0x30a6b6ff),
    .outline_color = fan::color::hex(0x30a6b6ff) * 2,
    .body_type = fan::physics::body_type_e::kinematic_body,
    .shape_properties{.presolve_events = false},
  } };
  platforms[1] = fan::graphics::physics::rectangle_t{ {
    .position = fan::vec2(500, 500),
    .size = wall_thickness / 4,
    .color = fan::color::hex(0x30a6b6ff),
    .outline_color = fan::color::hex(0x30a6b6ff) * 2,
    .body_type = fan::physics::body_type_e::static_body,
    .shape_properties{},
  } };

  loco_t::image_load_properties_t lp;
  lp.visual_output = loco_t::image_sampler_address_mode::repeat;
  auto character_images = fan::graphics::physics::human_t::load_character_images("characters/oldman", lp);

  fan::graphics::physics::human_t human({ 500.f, 500.0f }, 0.5f);


  fan::graphics::physics::character2d_t character;
  character.set_body(human.bones[fan::graphics::physics::bone_e::torso].visual);
  character.force = 100.f;
  character.impulse = 100.f;
  character.max_speed = 1000.f;
  //character.max_speed = character.max_speed / (character.max_speed  / human.scale);
  character.feet[0] = human.bones[fan::graphics::physics::bone_e::lower_left_leg].visual;
  character.feet[1] = human.bones[fan::graphics::physics::bone_e::lower_right_leg].visual;
  character.handle_jump = false;
  //fan::graphics::mouse_joint_t mouse_joint(human.bones[fan::graphics::physics::bone_e::torso].visual);

  pile.loco.loop([&] {

    //mouse_joint.update_mouse(pile.loco.physics_context.world_id, pile.loco.get_mouse_position());

//    human.animate_walk(character.walk_force, pile.loco.delta_time);
    human.animate_jump(character.impulse, pile.loco.delta_time, character.jumping);

    character.process_movement();

    pile.loco.physics_context.step(pile.loco.delta_time);
    if (platforms[0].get_position().x < window_size.x / 4) {
      platforms[0].set_linear_velocity({200, 0});
    }
    else if (platforms[0].get_position().x > window_size.x / 1.5) {
      platforms[0].set_linear_velocity({ -200, 0 });
    }
  });
}