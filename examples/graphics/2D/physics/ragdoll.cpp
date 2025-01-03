#include <fan/pch.h>

struct pile_t {
  loco_t loco;
};

int main() {
  pile_t pile;
  fan::vec2 window_size = pile.loco.window.get_size();
  f32_t wall_thickness = 50.f;
  auto walls = fan::graphics::physics_shapes::create_stroked_rectangle(
    window_size / 2, window_size / 2, wall_thickness, fan::color::hex(0x6e8d6eff),
    {
      fan::physics::shape_properties_t{.friction=1.0}, 
      fan::physics::shape_properties_t{.friction=1.0}, 
      fan::physics::shape_properties_t{.friction=1.0}, 
      fan::physics::shape_properties_t{.friction=1.0}
    }
  );

  fan::graphics::physics_shapes::rectangle_t platforms[2];
  platforms[0] = fan::graphics::physics_shapes::rectangle_t{ {
    .position = fan::vec2(window_size.x / 5, window_size.y / 1.5),
    .size = fan::vec2(wall_thickness * 4, wall_thickness / 4),
    .color = fan::color::hex(0x30a6b6ff),
    .outline_color = fan::color::hex(0x30a6b6ff) * 2,
    .body_type = fan::physics::body_type_e::static_body,
    .shape_properties{.enable_presolve_events = false},
  } };
  platforms[1] = fan::graphics::physics_shapes::rectangle_t{ {
    .position = fan::vec2(500, 500),
    .size = wall_thickness / 4,
    .color = fan::color::hex(0x30a6b6ff),
    .outline_color = fan::color::hex(0x30a6b6ff) * 2,
    .body_type = fan::physics::body_type_e::static_body,
    .shape_properties{},
  } };

  loco_t::image_t character_images[fan::graphics::bone_e::bone_count];
  loco_t::image_load_properties_t lp;
  lp.visual_output = loco_t::image_sampler_address_mode::repeat;
  character_images[fan::graphics::bone_e::head] = pile.loco.image_load("characters/head.webp", lp);
  character_images[fan::graphics::bone_e::torso] = pile.loco.image_load("characters/torso.webp", lp);
  character_images[fan::graphics::bone_e::hip] = pile.loco.image_load("characters/hip.webp", lp);
  character_images[fan::graphics::bone_e::upper_left_leg ] = pile.loco.image_load("characters/upper_left_leg.webp", lp);
  character_images[fan::graphics::bone_e::lower_left_leg ] = pile.loco.image_load("characters/lower_left_leg.webp", lp);
  character_images[fan::graphics::bone_e::upper_right_leg] = pile.loco.image_load("characters/upper_right_leg.webp", lp);
  character_images[fan::graphics::bone_e::lower_right_leg] = pile.loco.image_load("characters/lower_right_leg.webp", lp);
  character_images[fan::graphics::bone_e::upper_left_arm ] = pile.loco.image_load("characters/upper_left_arm.webp", lp);
  character_images[fan::graphics::bone_e::lower_left_arm ] = pile.loco.image_load("characters/lower_left_arm.webp", lp);
  character_images[fan::graphics::bone_e::upper_right_arm] = pile.loco.image_load("characters/upper_right_arm.webp", lp);
  character_images[fan::graphics::bone_e::lower_right_arm] = pile.loco.image_load("characters/lower_right_arm.webp", lp);

  fan::graphics::human_t human({ 800.f, 500.0f }, 150.f, std::to_array(character_images));


  fan::graphics::character2d_t character;
  character.body_id = human.bones[fan::graphics::bone_e::torso].visual;
  character.force = 200000.f * human.scale;
  character.impulse = character.force / 3;
  character.max_speed = character.max_speed / (character.max_speed  / human.scale);
  character.feet[0] = human.bones[fan::graphics::bone_e::lower_left_leg].visual;
  character.feet[1] = human.bones[fan::graphics::bone_e::lower_right_leg].visual;

  fan::graphics::mouse_joint_t mouse_joint(human.bones[fan::graphics::bone_e::torso].visual);

  pile.loco.loop([&] {

    mouse_joint.update_mouse(pile.loco.physics_context.world_id, pile.loco.get_mouse_position());

    human.animate_walk(character.walk_force, pile.loco.delta_time);
    human.animate_jump(character.impulse, pile.loco.delta_time, character.jumping);

    character.process_movement();

    pile.loco.physics_context.step(pile.loco.delta_time);
    if (platforms[0].get_position().x < window_size.x / 4) {
      b2Body_SetLinearVelocity(platforms[0], { 200, 0 });
    }
    else if (platforms[0].get_position().x > window_size.x / 1.5) {
      b2Body_SetLinearVelocity(platforms[0], { -200, 0 });
    }
  });
}