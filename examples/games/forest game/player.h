struct player_t {
  fan::vec2 velocity = 0;
  std::array<fan::graphics::image_t, 4> img_idle;
  std::array<std::array<fan::graphics::image_t, 4>, std::size(fan::movement_e::_strings)> img_movement;

  player_t() {
    pile.player.body.set_draw_offset(fan::vec2(0, pile.player.body.get_size().y / 1.5f));
    fan::graphics::image_load_properties_t lp;
    lp.min_filter = fan::graphics::image_filter::nearest;
    lp.mag_filter = fan::graphics::image_filter::nearest;
    for (std::size_t i = 0; i < std::size(img_idle); ++i) {
      img_idle[i] = pile.loco.image_load("npc/player/static_"_str + fan::movement_e::_strings[i] + ".png", lp);
    }
    static auto load_movement_images = [](std::array<fan::graphics::image_t, 4>& images, const std::string& direction) {
      const std::array<std::string, 4> pose_variants = {
          direction + "_left_hand_forward.png",
          "static_" + direction + ".png",
          direction + "_right_hand_forward.png",
          "static_" + direction + ".png"
      };

      fan::graphics::image_load_properties_t lp;
      lp.min_filter = fan::graphics::image_filter::nearest;
      lp.mag_filter = fan::graphics::image_filter::nearest;
      for (const auto& [i, pose] : fan::enumerate(pose_variants)) {
        images[i] = (pile.loco.image_load("npc/player/"_str + pose, lp));
      }
    };

    load_movement_images(img_movement[fan::movement_e::left], "left");
    load_movement_images(img_movement[fan::movement_e::right], "right");
    load_movement_images(img_movement[fan::movement_e::up], "up");
    load_movement_images(img_movement[fan::movement_e::down], "down");

    body.set_image(img_idle[fan::movement_e::down]);

    light = fan::graphics::light_t{ {
      .position = body.get_position(),
      .size = 200,
      .color = fan::colors::white / 4.f
    } };
  }

  void step() {
    using namespace fan::graphics;

    //animator.process_walk(
    //  body,
    //  body.get_linear_velocity(),
    //  img_idle, img_movement[fan::movement_e::left], img_movement[fan::movement_e::right],
    //  img_movement[fan::movement_e::up], img_movement[fan::movement_e::down]
    //);
    light.set_position(body.get_position());
    //fan::vec2 dir = animator.prev_dir;
    uint32_t flag = 0;
    //body updates

    // map renderer & camera update
    gloco()->camera_move_to_smooth(body);
    fan::vec2 position = body.get_position();
    //ImGui::Begin("A");
    //ImGui::DragFloat("z", &z, 1);
    ///ImGui::End();
    body.set_position(fan::vec3(position, fan::graphics::get_depth_from_y(position, 64.f))); /*hardcoded tile_size*/

    if (pile.is_map_changing) {
      return;
    }

  }

  fan::graphics::physics::character2d_t body{ fan::graphics::physics::circle_sprite_t{{
    .position = fan::vec3(1019.59076, 500.f, 10),
    // collision radius
    .radius = 4,
    // image size
    .size = fan::vec2(8, 16),
    /*.color = fan::color::from_rgba(0x715a5eff),*/
    .blending = true,
    .body_type = fan::physics::body_type_e::dynamic_body,
    .mass_data{.mass = 0.01f},
    .shape_properties{
      .friction = 0.6f,
      .density = 0.1f,
      .fixed_rotation = true,
      .linear_damping = 30,
      .collision_multiplier = fan::vec2(1, 1)
    },
  }} };
  fan::graphics::shape_t light;
};
