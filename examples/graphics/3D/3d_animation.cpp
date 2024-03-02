#include fan_pch

#include <glm/glm.hpp>

int main() {
  loco_t loco;
  loco.set_vsync(0);
  glm::vec2()
  //struct triangle_list_t {
  //  uint32_t matid;
  //  std::vector<fan_3d::animation::fms_t::one_triangle_t> triangle_vec;
  //};

  fan_3d::model::animator_t animation("models/model2.dae");

  //std::vector<triangle_list_t> triangles;

  auto anid = animation.fms.create_an("an_name", 1);

  auto animation_node_id1 = animation.fms.fk_set_rot(anid, "Armature_Chest", 0.001/* time in seconds */,
    fan::vec3(1, 0, 0), 0
  );

  auto animation_node_id = animation.fms.fk_set_rot(anid, "Armature_Chest", 0.3/* time in seconds */,
    fan::vec3(1, 0, 0), fan::math::pi / 2
  );

  auto animation_node_id3 = animation.fms.fk_set_rot(anid, "Armature_Chest", 0.6/* time in seconds */,
    fan::vec3(1, 0, 0), -fan::math::pi / 3
  );

  auto animation_node_id2 = animation.fms.fk_set_rot(anid, "Armature_Upper_Leg_L", 0.6/* time in seconds */,
    fan::vec3(1, 0, 0), -fan::math::pi
  );

  gloco->default_camera_3d->camera.position = { 3.46, 1.94, -6.22 };
 
  fan::time::clock timer;
  timer.start();
  
  fan::vec2 window_size = gloco->get_window()->get_size();

  gloco->m_post_draw.push_back([&] {
    

    static bool default_anim = false;
    ImGui::Checkbox("default animation", &default_anim);
    static constexpr uint32_t mesh_id = 0;
    if (default_anim) {
      // for default animation
      animation.fms.calculate_default_pose();
      auto default_animation_transform = animation.fms.calculate_transformations();
      animation.fms.calculate_modified_vertices(mesh_id, default_animation_transform);
    }
    else {
      animation.fms.calculate_poses();
      auto fk_animation_transform = animation.fms.fk_calculate_transformations();
      animation.fms.calculate_modified_vertices(mesh_id, fk_animation_transform);
    }


    for (int i = 0; i < animation.render_objects.size(); ++i) {
      animation.upload_modified_vertices(i);
    }


    animation.mouse_modify_joint();

    animation.display_animations();

    animation.draw(0);

    ImGui::End();
  });

  auto& camera = gloco->default_camera_3d->camera;

  fan::vec2 motion = 0;
  loco.window.add_mouse_motion([&](const auto& d) {
    motion = d.motion;
    if (ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
      camera.rotate_camera(d.motion);
    }
  });

  loco.window.add_key_callback(fan::key_g, fan::keyboard_state::press, [&](const auto&) {
    fan::string str;
    fan::io::file::read("1.glsl", &str);
    animation.m_shader.set_vertex(animation.animation_vs);
    animation.m_shader.set_fragment(str.c_str());
    animation.m_shader.compile();
    });

  loco.window.add_key_callback(fan::mouse_left, fan::keyboard_state::press, [&](const auto&) { 
    if (animation.fms.toggle_rotate) {
      animation.fms.toggle_rotate = false;
    }
  });
  loco.window.add_key_callback(fan::key_r, fan::keyboard_state::press, [&](const auto&) { 
    animation.fms.toggle_rotate = !animation.fms.toggle_rotate;
    if (animation.fms.toggle_rotate) {
      animation.fms.showing_temp_rot = true;
      auto& anim = animation.fms.get_active_animation();

      for (int i = 0; i < anim.joint_poses.size(); ++i) {
        animation.fms.temp_rotations[i] = anim.joint_poses[i].rotation;
      }
    }
  });
  loco.window.add_key_callback(fan::key_escape, fan::keyboard_state::press, [&](const auto&) {
    if (animation.fms.toggle_rotate) {
      animation.fms.toggle_rotate = !animation.fms.toggle_rotate;
    }
    else {
      animation.fms.active_joint = -1;
      std::fill(std::begin(animation.joint_controls), std::end(animation.joint_controls), loco_t::shape_t());
    }
  });

  int active_axis = -1;

  int render_time = 0;

  loco.loop([&] {
    ImGui::Begin("window");
    camera.move(100);
    fan::ray3_t ray = gloco->convert_mouse_to_ray(camera.position, camera.m_projection, camera.m_view);

    if (animation.fms.toggle_rotate && animation.fms.active_joint != -1 && active_axis != -1) {
      auto& anim = animation.fms.get_active_animation();
      auto& bt = anim.bone_transforms[animation.fms.bone_strings[animation.fms.active_joint]];
      fan::vec3 axis = 0;
      f32_t angle = 0;

      if (motion.x) {

        axis[active_axis] = 1;

        angle += motion.x / 2.f * gloco->delta_time;

        fan::quat new_rotation = fan::quat::from_axis_angle(axis, angle);

        animation.fms.temp_rotations[animation.fms.active_joint] = new_rotation * animation.fms.temp_rotations[animation.fms.active_joint];
      }
    }

    if (animation.fms.active_joint != -1) {
      for (int i = 0; i < std::size(animation.joint_controls); ++i) {
        if (i != active_axis) {
          animation.joint_controls[i].set_color(i == 0 ? fan::colors::red : i == 1 ? fan::colors::green : fan::colors::blue);
        }
        if (gloco->is_ray_intersecting_cube(ray, animation.joint_controls[i].get_position(), animation.joint_controls[i].get_size())) {
          if (ImGui::IsMouseClicked(0)) {
            active_axis = i;
            animation.joint_controls[i].set_color(fan::colors::white);
          }
        }
      }
    }


    for (int i = 0; i < animation.shapes.size(); ++i) {
      if (gloco->is_ray_intersecting_cube(ray, animation.shapes[i].get_position(), animation.shapes[i].get_size())) {
        animation.shapes[i].set_color(fan::colors::green);
        if (ImGui::IsMouseDown(0) && ImGui::IsAnyItemActive()) {
          animation.set_active_joint(i);
        }
      }
      else {
        animation.shapes[i].set_color(fan::colors::red);
      }
    }
    if (ImGui::IsKeyDown(ImGuiKey_LeftArrow)) {
      camera.rotate_camera(fan::vec2(-0.01, 0));
    }
    if (ImGui::IsKeyDown(ImGuiKey_RightArrow)) {
      camera.rotate_camera(fan::vec2(0.01, 0));
    }
    if (ImGui::IsKeyDown(ImGuiKey_UpArrow)) {
      camera.rotate_camera(fan::vec2(0, -0.01));
    }
    if (ImGui::IsKeyDown(ImGuiKey_DownArrow)) {
      camera.rotate_camera(fan::vec2(0, 0.01));
    }

    loco.get_fps();
    motion = 0;
  });
}