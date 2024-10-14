#include <fan/pch.h>

int main() {
  loco_t loco;
  loco.set_vsync(0);
  fan::graphics::model_t::properties_t p;
  p.path = "models/model2.dae";
  // todo make animation work with gpu
  p.use_flag = fan::graphics::model_t::use_flag_e::cpu;
  fan::graphics::model_t model(p);

  std::vector<loco_t::shape_t> joint_cubes;
  
  model.fms.iterate_joints(model.fms.parsed_model.model_data.skeleton, [&](const fan_3d::model::joint_t& joint) {
  /*  loco_t::shapes_t::rectangle_3d_t::properties_t rp;
    rp.size = 0.1;
    rp.color = fan::colors::red;
    rp.position = joint.global_transform.get_translation();
    rp.position = fan::vec3(rp.position.x, rp.position.z, rp.position.y);
    joint_cubes.push_back(rp);*/
  });

  auto anid = model.fms.create_an("an_name", 1);

  auto animation_node_id1 = model.fms.fk_set_rot(anid, "Armature_Chest", 0.001/* time in seconds */,
    fan::vec3(1, 0, 0), 0
  );

  auto animation_node_id = model.fms.fk_set_rot(anid, "Armature_Chest", 0.3/* time in seconds */,
    fan::vec3(1, 0, 0), -fan::math::pi / 2
  );

  auto animation_node_id3 = model.fms.fk_set_rot(anid, "Armature_Chest", 0.6/* time in seconds */,
    fan::vec3(1, 0, 0), fan::math::pi / 3
  );

  auto animation_node_id2 = model.fms.fk_set_rot(anid, "Armature_Upper_Leg_L", 0.6/* time in seconds */,
    fan::vec3(1, 0, 0), fan::math::pi
  );

  gloco->camera_set_position(gloco->perspective_camera.camera, { 3.46, 1.94, -6.22 });
  
  fan::vec2 window_size = gloco->window.get_size();

  gloco->m_post_draw.push_back([&] {
    static bool default_anim = true;
    ImGui::Checkbox("default model", &default_anim);
    static constexpr uint32_t mesh_id = 0;

    // THIS IS FOR SIMULATING ON CPU, properties_t::use_fs = false
    
    if (p.use_flag == fan::graphics::model_t::use_flag_e::cpu) {

      if (default_anim) {
        // for default model
        model.fms.calculate_default_pose();
        auto default_animation_transform = model.fms.calculate_transformations();
        model.fms.calculate_modified_vertices(mesh_id, default_animation_transform);
      }
      else {
        model.fms.calculate_poses();
        auto fk_animation_transform = model.fms.fk_calculate_transformations();
        model.fms.calculate_modified_vertices(mesh_id, fk_animation_transform);
      }


      for (int i = 0; i < model.render_objects.size(); ++i) {
        model.upload_modified_vertices(i);
      }
    }
    



    model.mouse_modify_joint();

    model.display_animations();

    model.draw();

    ImGui::End();
  });

  auto& camera = gloco->camera_get(gloco->perspective_camera.camera);

  fan::vec2 motion = 0;
  loco.window.add_mouse_motion([&](const auto& d) {
    motion = d.motion;
    if (ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
      camera.rotate_camera(d.motion);
    }
  });

  loco.window.add_key_callback(fan::mouse_left, fan::keyboard_state::press, [&](const auto&) { 
    if (model.fms.toggle_rotate) {
      model.fms.toggle_rotate = false;
    }
  });
  loco.window.add_key_callback(fan::key_r, fan::keyboard_state::press, [&](const auto&) { 
    model.fms.toggle_rotate = !model.fms.toggle_rotate;
    if (model.fms.toggle_rotate) {
      model.fms.showing_temp_rot = true;
      auto& anim = model.fms.get_active_animation();

      for (int i = 0; i < anim.joint_poses.size(); ++i) {
        model.fms.temp_rotations[i] = anim.joint_poses[i].rotation;
      }
    }
  });
  loco.window.add_key_callback(fan::key_escape, fan::keyboard_state::press, [&](const auto&) {
    if (model.fms.toggle_rotate) {
      model.fms.toggle_rotate = !model.fms.toggle_rotate;
    }
    else {
      model.fms.active_joint = -1;
      std::fill(std::begin(model.joint_controls), std::end(model.joint_controls), loco_t::shape_t());
    }
  });

  int active_axis = -1;

  loco.loop([&] {
    ImGui::Begin("window");
    camera.move(100);
    fan::ray3_t ray = gloco->convert_mouse_to_ray(camera.position, camera.m_projection, camera.m_view);

    if (model.fms.toggle_rotate && model.fms.active_joint != -1 && active_axis != -1) {
      auto& anim = model.fms.get_active_animation();
      auto& bt = anim.bone_transforms[model.fms.bone_strings[model.fms.active_joint]];
      fan::vec3 axis = 0;
      f32_t angle = 0;

      if (motion.x) {

        axis[active_axis] = 1;

        angle += motion.x / 2.f * gloco->delta_time;

        fan::quat new_rotation = fan::quat::from_axis_angle(axis, angle);

        model.fms.temp_rotations[model.fms.active_joint] = new_rotation * model.fms.temp_rotations[model.fms.active_joint];
      }
    }

    if (model.fms.active_joint != -1) {
      for (int i = 0; i < std::size(model.joint_controls); ++i) {
        if (i != active_axis) {
          model.joint_controls[i].set_color(i == 0 ? fan::colors::red : i == 1 ? fan::colors::green : fan::colors::blue);
        }
        if (gloco->is_ray_intersecting_cube(ray, model.joint_controls[i].get_position(), model.joint_controls[i].get_size())) {
          if (ImGui::IsMouseClicked(0)) {
            active_axis = i;
            model.joint_controls[i].set_color(fan::colors::white);
          }
        }
      }
    }


    for (int i = 0; i < joint_cubes.size(); ++i) {
      if (gloco->is_ray_intersecting_cube(ray, joint_cubes[i].get_position(), joint_cubes[i].get_size())) {
        joint_cubes[i].set_color(fan::colors::green);
        if (ImGui::IsMouseDown(0) && ImGui::IsAnyItemActive()) {
          model.set_active_joint(i);
        }
      }
      else {
        joint_cubes[i].set_color(fan::colors::red);
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
    fan::print(camera.position, camera.get_yaw(), camera.get_pitch());

    loco.get_fps();
    motion = 0;
  });
}