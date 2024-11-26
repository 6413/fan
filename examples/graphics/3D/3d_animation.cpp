#include <fan/pch.h>
#include <fan/graphics/opengl/3D/objects/model.h>

int main() {
  loco_t loco;
  loco.set_vsync(0);
  fan::graphics::model_t::properties_t p;
#if 1
  p.path = "models/player.gltf";
#else
  p.path = "models/X Bot.fbx";
#endif
  p.texture_path = "models/textures";
  p.use_cpu = 0;
  fan::graphics::model_t model(p);
  //model.export_animation("Idle", "anim0.gltf");
  p.path = "anim0.gltf";
  fan::graphics::model_t anim0(p);
  //anim0.fk_calculate_poses();
  //
  model.import_animation(anim0, "Idle2");

  loco.console.commands.call("show_fps 1");

    //auto anid = model.create_an("an_name", 0.5);
    ////auto anid2 = bcol_model.create_an("an_name2", 0.5);

    //auto animation_node_id1 = model.fk_set_rot(anid, "Armature_Chest", 0.001/* time in seconds */,
    //  fan::vec3(1, 0, 0), 0
    //);
    //auto animation_node_id = model.fk_set_rot(anid, "Armature_Chest", 0.3/* time in seconds */,
    //  fan::vec3(1, 0, 0), fan::math::pi / 2
    //);

    //auto animation_node_id3 = model.fk_set_rot(anid, "Armature_Chest", 0.6/* time in seconds */,
    //  fan::vec3(1, 0, 0), -fan::math::pi / 3
    //);

    //auto animation_node_id2 = model.fk_set_rot(anid, "Armature_Upper_Leg_L", 0.6/* time in seconds */,
    //  fan::vec3(1, 0, 0), -fan::math::pi
    //);

  //auto anid = model.create_an("an_name", 1.0f, 2.f);
  //auto animation_node_id = model.fk_set_rot(anid, "Left_leg", 0.001/* time in seconds */,
  //  fan::vec3(1, 0, 0), 0
  //);
  //auto animation_node_id3 = model.fk_set_rot(anid, "Left_leg", 1/* time in seconds */,
  //  fan::vec3(1, 0, 0), fan::math::pi
  //);

  std::vector<loco_t::shape_t> joint_cubes;

  gloco->camera_set_position(gloco->perspective_camera.camera, { 3.46, 1.94, -6.22 });
  
  fan::vec2 window_size = gloco->window.get_size();

 // model.UpdateBoneRotation("Left_leg", -90.f, 0, 0);

  bool draw_lines = 0;
  bool draw_bones = 1;

  
  gloco->m_pre_draw.push_back([&] {
    ImGui::Begin("window");

    ImGui::Text("use cpu");
    ImGui::SameLine();
    static bool toggle = model.p.use_cpu;
    if (ImGui::ToggleButton("##use cpu", &toggle)) {
      model.p.use_cpu = toggle;
      if (toggle == false) {
        model.calculated_meshes = model.meshes;
      }
    }

    ImGui::Text("draw lines");
    ImGui::SameLine();
    if (ImGui::ToggleButton("##draw lines", &draw_lines)) {
      if (draw_lines) {
        gloco->opengl.glPolygonMode(fan::opengl::GL_FRONT_AND_BACK, fan::opengl::GL_LINE);
      }
      else {
        gloco->opengl.glPolygonMode(fan::opengl::GL_FRONT_AND_BACK, fan::opengl::GL_FILL);
      }
    }

    fan::mat4 m = fan::mat4(1).scale(0.05);
    model.fk_calculate_poses();
    auto bts = model.fk_calculate_transformations();
    model.upload_modified_vertices();

    ImGui::Text("draw bones");
    ImGui::SameLine();
    if (ImGui::ToggleButton("##draw bones", &draw_bones)) {
      joint_cubes.clear();
    }
    if (draw_bones) {
      joint_cubes.clear();
      for (auto& bone : model.bones) {
        if (model.get_active_animation_id() == -1) {
          break;
        }
        const auto& bt = model.get_active_animation().bone_transforms;
        if (!bt.empty() && bt.find(bone->name) == bt.end()) {
          break;
        }
        loco_t::rectangle3d_t::properties_t rp;
        rp.size = 0.01;
        rp.color = fan::colors::red;
        fan::vec4 v = (m * fan::vec4(bone->global_transform.get_translation(), 1.0));
        rp.position = *(fan::vec3*)&v;
        joint_cubes.push_back(rp);
      }
    }
    //model.print_bone_recursive(model.root_bone);
    fan_imgui_dragfloat1(model.light_position, 0.2);
    ImGui::ColorEdit3("model.light_color", model.light_color.data());
    fan_imgui_dragfloat1(model.light_intensity, 0.1);

    model.print_bone_recursive(model.root_bone);
    model.mouse_modify_joint();

    model.display_animations();

    model.draw(m, bts);

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
    if (model.toggle_rotate) {
      model.toggle_rotate = false;
    }
  });
  loco.window.add_key_callback(fan::key_r, fan::keyboard_state::press, [&](const auto&) { 
    model.toggle_rotate = !model.toggle_rotate;
    if (model.toggle_rotate) {
      model.showing_temp_rot = true;
      auto& anim = model.get_active_animation();

     /* for (int i = 0; i < anim.bone_poses.size(); ++i) {
        model.temp_rotations[i] = anim.bone_poses[i].rotation;
      }*/
    }
  });
  loco.window.add_key_callback(fan::key_escape, fan::keyboard_state::press, [&](const auto&) {
   /* if (model.toggle_rotate) {
      model.toggle_rotate = !model.toggle_rotate;
    }
    else {
      model.active_joint = -1;
      std::fill(std::begin(model.joint_controls), std::end(model.joint_controls), loco_t::shape_t());
    }*/
  });

  int active_axis = -1;

  loco.loop([&] {
    camera.move(100);
    fan::ray3_t ray = gloco->convert_mouse_to_ray(camera.position, camera.m_projection, camera.m_view);

    fan::graphics::text(
      camera.position.to_string() + " " +
      std::to_string(camera.get_yaw()) + " " +
      std::to_string(camera.get_pitch())
    );

    motion = 0;
  });
}