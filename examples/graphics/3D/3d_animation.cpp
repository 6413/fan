#include <fan/pch.h>
static f32_t bone_render_size = 0.05;
static f32_t counter = 0;
std::vector<loco_t::shape_t> debug_rects;
std::vector<loco_t::shape_t> skeleton_lines;
fan::vec3 position = 0;
fan::vec3 rotation = fan::vec3(fan::math::pi, 0, 0);
f32_t all_scale = -0.5;
fan::vec3 scale = all_scale;
fan::mat4 m = fan::mat4(1).translate(position) * fan::mat4(1).rotate(rotation) * fan::mat4(1).scale(scale);

std::map<std::string, std::string> model_converted_bone_names;
std::map<std::string, std::string> anim_converted_bone_names;


void add_debug_rect(const fan::vec3& position, const fan::color& c) {
  loco_t::rectangle3d_t::properties_t rp;
  rp.position = position;
  rp.color = c;
  rp.size = bone_render_size;
  debug_rects.push_back(rp);
}
#include <fan/graphics/opengl/3D/objects/model.h>

fan::graphics::model_t* model = nullptr;
fan::graphics::model_t* anim0 = nullptr;

int main() {
  loco_t loco;
  loco.clear_color = 0;
  loco.set_vsync(0);
  fan::graphics::model_t::properties_t p;
#if !1
  p.path = "models/xbot_idle.fbx";
#else
  p.path = "models/final_provence.fbx";
#endif
  p.texture_path = "models/textures";
  p.use_cpu = 0;
  model = new fan::graphics::model_t(p);
 // model->export_animation("mixamo.com", "anim0.gltf");
  //return 0;
  p.path = "anim0.gltf";
  //anim0.fk_calculate_poses();
  //
  anim0 = new fan::graphics::model_t(p);
  model->import_animation(*anim0, "Idle2");
  //model->animation_list["Idle2"].type = fan_3d::model::animation_data_t::type_e::nonlinear_animation;
  model->animation_list["Idle2"].weight = 1;
  model->active_anim = "Idle2";
  //model->play_animation = true;
  
  
 // loco.console.commands.call("show_fps 1");

    //auto anid = model->create_an("an_name", 0.5);
    ////auto anid2 = bcol_model->create_an("an_name2", 0.5);

    //auto animation_node_id1 = model->fk_set_rot(anid, "Armature_Chest", 0.001/* time in seconds */,
    //  fan::vec3(1, 0, 0), 0
    //);
    //auto animation_node_id = model->fk_set_rot(anid, "Armature_Chest", 0.3/* time in seconds */,
    //  fan::vec3(1, 0, 0), fan::math::pi / 2
    //);

    //auto animation_node_id3 = model->fk_set_rot(anid, "Armature_Chest", 0.6/* time in seconds */,
    //  fan::vec3(1, 0, 0), -fan::math::pi / 3
    //);

    //auto animation_node_id2 = model->fk_set_rot(anid, "Armature_Upper_Leg_L", 0.6/* time in seconds */,
    //  fan::vec3(1, 0, 0), -fan::math::pi
    //);

  //auto anid = model->create_an("an_name", 1.0f, 2.f);
  //auto animation_node_id = model->fk_set_rot(anid, "Left_leg", 0.001/* time in seconds */,
  //  fan::vec3(1, 0, 0), 0
  //);
  //auto animation_node_id3 = model->fk_set_rot(anid, "Left_leg", 1/* time in seconds */,
  //  fan::vec3(1, 0, 0), fan::math::pi
  //);

  std::vector<loco_t::shape_t> joint_cubes;
  
  fan::vec2 window_size = gloco->window.get_size();

 // model->UpdateBoneRotation("Left_leg", -90.f, 0, 0);


  static constexpr auto axis_count = 3;
  loco_t::shape_t joint_controls[axis_count];

  bool draw_lines = 0;
  bool draw_bones = 0;

  fan::graphics::file_save_dialog_t save_file_dialog;
  fan::graphics::file_open_dialog_t open_file_dialog;
  std::string fn;

  model->light_intensity = 1;

  // for top left text
  f32_t menu_height = 0;

  std::unordered_map<std::string, std::vector<fan::quat>> original_bone_rotations;
  uint32_t anim_index = -1;
  {
    for (unsigned int i = 0; i < model->scene->mNumAnimations; ++i) {
      std::string str = std::string(model->scene->mAnimations[i]->mName.C_Str()).c_str();
      if (str == "Idle2") {
        anim_index = i;
        break;
      }
    }
    if (anim_index == -1) {
      fan::throw_error("anim not found");
    }
    auto* anim = model->scene->mAnimations[anim_index];
    for (uint32_t i = 0; i < anim->mNumChannels; ++i) {
      aiNodeAnim* channel = anim->mChannels[i];
      for (unsigned int j = 0; j < channel->mNumRotationKeys; j++) {
        original_bone_rotations[channel->mNodeName.C_Str()].push_back(channel->mRotationKeys[j].mValue);
      }
    }
  }

  fan::vec3 src = 0;
  fan::vec3 dst = 0;

  fan::time::clock c;
  c.start(0.5e9);

  gloco->m_pre_draw.push_back([&]() {

    if (ImGui::BeginMainMenuBar()) {

      if (ImGui::BeginMenu("Animation"))
      {
        if (ImGui::MenuItem("Open model")) {
          open_file_dialog.load("gltf,fbx,glb,dae", &fn);
        }
        if (ImGui::MenuItem("Save as")) {
          save_file_dialog.save("gltf", &fn);
        }
        ImGui::EndMenu();
      }
    }
    menu_height = ImGui::GetWindowHeight();
    ImGui::EndMainMenuBar();

    if (open_file_dialog.is_finished()) {
      p.path = fn;
      delete model->scene;
      model->scene = 0;
      delete model;
      model = 0;
      delete anim0;
      anim0 = 0;
      model = new fan::graphics::model_t(p);
      p.path = "anim0.gltf";
      anim0 = new fan::graphics::model_t(p);
      model->import_animation(*anim0, "Idle2");
      model->animation_list["Idle2"].weight = 1;
      model->active_anim = "Idle2";
      //model->play_animation = true;
      open_file_dialog.finished = false;
    }

    if (save_file_dialog.is_finished()) {
      if (fn.size() != 0) {
        auto ext = std::filesystem::path(fn).extension();
        if (ext != ".gltf") {
          fn += ".gltf";
        }
        // exporter will not export custom animations made, yet
        model->export_animation(model->get_active_animation().name, fn);
      }
      save_file_dialog.finished = false;
    }

    ImGui::Begin("window");

    static fan::vec3 aa = 0;
    static fan::vec3 pp = 0;
    //static fan::quat q(std::cos(fan::math::pi / 2), 0, std::sin(fan::math::pi / 2), 0);

    //static fan::quat rotation_x = fan::quat(0, 1, 0, 0);
    //static fan::quat rotation_y = fan::quat(0, 0, 1, 0);
    //static fan::quat rotation_z = fan::quat(0, 0, 0, 1);

    //ImGui::DragFloat4("bruteforce", rotation_x.data(), 0.001);
    //ImGui::DragFloat4("bruteforce1", rotation_y.data(), 0.001);
    //ImGui::DragFloat4("bruteforce2", rotation_z.data(), 0.001);

    //auto* anim = model->scene->mAnimations[anim_index];
    //for (uint32_t i = 0; i < anim->mNumChannels; ++i) {
    //  aiNodeAnim* channel = anim->mChannels[i];
    //  for (unsigned int j = 0; j < channel->mNumRotationKeys; j++) {
    //    const auto& source_bone = *anim0->bone_map[anim_converted_bone_names[channel->mNodeName.C_Str()]];
    //    auto found = model->bone_map.find(channel->mNodeName.C_Str());
    //    if (found == model->bone_map.end()) {
    //      continue;
    //    }
    //    const auto& destination_bone = *found->second;
    //    //fan::quat correctionQuat(std::cos(fan::math::pi / 2),  std::sin(fan::math::pi / 2),0, 0);
    //    fan::quat rotation = original_bone_rotations[destination_bone.name][j];
    //    //fan::quat transformedRotation = (destination_bone.transform * fan::translation_matrix(pp)) * rotation;
    //    auto& btt = model->animation_list["Idle2"].bone_transforms;
    //    auto ff = btt.find(channel->mNodeName.C_Str());
    //    if (ff == btt.end()) {
    //      fan::throw_error("AAA");
    //    }
    //    if (std::string(channel->mNodeName.C_Str()).c_str() != std::string("Left_leg")) {
    //    //  ff->second.rotations[j] = rotation;
    //    }
    //    else {
    //    }
    //    fan::quat src = source_bone.transform;
    //    fan::quat dst = fan::quat(destination_bone.transform);
    //    fan::quat anim = rotation;
    //    if (c.finished()) {
    //    //  rotation_x = fan::quat(fan::random::f32(-1, 1), fan::random::f32(-1, 1), fan::random::f32(-1, 1), fan::random::f32(-1, 1));
    //    //  fan::print(rotation_x);
    //      c.restart();
    //    }
    //    fan::quat src_conjugate = src;
    //    src_conjugate = src_conjugate.normalize();
    //    fan::quat relative_rotation = src_conjugate * anim;
    //    fan::quat updated_dst = relative_rotation * dst;
    //    updated_dst = updated_dst.normalize();
    //    rotation = rotation.normalize();
    //    rotation_x=rotation_x.normalize();
    //    rotation_z = rotation_z.normalize();
    //    fan::quat combined = rotation_z * rotation_y * rotation_x;
    //    ff->second.rotations[j] = rotation * combined;
    //    //found->second. = q * transformedRotation;
    //  }
    //}

    


    static bool toggle = model->p.use_cpu;
    if (ImGui::ToggleButton("use cpu", &toggle)) {
      model->p.use_cpu = toggle;
      if (toggle == false) {
        model->calculated_meshes = model->meshes;
      }
    }

    if (ImGui::ToggleButton("draw lines", &draw_lines)) {
      if (draw_lines) {
        gloco->opengl.glPolygonMode(fan::opengl::GL_FRONT_AND_BACK, fan::opengl::GL_LINE);
      }
      else {
        gloco->opengl.glPolygonMode(fan::opengl::GL_FRONT_AND_BACK, fan::opengl::GL_FILL);
      }
    }

    model->fk_calculate_poses();
    auto bts = model->fk_calculate_transformations();
    model->upload_modified_vertices();

    skeleton_lines.clear();
    {
      model->iterate_bones(*model->root_bone, [&](auto& bone) {
        src = (m * fan::vec4(bone.bone_transform.get_translation(), 1.0));
        if (bone.parent != nullptr) {
          dst = (m * fan::vec4(bone.parent->bone_transform.get_translation(), 1.0));
        }
        else {
          dst = src;
        }
        skeleton_lines.push_back(fan::graphics::line3d_t{ {
          .src = src,
          .dst = dst,
          .color = fan::colors::red
        }});
      });
    }


    if (ImGui::CollapsingHeader("bone settings")) {
      ImGui::Indent(40.f);
      if (ImGui::ToggleButton("draw bones", &draw_bones)) {
        joint_cubes.clear();
      }
      if (draw_bones) {
        ImGui::DragFloat("bone size", &bone_render_size, 0.01);
      }
      ImGui::Unindent(40.f);
    }
    if (draw_bones) {
      joint_cubes.clear();
      for (auto& it : model->bone_map) {
        auto& bone = it.second;
        loco_t::rectangle3d_t::properties_t rp;
        rp.size = bone_render_size;
        rp.color = fan::colors::red;
        fan::vec4 v = (m * fan::vec4(bone->bone_transform.get_translation(), 1.0));
        rp.position = *(fan::vec3*)&v;
        joint_cubes.push_back(rp);
      }
    }
    
    static bool spin = false;
    if (spin) {
      rotation.y += loco.delta_time / 3;
      rotation.y = fmod(rotation.y, fan::math::two_pi);
    }
    if (ImGui::CollapsingHeader("model settings")) {
      ImGui::ToggleButton("spin", &spin);
      ImGui::DragFloat3("position", position.data(), 0.1);
      ImGui::DragFloat3("rotation", rotation.data(), 0.01);
      ImGui::DragFloat("scale", &all_scale, 0.01);
      scale = all_scale;

      model->print_bone_recursive(model->root_bone);

      ImGui::Indent(40.f);
      fan_imgui_dragfloat1(model->light_position, 0.2);
      ImGui::ColorEdit3("model->light_color", model->light_color.data());
      fan_imgui_dragfloat1(model->light_intensity, 0.1);
      static f32_t specular_strength = 0.5;
      if (fan_imgui_dragfloat1(specular_strength, 0.01)) {
        loco.shader_set_value(model->m_shader, "specular_strength", specular_strength);
      }
      ImGui::Unindent(40.f);
    }
    m = fan::mat4(1).translate(position) * fan::mat4(1).rotate(rotation) * fan::mat4(1).scale(scale);

    model->mouse_modify_joint(loco.delta_time);
    if (ImGui::CollapsingHeader("animation settings")) {
      ImGui::Indent(40.f);
      model->display_animations();
      ImGui::Unindent(40.f);
    }


    model->draw(m, bts);

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
    if (model->toggle_rotate) {
      model->toggle_rotate = false;
    }
  });
  loco.window.add_key_callback(fan::key_r, fan::keyboard_state::press, [&](const auto&) { 
    model->toggle_rotate = !model->toggle_rotate;
    if (model->toggle_rotate) {
      model->showing_temp_rot = true;
      auto& anim = model->get_active_animation();

     /* for (int i = 0; i < anim.bone_poses.size(); ++i) {
        model->temp_rotations[i] = anim.bone_poses[i].rotation;
      }*/
    }
  });
  loco.window.add_key_callback(fan::key_escape, fan::keyboard_state::press, [&](const auto&) {
    if (model->toggle_rotate) {
      model->toggle_rotate = !model->toggle_rotate;
    }
    else {
      model->active_bone = -1;
    }
  });

  int active_axis = -1;

  loco.camera_set_position(loco.perspective_camera.camera, { -5.36, 2.93, 4.99 });
  camera.m_yaw = 154.49;
  camera.m_pitch = -19.68;
  camera.update_view();


  loco.loop([&] {
    camera.move(100);
    fan::ray3_t ray = gloco->convert_mouse_to_ray(camera.position, camera.m_projection, camera.m_view);

    if (model->toggle_rotate && model->active_bone != -1 && active_axis != -1) {
      auto& anim = model->get_active_animation();
      auto& bt = anim.bone_transforms[model->bone_strings[model->active_bone]];
      fan::vec3 axis = 0;
      f32_t angle = 0;

      if (motion.x) {
        angle += motion.x / 2.f * gloco->delta_time;

        fan::vec3 apply = angle;
        apply[(active_axis + 1) % apply.size()] = 0;
        apply[(active_axis + 2) % apply.size()] = 0;
        model->bones[model->active_bone]->rotation += apply;
      }
    }

    {// axis
      bool clicked_axis = 0;
      if (model->active_bone != -1) {
        for (int i = 0; i < std::size(joint_controls); ++i) {
          if (i != active_axis) {
            joint_controls[i].set_color(i == 0 ? fan::colors::red : i == 1 ? fan::colors::green : fan::colors::blue);
          }
          if (gloco->is_ray_intersecting_cube(ray, joint_controls[i].get_position(), joint_controls[i].get_size3())) {
            if (ImGui::IsMouseClicked(0)) {
              active_axis = i;
              joint_controls[i].set_color(fan::colors::white);
              clicked_axis = 1;
            }
          }
        }
      }
       if (ImGui::IsMouseClicked(0) && !clicked_axis) {
          active_axis = -1;
          model->active_bone = -1;
          for (int i = 0; i < std::size(joint_controls); ++i)
            joint_controls[i].erase();
        }
    }

    for (int i = 0; i < joint_cubes.size(); ++i) {
      if (gloco->is_ray_intersecting_cube(ray, joint_cubes[i].get_position(), joint_cubes[i].get_size3())) {
        joint_cubes[i].set_color(fan::colors::green);
        if (ImGui::IsMouseDown(0) && ImGui::IsAnyItemActive()) {
          model->active_bone = i;
          for (int j = 0; j < std::size(joint_controls); ++j) {
            loco_t::rectangle3d_t::properties_t rp;
            rp.size = fan::vec3(0.1, 0.5, 0.1);
            rp.color = std::to_array({ fan::colors::red, fan::colors::green, fan::colors::blue })[j];
            rp.position =
              std::to_array({
              fan::vec3(1, 0, 0),
              fan::vec3(0, 0, 1),
              fan::vec3(-1, 0, 0)
              })[j]
              + m * fan::vec4(model->bones[model->active_bone]->bone_transform.get_translation(), 1.0);
            joint_controls[j] = rp;
          }
        }
      }
      else {
        if (model->active_bone == i) {
          joint_cubes[i].set_color(fan::colors::green);
        }
        else {
          joint_cubes[i].set_color(fan::colors::red);
        }
      }
    }
    counter++;

    fan::graphics::text(
      camera.position.to_string() + " " +
      std::to_string(camera.get_yaw()) + " " +
      std::to_string(camera.get_pitch()),
      fan::vec2(0, menu_height)
    );


    motion = 0;
  });
}