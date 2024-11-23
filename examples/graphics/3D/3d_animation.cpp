#include <fan/pch.h>

#include <fan/graphics/opengl/3D/objects/model.h>

int main() {
  loco_t loco;
  loco.set_vsync(0);
  fan::graphics::model_t::properties_t p;
  p.path = "models/testt5.fbx";
  // todo make animation work with gpu
  p.use_flag = fan::graphics::model_t::use_flag_e::model_cpu;
  fan::graphics::model_t model(p);


  //auto anid = model.fms.create_an("an_name", 1);

  //auto animation_node_id1 = model.fms.fk_set_rot(anid, "Left_leg", 0.001/* time in seconds */,
  //  fan::vec3(1, 0, 0), 0
  //);

  //auto animation_node_id = model.fms.fk_set_rot(anid, "Left_leg", 0.3/* time in seconds */,
  //  fan::vec3(1, 0, 0), -fan::math::pi / 2
  //);

  //auto animation_node_id3 = model.fms.fk_set_rot(anid, "Armature_Chest", 0.6/* time in seconds */,
  //  fan::vec3(1, 0, 0), fan::math::pi / 3
  //);

  //auto animation_node_id2 = model.fms.fk_set_rot(anid, "Armature_Upper_Leg_L", 0.6/* time in seconds */,
  //  fan::vec3(1, 0, 0), fan::math::pi
  //);

  gloco->camera_set_position(gloco->perspective_camera.camera, { 3.46, 1.94, -6.22 });
  
  fan::vec2 window_size = gloco->window.get_size();

  model.fms.UpdateBoneRotation("Left_leg", -90.f, 0, 0);

  gloco->m_post_draw.push_back([&] {
    static bool default_anim = false;
    ImGui::Checkbox("default model", &default_anim);
    //static constexpr uint32_t mesh_id = 0;

    // THIS IS FOR SIMULATING ON CPU, properties_t::use_fs = false
    
    if (p.use_flag == fan::graphics::model_t::use_flag_e::model_cpu) {

      auto fk_animation_transform = model.fms.fk_calculate_transformations();
      for (uint32_t i = 0; i < model.fms.meshes.size(); ++i) {
        model.fms.calculate_modified_vertices(i, fk_animation_transform);
      }


      for (int i = 0; i < model.fms.meshes.size(); ++i) {
        model.upload_modified_vertices(i);
      }
    }
    



//    model.mouse_modify_joint();

//    model.display_animations();

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

  int active_axis = -1;

  model.fms.dt = 0;

  loco.loop([&] {

  model.fms.dt += loco.delta_time * 10;  

    ImGui::Begin("window");
    camera.move(100);
    fan::ray3_t ray = gloco->convert_mouse_to_ray(camera.position, camera.m_projection, camera.m_view);

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