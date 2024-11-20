#include <fan/pch.h>


int main() {
  loco_t loco;
  loco.set_vsync(0);
  
  static constexpr uint8_t use_flag = fan::graphics::model_t::use_flag_e::model;

  fan::graphics::model_t::properties_t p;
  p.path = "models/NewSponza_Main_glTF_003.gltf";
  p.model = p.model.scale(0.5);
//  p.model = p.model.scale(0.05);
  // sponza model has different coordinate system so fix it by rotating model matrix
  p.model = p.model.rotate(fan::math::pi, fan::vec3(1, 0, 0));
  fan::graphics::model_t model(p);

  loco.window.add_key_callback(fan::key_r, fan::keyboard_state::press, [&](const auto&) {
    fan::string str;
    fan::io::file::read("1.glsl", &str);

    loco.shader_set_vertex(model.m_shader, model.vertex_shaders[fan::graphics::model_t::use_flag_e::model]);
    loco.shader_set_fragment(model.m_shader, str.c_str());
    loco.shader_compile(model.m_shader);
  });

  //
 // GLFWwindow* offscreen_context = create_window(1, 1, "", loco.window);
  //

  gloco->camera_set_position(gloco->perspective_camera.camera, { 3.46, 1.94, -6.22 });

  gloco->m_post_draw.push_back([&] {

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

  //loco.shader_set_vertex(model.m_shader, model.vertex_shaders[use_flag]);
  //loco.shader_set_fragment(model.m_shader, model.material_fs);
  //loco.shader_compile(model.m_shader);

  fan::mat4 m(1);



  fan::vec3 orientation = fan::vec3(fan::math::pi, 0, 0);


  loco.loop([&] {
    ImGui::Begin("window");

    model.m = fan::mat4(1).rotate(orientation);

    static fan::vec3 light_pos = 0;
    {
      auto str = gloco->camera_get_position(gloco->perspective_camera.camera).to_string();
      ImGui::Text("%s", str.c_str());
    }
    ImGui::DragFloat3("light position", light_pos.data());
    fan::vec4 lpt = fan::vec4(light_pos, 1);
    gloco->shader_set_value(model.m_shader, "light_pos", *(fan::vec3*)&lpt);


    static f32_t f0 = 0;
    ImGui::DragFloat("f0", &f0, 0.001, 0, 1);
    gloco->shader_set_value(model.m_shader, "F0", f0);


    static f32_t metallic = 0;
    ImGui::DragFloat("metallic", &metallic, 0.001, 0, 1);
    gloco->shader_set_value(model.m_shader, "metallic", metallic);

    static f32_t roughness = 0;
    ImGui::DragFloat("rough", &roughness, 0.001, 0, 1);
    gloco->shader_set_value(model.m_shader, "rough", roughness);

    static f32_t light_intensity = 1;
    ImGui::DragFloat("light_intensity", &light_intensity, 0.1);
    gloco->shader_set_value(model.m_shader, "light_intensity", light_intensity);

    camera.move(100);

    //m = fan::mat4(1).rotate(orientation);

    ImGui::DragFloat3("orientation", orientation.data(), 0.01);

    //ImGui::DragFloat4("0", (float*)&m[0], 0.01);
    //ImGui::DragFloat4("1", (float*)&m[1], 0.01);
    //ImGui::DragFloat4("2", (float*)&m[2], 0.01);
    //ImGui::DragFloat4("3", (float*)&m[3], 0.01);

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