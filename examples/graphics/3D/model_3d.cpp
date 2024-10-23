#include <fan/pch.h>

int main() {
  loco_t loco;
  loco.set_vsync(0);

  static constexpr uint8_t use_flag = fan::graphics::model_t::use_flag_e::model;

  fan::graphics::model_t::properties_t p;
  p.path = "models/sponza6.fbx";
  // sponza model has different coordinate system so fix it by rotating model matrix
  p.model = fan::mat4(1).rotate(fan::math::pi, fan::vec3(1, 0, 0));
  p.model = p.model.scale(0.01);
  fan::graphics::model_t model(p);

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
  m = m.scale(0.01);

  loco.loop([&] {
    ImGui::Begin("window");

    camera.move(100);

    model.render_objects[0].m = m;

    ImGui::DragFloat4("0", (float*)&m[0], 0.01);
    ImGui::DragFloat4("1", (float*)&m[1], 0.01);
    ImGui::DragFloat4("2", (float*)&m[2], 0.01);
    ImGui::DragFloat4("3", (float*)&m[3], 0.01);

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