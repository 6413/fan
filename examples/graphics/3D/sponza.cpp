#include <fan/pch.h>

int main() {
  loco_t loco;

  loco.set_vsync(0);

  fan::graphics::model_t::properties_t p;
  p.path = "models/sponza6.fbx";
  // sponza model has different coordinate system so fix it by rotating model matrix
  //p.model = fan::mat4(1).rotate(fan::math::pi / 2, fan::vec3(1, 0, 0));
  p.model = p.model.scale(0.01);
  fan::graphics::model_t model(p);


  //gloco()->perspective_camera.camera.position = { 3.46, 1.94, -6.22 };

  auto& opengl = gloco()->get_context().opengl;

  fan_opengl_call(GenTextures(1, &model.envMapTexture));
  fan_opengl_call(glBindTexture(GL_TEXTURE_CUBE_MAP, model.envMapTexture));

  fan::vec2 window_size = gloco()->window.get_size();


  fan_opengl_call(TexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
  fan_opengl_call(TexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
  fan_opengl_call(TexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
  fan_opengl_call(TexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
  fan_opengl_call(TexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE));

  for (GLuint i = 0; i < 6; ++i) {
    fan::image::info_t image_info;
    if (fan::image::load(("images/" + std::to_string(i) + ".webp"), &image_info)) {
      fan::throw_error("a");
    }
    fan_opengl_call(TexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGBA, image_info.size.x, image_info.size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_info.data));
    fan::image::free(&image_info);
  }

  gloco()->m_post_draw.push_back([&] {
   
    model.draw();
    ImGui::End();
  });

  auto& camera = gloco()->camera_get(gloco()->perspective_camera.camera);

  fan::vec2 motion = 0;
  loco.window.add_mouse_motion([&](const auto& d) {
    motion = d.motion;
    if (ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
      camera.rotate_camera(d.motion);
    }
  });

  loco.window.add_key_callback(fan::key_r, fan::keyboard_state::press, [&](const auto&) {
    fan::string str;
    fan::io::file::read("1.glsl", &str);

    loco.shader_set_vertex(model.m_shader, model.vertex_shaders[fan::graphics::model_t::use_flag_e::model]);
    loco.shader_set_fragment(model.m_shader, str.c_str());
    loco.shader_compile(model.m_shader);
    });
  int active_axis = -1;

  int render_time = 0;

  fan::vec3 orientation = fan::vec3(fan::math::pi, 0, 0);

  loco.loop([&] {
    ImGui::Begin("window");

    model.m = fan::mat4(1).rotate(orientation);

    ImGui::DragFloat3("orientation", orientation.data(), 0.01);

    camera.move(100);

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