#include fan_pch

int main() {
  loco_t loco;

  loco.set_vsync(0);

  fan::graphics::model_t::properties_t p;
  p.path = "models/sponza6.fbx";
  // sponza model has different coordinate system so fix it by rotating model matrix
  //p.model = fan::mat4(1).rotate(fan::math::pi / 2, fan::vec3(1, 0, 0));
  p.model = p.model.scale(0.01);
  fan::graphics::model_t model(p);

  gloco->default_camera_3d->camera.position = { 3.46, 1.94, -6.22 };

  auto& opengl = gloco->get_context().opengl;

  opengl.glGenTextures(1, &model.envMapTexture);
  opengl.glBindTexture(fan::opengl::GL_TEXTURE_CUBE_MAP, model.envMapTexture);

  fan::vec2 window_size = gloco->get_window()->get_size();


  opengl.glTexParameteri(fan::opengl::GL_TEXTURE_CUBE_MAP, fan::opengl::GL_TEXTURE_MIN_FILTER, fan::opengl::GL_LINEAR);
  opengl.glTexParameteri(fan::opengl::GL_TEXTURE_CUBE_MAP, fan::opengl::GL_TEXTURE_MAG_FILTER, fan::opengl::GL_LINEAR);
  opengl.glTexParameteri(fan::opengl::GL_TEXTURE_CUBE_MAP, fan::opengl::GL_TEXTURE_WRAP_S, fan::opengl::GL_CLAMP_TO_EDGE);
  opengl.glTexParameteri(fan::opengl::GL_TEXTURE_CUBE_MAP, fan::opengl::GL_TEXTURE_WRAP_T, fan::opengl::GL_CLAMP_TO_EDGE);
  opengl.glTexParameteri(fan::opengl::GL_TEXTURE_CUBE_MAP, fan::opengl::GL_TEXTURE_WRAP_R, fan::opengl::GL_CLAMP_TO_EDGE);

  for (fan::opengl::GLuint i = 0; i < 6; ++i) {
    fan::webp::image_info_t image_info;
    if (fan::webp::load(("images/" + std::to_string(i) + ".webp"), &image_info)) {
      fan::throw_error("a");
    }
    opengl.glTexImage2D(fan::opengl::GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, fan::opengl::GL_RGBA, image_info.size.x, image_info.size.y, 0, fan::opengl::GL_RGBA, fan::opengl::GL_UNSIGNED_BYTE, image_info.data);
    fan::webp::free_image(image_info.data);
  }

  gloco->m_post_draw.push_back([&] {
   
    auto temp_view = gloco->default_camera_3d->camera.m_view;
    model.draw();
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

  loco.window.add_key_callback(fan::key_r, fan::keyboard_state::press, [&](const auto&) {
    fan::string str;
    fan::io::file::read("1.glsl", &str);
    model.m_shader.set_vertex(model.vertex_shaders[fan::graphics::model_t::use_flag_e::model]);
    model.m_shader.set_fragment(str.c_str());
    model.m_shader.compile();
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