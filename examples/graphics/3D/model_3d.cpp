#include <fan/pch.h>

GLFWwindow* create_window(int width, int height, const char* title, GLFWwindow* shared_context = nullptr) {
  GLFWwindow* window = glfwCreateWindow(width, height, title, NULL, shared_context);
  if (!window) {
    glfwTerminate();
    exit(EXIT_FAILURE);
  }
  return window;
}

int main() {
  loco_t loco;
  loco.set_vsync(0);
  
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  GLFWwindow* offscreen_context = create_window(1, 1, "", loco.window.glfw_window);
  glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);

  static constexpr uint8_t use_flag = fan::graphics::model_t::use_flag_e::model;

  fan::graphics::model_t::properties_t p;
  p.path = "models/model2.dae";
  // sponza model has different coordinate system so fix it by rotating model matrix
  //p.model = fan::mat4(1).rotate(fan::math::pi, fan::vec3(1, 0, 0));
  fan::graphics::model_t model(p);

  //
 // GLFWwindow* offscreen_context = create_window(1, 1, "", loco.window);
  //


  fan::opengl::GLuint pbo;
  loco.opengl.glGenBuffers(1, &pbo);
  loco.opengl.glBindBuffer(fan::opengl::GL_PIXEL_UNPACK_BUFFER, pbo);
  loco.opengl.glBufferData(fan::opengl::GL_PIXEL_UNPACK_BUFFER, 640 * 480 * 3, NULL, fan::opengl::GL_STREAM_DRAW);
  //model..m

  std::thread worker([&]() {
    glfwMakeContextCurrent(offscreen_context);
    loco.opengl.glBindBuffer(fan::opengl::GL_PIXEL_UNPACK_BUFFER, pbo);

    // Simulate uploading data
    unsigned char* ptr = (unsigned char*)loco.opengl.glMapBuffer(fan::opengl::GL_PIXEL_UNPACK_BUFFER, fan::opengl::GL_WRITE_ONLY);
    if (ptr) {
      for (int i = 0; i < 640 * 480 * 3; ++i) {
        ptr[i] = 255; // Example: setting all pixels to white
      }
      loco.opengl.glUnmapBuffer(fan::opengl::GL_PIXEL_UNPACK_BUFFER);
    }
    glfwMakeContextCurrent(NULL);
  });

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

    camera.move(100);

    model.render_objects[0].m = m;

    m = fan::mat4(1).rotate(orientation);

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

    loco.opengl.glBindBuffer(fan::opengl::GL_PIXEL_UNPACK_BUFFER, pbo);

    loco.get_fps();
    motion = 0;
  });
}