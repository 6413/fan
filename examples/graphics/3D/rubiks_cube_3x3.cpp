#include fan_pch

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>

int main() {
  //fan::window_t::set_flag_value<fan::window_t::flags::no_mouse>(true);

  loco_t loco;
  //loco.window.lock_cursor_and_set_invisible(true);


  loco.set_vsync(0);

  static constexpr bool use_gpu = true;
  fan_3d::model::animator_t model("models/rubiks_cube.fbx", use_gpu);


  gloco->default_camera_3d->camera.position = { 3.46, 1.94, -6.22 };
  //fan_3d::graphics::add_camera_rotation_callback(&camera);

  fan::time::clock timer;
  timer.start();

  fan::vec2 window_size = gloco->get_window()->get_size();

  f32_t angle = 0;

  //fan::print(model.render_objects.size());

  auto rot_front = [&] {
    auto default_animation_transform = model.fms.calculate_transformations();

    };

  auto get_mesh_beg_id = [](int start) {
    int overall_index = 0;
    for (int c = 0; c < start; ++c) {
      overall_index += fan_3d::model::mesh_id_table[overall_index];
    }
    return overall_index;
    };

  auto rotate_cube = [&](int start, int end, fan::mat4 q) {
    auto default_animation_transform = model.fms.calculate_transformations();
    int start_idx = get_mesh_beg_id(start);
    int idx = start_idx;
    for (int c = start; c < end; ++c) {
      for (int i = 0; i < fan_3d::model::mesh_id_table[start_idx]; ++i) {
        model.render_objects[idx].m = q;
        if constexpr (use_gpu == false) {
          model.fms.calculate_modified_vertices(idx, model.render_objects[idx].m, default_animation_transform);
          model.upload_modified_vertices(idx);
        }
        idx += 1;
      }
      start_idx += fan_3d::model::mesh_id_table[start_idx];
    }
    };

  struct piece_t {
    fan::vec3ui index;
  };

  struct rotation_t {
    std::array<piece_t, 9> arr;
    fan::vec3 axis;
  };


  std::array<uint32_t, 3 * 3 * 3> cube_ids;
  {
    int index = 0;
    for (auto& i : cube_ids) {
      i = index++;
    }
  }


  enum rotation_e {
    U_ROTATION,
    F_ROTATION,
    B_ROTATION,
    D_ROTATION,
    R_ROTATION,
    L_ROTATION,
    NUM_ROTATIONS
  };

  // order for everything is the order in enum

  static constexpr fan::vec3 axis[]{
    {0.00, -1.00, 0.00},
    {0.00, 0.00, 1.00},
    {0.00, 0.00, 1.00},
    {0.00, 1.00, 0.00},
    {-1.00, 0.00, 0.00},
    {-1.00, 0.00, 0.00}
  };
  static constexpr uint32_t rotations[] = {
    18, 19, 20, 21, 22, 23, 24, 25, 26,
    0, 1, 2, 9, 10, 11, 18, 19, 20,
    6, 7, 8, 15, 16, 17, 24, 25, 26,
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    2, 11, 20, 5, 14, 23, 8, 17, 26,
    0, 9, 18, 3, 12, 21, 6, 15, 24
  };

  // for 3x3 - 12 changes per face (could be optimized)
  static constexpr uint32_t affected_to[]{
    18, 19, 20, 24, 21, 18, 26, 25, 24, 20, 23, 26,
    0, 1, 2, 0, 9, 18, 18, 19, 20, 2, 11, 20,
    6, 7, 8, 6, 15, 24, 26, 25, 24, 8, 17, 26,
    0, 1, 2, 2, 5, 8, 6, 7, 8, 0, 3, 6,
    2, 11, 20, 20, 23, 26, 8, 17, 26, 2, 5, 8,
    0, 9, 18, 18, 21, 24, 6, 15, 24, 0, 3, 6
  };

  // for 3x3 - 12 changes per face (could be optimized)
  static constexpr uint32_t affected_from[]{
    20, 23, 26, 18, 19, 20, 24, 21, 18, 26, 25, 24,
    2, 11, 20, 2, 1, 0, 0, 9, 18, 20, 19, 18,
    8, 17, 26, 8, 7, 6, 24, 15, 6, 26, 25, 24,
    6, 3, 0, 0, 1, 2, 8, 5, 2, 6, 7, 8,
    8, 5, 2, 2, 11, 20, 26, 23, 20, 8, 17, 26,
    6, 3, 0, 0, 9, 18, 24, 21, 18, 6, 15, 24
  };

  glm::mat4 models[3 * 3 * 3];
  std::fill(models, models + std::size(models), glm::mat4(1));
  if constexpr (use_gpu) {
    for (auto& i : models) {
      i = glm::scale(i, glm::vec3(0.01));
    }
    fan::mat4 m;
    for (int k = 0; k < model.render_objects.size(); ++k) {
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          model.render_objects[k].m[i][j] = models[k / 3][i][j];
        }
      }
    }
  }

  auto update_indices = [&](rotation_e rot) {
    auto copy = cube_ids;
    static constexpr uint32_t hardcoded_rotation_affection_count = 12;
    for (int i = 0; i < hardcoded_rotation_affection_count; ++i) {
      cube_ids[affected_to[(int)rot * 12 + i]] = copy[affected_from[(int)rot * 12 + i]];
    }
  };

  auto perform_rotations = [&](rotation_e rot, f32_t angle) {
    for (int i = 0; i < 9; ++i) {
      uint32_t index = cube_ids[rotations[rot * 9 + i]];
      glm::mat4& model = models[index];
      model = glm::rotate(model, angle, glm::vec3(axis[rot].x, axis[rot].y, axis[rot].z));
      fan::mat4 m;
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          m[j][i] = model[i][j];
        }
      }
      rotate_cube(index, index + 1, m);
    }
  };

  int side = F_ROTATION;

  f32_t prev_angle = 0;

  f32_t total_rot = 0;
  bool pause = true;

  uint64_t rotation_count = 0;
  int prev_rot = 0;



  gloco->m_post_draw.push_back([&] {
    ImGui::Checkbox("pause", &pause);

    static bool shuffle = true;
    ImGui::Checkbox("shuffle", &shuffle);

    do {
      f32_t rot_count = (angle - prev_angle);
      if (total_rot + rot_count > fan::math::pi / 2) {
        rot_count = fan::math::pi / 2 - total_rot;
        perform_rotations((rotation_e)side, rot_count);
        update_indices((rotation_e)side);
        prev_rot = side;
        side = fan::random::value_i64(0, 5);
        //while(prev_rot == side)
        //  side = fan::random::value_i64(0, 1);
        prev_angle = 0;
        angle = 0;
        total_rot = 0;
        int index = 0;
        int fail_count = 0;
        if (shuffle == false) {
          for (int i = 0; i < 1; ++i) {
            for (int j = 0; j < 1; ++j) {
              for (int k = 0; k < 3; ++k) {
                if (cube_ids[i * 9 + j * 3 + k] != index) {
                  goto gt_skip;
                }
                else {
                  glm::vec3 scale;
                  glm::quat rotation;
                  glm::vec3 translation;
                  glm::vec3 skew;
                  glm::vec4 perspective;
                  glm::decompose(models[cube_ids[i * 9 + j * 3 + k]], scale, rotation, translation, skew, perspective);
                  rotation = glm::conjugate(rotation);
                  if ((round(rotation.w) == 1 && round(rotation.x) == 0 && round(rotation.y) == 0 && round(rotation.z) == 0) == false) {
                    goto gt_skip;
                  }
                }
                index++;
              }
            }
          }
          pause = true;
        gt_skip:;
        }
        ++rotation_count;
        break;
      }
      perform_rotations((rotation_e)side, rot_count);
      total_rot += (angle - prev_angle);
    } while (0);


    ImGui::Text(fan::format("rotations:{}", rotation_count).c_str());
    static bool full_speed = false;
    ImGui::Checkbox("full speed", &full_speed);

    if (full_speed && !pause) {
      angle += fan::math::pi / 2;
      ++rotation_count;
    }
    else {
      static f32_t speed = 1;
      ImGui::DragFloat("speed", &speed);

      prev_angle = angle;
      if (!pause)
        angle += gloco->delta_time * speed;
    }

    auto temp_view = gloco->default_camera_3d->camera.m_view;
    model.draw(0);
    ImGui::End();
    });


  auto& camera = gloco->default_camera_3d->camera;

  fan::vec2 motion = 0;
  loco.window.add_mouse_motion([&](const auto& d) {
    motion = d.motion;
    if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
      camera.rotate_camera(d.motion);
    }
    });

  fan::string str;
  static constexpr const char* shader_path = _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/effects/loco_post_fbo.fs);
  fan::io::file::read(shader_path, &str);
  str.resize(4096);
  loco.window.add_key_callback(fan::key_r, fan::keyboard_state::press, [&](const auto&) {
    
    gloco->m_fbo_post_gui_shader.set_vertex(loco_t::read_shader(_FAN_PATH_QUOTE(graphics/glsl/opengl/2D/effects/loco_fbo.vs)));
    gloco->m_fbo_post_gui_shader.set_fragment(str.c_str());
    gloco->m_fbo_post_gui_shader.compile();
  });

  int active_axis = -1;

  int render_time = 0;

  loco.loop([&] {

    if (ImGui::InputTextMultiline("##TextFileContents", str.data(), str.size(), ImVec2(-1.0f, -1.0f), ImGuiInputTextFlags_AllowTabInput | ImGuiInputTextFlags_AutoSelectAll)) {
      fan::io::file::write(shader_path, str.c_str(), std::ios_base::binary);
    }

    model.fms.x += gloco->delta_time;
    ImGui::Begin("window");

    fan::ray3_t ray = gloco->convert_mouse_to_ray(camera.position, camera.m_projection, camera.m_view);

    if (!ImGui::IsAnyItemActive()) {
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
      camera.move(100);
    }

    loco.get_fps();
    motion = 0;
    });
}