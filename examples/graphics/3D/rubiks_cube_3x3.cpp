#include <fan/pch.h>

int main() {
  loco_t loco{ {.window_size = 800} };

  loco.set_vsync(0);

  static constexpr uint8_t use_flag = fan::graphics::model_t::use_flag_e::model;
  fan::graphics::model_t::properties_t p;
  p.path = "models/rubiks_cube_material.fbx";
  p.use_flag = use_flag;
  p.model = fan::mat4(1).rotate(fan::math::pi, fan::vec3(1, 0, 0));
  p.model = fan::mat4(1).rotate(-fan::math::pi / 2, fan::vec3(0, 1, 0)) * p.model;
  fan::graphics::model_t model(p);

  loco.shader_set_vertex(model.m_shader, model.vertex_shaders[use_flag]);
  loco.shader_set_fragment(model.m_shader, model.material_fs);
  loco.shader_compile(model.m_shader);

  loco.camera_set_position(gloco->perspective_camera.camera, { 3.46, 1.94, -6.22 });
  //fan_3d::graphics::add_camera_rotation_callback(&camera);

  fan::time::timer timer;
  timer.start();

  fan::vec2 window_size = gloco->window.get_size();

  f32_t angle = 0;

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
        if constexpr (use_flag == fan::graphics::model_t::use_flag_e::cpu) {
          model.fms.calculate_modified_vertices(idx, model.render_objects[idx].m, default_animation_transform);
          model.upload_modified_vertices(idx);
        }
        idx += 1;
      }
      start_idx += fan_3d::model::mesh_id_table[start_idx];
    }
    };

  std::array<uint32_t, 3 * 3 * 3> cube_ids;
  {
    int index = 0;
    for (auto& i : cube_ids) {
      i = index++;
    }
  }
  // order for everything is the order in enum

  static constexpr fan::vec3 axis[]{
    {0.00, -1.00, 0.00},
    {0.00, 0.00, 1.00},
    {0.00, 0.00, 1.00},
    {0.00, 1.00, 0.00},
    {-1.00, 0.00, 0.00},
    {-1.00, 0.00, 0.00}
  };


  enum rotation_e {
    U_ROTATION,
    F_ROTATION,
    B_ROTATION,
    D_ROTATION,
    R_ROTATION,
    L_ROTATION,
    NUM_ROTATIONS
  };


  enum {
    U0, U1, U2,
    U3, U4, U5,
    U6, U7, U8,

    F0, F1, F2,
    F3, F4, F5,
    F6, F7, F8,

    B0, B1, B2,
    B3, B4, B5,
    B6, B7, B8,

    D0, D1, D2,
    D3, D4, D5,
    D6, D7, D8,

    L0, L1, L2,
    L3, L4, L5,
    L6, L7, L8,

    R0, R1, R2,
    R3, R4, R5,
    R6, R7, R8,

    last
  };

  uint8_t cube_map[last]{};
  auto set_cubemap = [&](uint8_t value, std::initializer_list<uint8_t> keys) {
    for (uint8_t key : keys) {
      cube_map[key] = value;
    }
    };

  set_cubemap(0, { R8, R6, D8 });
  set_cubemap(1, { R7, D5 });
  set_cubemap(2, { R6, D2, F8 });
  set_cubemap(3, { R5, B3 });
  cube_map[R4] = 4;
  set_cubemap(5, { R3, F5 });
  set_cubemap(6, { R2, B0, U2 });
  set_cubemap(7, { R1, U5 });
  set_cubemap(8, { R0, U8, F2 });
  set_cubemap(9, { B7, D7 });
  cube_map[D4] = 10;
  set_cubemap(11, { D1, F7 });
  cube_map[B4] = 12;
  cube_map[F4] = 14;
  set_cubemap(15, { B1, U1 });
  cube_map[U4] = 16;
  set_cubemap(17, { U7, F1 });
  set_cubemap(18, { L6, B8, D6 });
  set_cubemap(19, { L7, D3 });
  set_cubemap(20, { L8, D0, F6 });
  set_cubemap(21, { L3, B5 });
  cube_map[L4] = 22;
  set_cubemap(23, { L5, F3 });
  set_cubemap(24, { L0, B2, U0 });
  set_cubemap(25, { L1, U3 });
  set_cubemap(26, { L2, U6, F0 });

  uint8_t rotations[9 * 6] = {
    cube_map[U0], cube_map[U1], cube_map[U2], cube_map[U3], cube_map[U4], cube_map[U5], cube_map[U6], cube_map[U7], cube_map[U8],
    cube_map[F0], cube_map[F1], cube_map[F2], cube_map[F3], cube_map[F4], cube_map[F5], cube_map[F6], cube_map[F7], cube_map[F8],
    cube_map[B0], cube_map[B1], cube_map[B2], cube_map[B3], cube_map[B4], cube_map[B5], cube_map[B6], cube_map[B7], cube_map[B8],
    cube_map[D0], cube_map[D1], cube_map[D2], cube_map[D3], cube_map[D4], cube_map[D5], cube_map[D6], cube_map[D7], cube_map[D8],
    cube_map[R0], cube_map[R1], cube_map[R2], cube_map[R3], cube_map[R4], cube_map[R5], cube_map[R6], cube_map[R7], cube_map[R8],
    cube_map[L0], cube_map[L1], cube_map[L2], cube_map[L3], cube_map[L4], cube_map[L5], cube_map[L6], cube_map[L7], cube_map[L8],
  };

  uint8_t affected_to[12 * 6]{};
  uint8_t affected_from[12 * 6]{};

  auto rotate = [&](uint8_t* affected_from, uint8_t* affected_to, uint8_t* cube_map, int& index, const uint8_t* rot_table) {
    for (int i = 0; i < 12; ++i) {
      affected_from[index + i] = cube_map[rot_table[(i + 3) % 12]];
      affected_to[index + i] = cube_map[rot_table[i % 12]];
    }
  };

  uint8_t rotation_table[]{
    L0, L1, L2, F0, F1, F2, R0, R1, R2, B0, B1, B2,
    L2, L5, L8, D0, D1, D2, R6, R3, R0, U8, U7, U6,
    R8, R5, R2, U2, U1, U0, L0, L3, L6, D6, D7, D8,
    B8, B7, B6, R8, R7, R6, F8, F7, F6, L8, L7, L6,
    D2, D5, D8, B6, B3, B0, U2, U5, U8, F2, F5, F8,
    U0, U3, U6, F0, F3, F6, D0, D3, D6, B8, B5, B2
  };

  for (int i = 0; i < std::size(rotation_table); i += 12) {
    rotate(affected_from, affected_to, cube_map, i, &rotation_table[i]);
  }


  fan::mat4 models[3 * 3 * 3];
  std::fill(models, models + std::size(models), fan::mat4(1));
  if constexpr (use_flag != fan::graphics::model_t::use_flag_e::cpu) {
    for (auto& i : models) {
      i = i.scale(fan::vec3(0.01));
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
      fan::mat4& model = models[index];
      model = model.rotate(angle, fan::vec3(axis[rot].x, axis[rot].y, axis[rot].z));
      fan::mat4 m;
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          m[j][i] = model[i][j];
        }
      }
      rotate_cube(index, index + 1, m);
    }
  };

  int side = U_ROTATION;

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
        side = fan::random::value_i64(0, std::size(rotation_table) / 12 - 1);
        prev_angle = 0;
        angle = 0;
        total_rot = 0;
        int index = 0;
        int fail_count = 0;
        if (shuffle == false) {
          for (int i = 0; i < 1; ++i) {
            for (int j = 0; j < 1; ++j) {
              for (int k = 0; k < 3; ++k) {
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


    //ImGui::Text(fan::format("rotations:{}", rotation_count).c_str());
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

    model.draw();
    ImGui::End();
  });

  auto& camera = gloco->camera_get(gloco->perspective_camera.camera);

  fan::vec2 motion = 0;
  loco.window.add_mouse_motion([&](const auto& d) {
    motion = d.motion;
    if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
      camera.rotate_camera(d.motion);
    }
    });

  f32_t global_angle = 0;

  loco.loop([&] {

    model.fms.x += gloco->delta_time;
    ImGui::Begin("window");

    fan::mat4 rot = fan::mat4(1).rotate(loco.delta_time, 1);
    //global_angle += loco.delta_time;
    model.m = model.m * rot;
    /*for (auto& i : model.render_objects) {
      i.transform = i.transform * rot;
    }*/

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