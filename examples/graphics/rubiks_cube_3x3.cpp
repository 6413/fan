#include fan_pch

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

int main() {
  //fan::window_t::set_flag_value<fan::window_t::flags::no_mouse>(true);

  loco_t loco;
  //loco.window.lock_cursor_and_set_invisible(true);


  loco.set_vsync(0);

  fan_3d::model::animator_t model("models/rubiks_cube.fbx");


  gloco->default_camera_3d->camera.position = { 3.46, 1.94, -6.22 };
  //fan_3d::graphics::add_camera_rotation_callback(&camera);

  fan::time::clock timer;
  timer.start();

  fan::vec2 window_size = gloco->get_window()->get_size();

  f32_t angle = 0;

  fan::print(model.render_objects.size());

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
        model.fms.calculate_modified_vertices(idx, model.render_objects[idx].m, default_animation_transform);

        model.upload_modified_vertices(idx);
        idx += 1;
      }
      start_idx += fan_3d::model::mesh_id_table[start_idx];
    }
    };


  struct cube_id_t {
    int index;
  };

  struct piece_t {
    fan::vec3ui index;
  };

  struct rotation_t {
    std::array<piece_t, 9> arr;
    fan::vec3 axis;
  };

  std::array<std::array<std::array<cube_id_t, 3>, 3>, 3> cube_ids;

  {
    int index = 0;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          cube_ids[i][j][k].index = index;
          index += 1;
        }
      }
    }
  }
  /*
  3, 0, 0 = 27
  2, 2, 2 = 26
  2, 2, 1 = 25
  2, 2, 0 = 24
  2, 1, 2 = 23
  2, 1, 1 = 22
  2, 1, 0 = 21
  2, 0, 2 = 20
  2, 0, 1 = 19
  2, 0, 0 = 18
  1, 2, 2 = 17
  1, 2, 1 = 16
  1, 2, 0 = 15
  1, 1, 2 = 14
  1, 1, 1 = 13
  1, 1, 0 = 12
  1, 0, 2 = 11
  1, 0, 1 = 10
  1, 0, 0 = 9
  0, 2, 2 = 8
  0, 2, 1 = 7
  0, 2, 0 = 6
  0, 1, 2 = 5
  0, 1, 1 = 4
  0, 1, 0 = 3
  0, 0, 2 = 2
  0, 0, 1 = 1
  0, 0, 0 = 0

  */
  rotation_t U;
  {
    for (int i = 0; i < 9; ++i) {
      U.arr[i].index.x = 2;
      U.arr[i].index.y = i / 3;
      U.arr[i].index.z = i % 3;
    }
    U.axis = fan::vec3(0, -1, 0);
  }
  rotation_t F;
  {
    for (int i = 0; i < 9; ++i) {
      F.arr[i].index.x = i / 3;
      F.arr[i].index.y = 0;
      F.arr[i].index.z = i % 3;
    }
    F.axis = fan::vec3(0, 0, 1);
  }

  rotation_t B;
  {
    for (int i = 0; i < 9; ++i) {
      B.arr[i].index.x = i / 3;
      B.arr[i].index.y = 2;
      B.arr[i].index.z = i % 3;
    }
    B.axis = fan::vec3(0, 0, 1);
  }

  rotation_t D;
  {
    for (int i = 0; i < 9; ++i) {
      D.arr[i].index.x = 0;
      D.arr[i].index.y = i / 3;
      D.arr[i].index.z = i % 3;
    }
    D.axis = fan::vec3(0, 1, 0);
  }

  rotation_t R;
  {
      for (int i = 0; i < 9; ++i) {
          R.arr[i].index.x = i % 3;
          R.arr[i].index.y = i / 3;
          R.arr[i].index.z = 2;
      }
      R.axis = fan::vec3(-1, 0, 0);
  }

  rotation_t L;
  {
    for (int i = 0; i < 9; ++i) {
      L.arr[i].index.x = i % 3;
      L.arr[i].index.y = i / 3;
      L.arr[i].index.z = 0;
    }
    L.axis = fan::vec3(-1, 0, 0);
  }
  enum Rotation {
    U_ROTATION,
    F_ROTATION,
    B_ROTATION,
    D_ROTATION,
    R_ROTATION,
    L_ROTATION,
    NUM_ROTATIONS
  };

  auto rotations = std::to_array({ U, F, B, D, R, L });
  std::array<std::array<std::array<fan::vec3ui, 2>, 12>, rotations.size()> affected_rotations = {
    std::array<std::array<fan::vec3ui, 2>, 12>{

      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 0, 0}, {2, 0, 2}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 0, 1}, {2, 1, 2}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 0, 2}, {2, 2, 2}},

      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 2, 0}, {2, 0, 0}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 1, 0}, {2, 0, 1}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 0, 0}, {2, 0, 2}},

      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 2, 2}, {2, 2, 0}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 2, 1}, {2, 1, 0}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 2, 0}, {2, 0, 0}},

      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 0, 2}, {2, 2, 2}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 1, 2}, {2, 2, 1}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 2, 2}, {2, 2, 0}},
    }, 
    {
      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 0, 0}, {0, 0, 2}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 0, 1}, {1, 0, 2}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 0, 2}, {2, 0, 2}},

      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 0, 0}, {0, 0, 0}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{1, 0, 0}, {0, 0, 1}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 0, 0}, {0, 0, 2}},

      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 0, 2}, {2, 0, 0}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 0, 1}, {1, 0, 0}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 0, 0}, {0, 0, 0}},

      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 0, 2}, {2, 0, 2}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{1, 0, 2}, {2, 0, 1}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 0, 2}, {2, 0, 0}},
    }, {
      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 2, 0}, {0, 2, 2}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 2, 1}, {1, 2, 2}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 2, 2}, {2, 2, 2}},

      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 2, 0}, {0, 2, 0}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{1, 2, 0}, {0, 2, 1}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 2, 0}, {0, 2, 2}},

      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 2, 2}, {2, 2, 0}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 2, 1}, {1, 2, 0}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 2, 0}, {0, 2, 0}},

      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 2, 2}, {2, 2, 2}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{1, 2, 2}, {2, 2, 1}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 2, 2}, {2, 2, 0}},
    }, {
      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 0, 0}, {0, 2, 0}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 0, 1}, {0, 1, 0}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 0, 2}, {0, 0, 0}},

      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 2, 2}, {0, 0, 2}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 1, 2}, {0, 0, 1}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 0, 2}, {0, 0, 0}},

      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 2, 0}, {0, 2, 2}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 2, 1}, {0, 1, 2}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 2, 2}, {0, 0, 2}},

      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 0, 0}, {0, 2, 0}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 1, 0}, {0, 2, 1}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 2, 0}, {0, 2, 2}},
    }, {

      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 0, 2}, {0, 0, 2}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{1, 0, 2}, {0, 1, 2}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 0, 2}, {0, 2, 2}},

      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 2, 2}, {2, 0, 2}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 1, 2}, {1, 0, 2}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 0, 2}, {0, 0, 2}},

      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 2, 2}, {2, 2, 2}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{1, 2, 2}, {2, 1, 2}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 2, 2}, {2, 0, 2}},

      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 0, 2}, {0, 2, 2}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 1, 2}, {1, 2, 2}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 2, 2}, {2, 2, 2}},
    }, {

      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 0, 0}, {0, 0, 0}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{1, 0, 0}, {0, 1, 0}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 0, 0}, {0, 2, 0}},

      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 2, 0}, {2, 0, 0}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 1, 0}, {1, 0, 0}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 0, 0}, {0, 0, 0}},

      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 2, 0}, {2, 2, 0}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{1, 2, 0}, {2, 1, 0}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{2, 2, 0}, {2, 0, 0}},

      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 0, 0}, {0, 2, 0}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 1, 0}, {1, 2, 0}},
      std::array<fan::vec3ui, 2>{fan::vec3ui{0, 2, 0}, {2, 2, 0}},
    }
  };
  glm::mat4 models[3 * 3 * 3];
  std::fill(models, models + std::size(models), glm::mat4(1));

  auto update_indices = [&] (Rotation rot){
    auto copy = cube_ids;
    for (auto& i : affected_rotations[(int)rot]) {
      cube_ids[i[0].x][i[0].y][i[0].z].index = copy[i[1].x][i[1].y][i[1].z].index;
    }
  };

  auto perform_rotations = [&](Rotation rot, f32_t angle) {

    for (auto& piece : rotations[rot].arr) {
      glm::mat4& model = models[cube_ids[piece.index.x][piece.index.y][piece.index.z].index];
      model = glm::rotate(model, angle, glm::vec3(rotations[rot].axis.x, rotations[rot].axis.y, rotations[rot].axis.z));
      fan::mat4 m;
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          m[j][i] = model[i][j];
        }
      }
      rotate_cube(cube_ids[piece.index.x][piece.index.y][piece.index.z].index, cube_ids[piece.index.x][piece.index.y][piece.index.z].index + 1, m);
    }
  };
  fan::print("\n");
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        fan::print(cube_ids[i][j][k].index);
      }
    }
  }


  
  //perform_rotations(D_ROTATION, fan::math::radians(90));
  //update_indices(D_ROTATION);
  //
  //perform_rotations(F_ROTATION, fan::math::radians(90));
  //update_indices(F_ROTATION);
  //perform_rotations(U_ROTATION, fan::math::radians(-90));
  //perform_rotations(F_ROTATION, fan::math::radians(-90));

  int side = 0;

  f32_t prev_angle = 0;

  f32_t total_rot = 0;
  bool pause = 0;

  gloco->m_post_draw.push_back([&] {

    
    do {
      f32_t rot_count = (angle - prev_angle);
      if (total_rot + rot_count > fan::math::pi / 2) {
        rot_count = fan::math::pi / 2 - total_rot;
        perform_rotations((Rotation)side, rot_count);
        update_indices((Rotation)side);
        side = fan::random::value_i64(0, rotations.size() - 1);
        prev_angle = 0;
        angle = 0;
        total_rot = 0;
        int index = 0;
        bool found = true;
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
              if (cube_ids[i][j][k].index != index) {
                found = false;
              }
              index++;
              
            }
          }
        }
        if (found) {
          pause = true;
        }
        break;
      }
      perform_rotations((Rotation)side, rot_count);
      total_rot += (angle - prev_angle);
    } while (0);

    static f32_t speed = 1;
    ImGui::DragFloat("speed", &speed);

    prev_angle = angle;
    if (!pause)
    angle += gloco->delta_time * speed;

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

  loco.window.add_key_callback(fan::key_r, fan::keyboard_state::press, [&](const auto&) {
    fan::string str;
    fan::io::file::read("1.glsl", &str);
    model.m_shader.set_vertex(model.animation_vs);
    model.m_shader.set_fragment(str.c_str());
    model.m_shader.compile();
    });

  int active_axis = -1;

  int render_time = 0;

  loco.loop([&] {

    model.fms.x += gloco->delta_time;
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

    loco.get_fps();
    motion = 0;
    });
}