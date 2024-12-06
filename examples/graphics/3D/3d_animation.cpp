#include <fan/pch.h>
static f32_t bone_render_size = 0.05;
static f32_t counter = 0;
std::vector<loco_t::shape_t> debug_rects;
std::vector<loco_t::shape_t> skeleton_lines;
fan::vec3 position = 0;
fan::vec3 rotation = fan::vec3(fan::math::pi, 0, 0);
f32_t all_scale = -0.05;
fan::vec3 scale = all_scale;
fan::mat4 m = fan::mat4(1).translate(position) * fan::mat4(1).rotate(rotation) * fan::mat4(1).scale(scale);

#include <fan/graphics/opengl/3D/objects/model.h>

static int cursor_mode = 1;

inline struct pile_t {
  loco_t loco;
}*pile=0;


struct editor_t {

  struct flags_e {
    enum {
      hovered = 1 << 0,

    };
  };

  void begin_render() {
    static int initial = 0;
    if (initial == 0) {
      ImGui::SetNextWindowSize(fan::vec2(200, 200));
      initial = 1;
    }
    ImGui::Begin("Editor", 0, ImGuiWindowFlags_NoBackground);

    flags &= ~flags_e::hovered;
    flags |= (uint16_t)ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);

    pile->loco.set_imgui_viewport(pile->loco.perspective_camera.viewport);
  }
  void end_render() {
    ImGui::End();
  }

  bool hovered() const {
    return flags & flags_e::hovered;
  }

  uint16_t flags = 0;
}editor;

void add_debug_rect(const fan::vec3& position, const fan::color& c) {
  loco_t::rectangle3d_t::properties_t rp;
  rp.position = position;
  rp.color = c;
  rp.size = bone_render_size;
  debug_rects.push_back(rp);
}


fan::graphics::model_t* model = nullptr;
fan::graphics::model_t* anim0 = nullptr;

int main() {
  pile = new pile_t;

  pile->loco.clear_color = 0;
  pile->loco.set_vsync(0);
  fan::graphics::model_t::properties_t p;
#if !1
  p.path = "models/xbot_idle.fbx";
#else
  p.path = "models/final_provence.fbx";
#endif
  p.texture_path = "models/textures";
  p.use_cpu = 0;
  delete anim0;
  delete model;
  model = new fan::graphics::model_t(p);
  //return 0;
  p.path = "anim0.gltf";

  anim0 = new fan::graphics::model_t(p);
  model->import_animation(*anim0, "Idle2");
  
  model->animation_list["Idle2"].weight = 1;
  model->active_anim = "Idle2";
  
  fan::vec2 window_size = gloco->window.get_size();

  bool draw_lines = 0;

  fan::graphics::file_save_dialog_t save_file_dialog;
  fan::graphics::file_open_dialog_t open_file_dialog;
  std::string fn;

  model->light_intensity = 1;

  // for top left text
  f32_t menu_height = 0;

  fan::vec3 src = 0;
  fan::vec3 dst = 0;

  gloco->m_pre_draw.push_back([&]() {

    ImGui::BeginDisabled(!cursor_mode);

    if (ImGui::BeginMainMenuBar()) {

      if (ImGui::BeginMenu("Animation"))
      {
        if (ImGui::MenuItem("Open model")) {
          open_file_dialog.load("gltf,fbx,glb,dae,vrm", &fn);
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
      //delete model->scene;
     // model->scene = 0;
      //delete model;
      //model = 0;
      //delete anim0;
      //anim0 = 0;
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

    static bool draw_skeleton = 0;
    if (ImGui::ToggleButton("draw skeleton", &draw_skeleton)) {
      skeleton_lines.clear();
    }

    if (draw_skeleton) {
      skeleton_lines.clear();
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


    static bool spin = false;
    if (spin) {
      rotation.y += pile->loco.delta_time / 3;
      rotation.y = fmod(rotation.y, fan::math::two_pi);
    }
    if (ImGui::CollapsingHeader("model settings")) {
      ImGui::ToggleButton("spin", &spin);
      ImGui::DragFloat3("position", position.data(), 0.1);
      ImGui::DragFloat3("rotation", rotation.data(), 0.01);
      ImGui::DragFloat("scale", &all_scale, 0.01);
      scale = all_scale;

      model->print_bone_recursive(model->root_bone);
      //
      ImGui::Indent(40.f);
      fan_imgui_dragfloat1(model->light_position, 0.2);
      ImGui::ColorEdit3("model->light_color", model->light_color.data());
      fan_imgui_dragfloat1(model->light_intensity, 0.1);
      static f32_t specular_strength = 0.5;
      if (fan_imgui_dragfloat1(specular_strength, 0.01)) {
        pile->loco.shader_set_value(model->m_shader, "specular_strength", specular_strength);
      }
      ImGui::Unindent(40.f);
    }
    m = fan::mat4(1).translate(position) * fan::mat4(1).rotate(rotation) * fan::mat4(1).scale(scale);

    model->mouse_modify_joint(pile->loco.delta_time);
    if (ImGui::CollapsingHeader("animation settings")) {
      ImGui::Indent(40.f);
      model->display_animations();
      ImGui::Unindent(40.f);
    }

    auto cam_nr = pile->loco.perspective_camera.camera;
    auto& camera_data = pile->loco.camera_get(cam_nr);
    auto& v = pile->loco.viewport_get(pile->loco.perspective_camera.viewport);
    pile->loco.viewport_set(v.viewport_position, v.viewport_size, pile->loco.window.get_size());
    pile->loco.camera_set_perspective(cam_nr, 90.f, v.viewport_size);
    model->draw(m, bts);

    ImGui::End();
    ImGui::EndDisabled();
  });

  auto& camera = gloco->camera_get(gloco->perspective_camera.camera);

  fan::vec2 motion = 0;

  pile->loco.window.add_mouse_motion([&](const auto& d) {
    motion = d.motion;
    if (cursor_mode == 0) {
      camera.rotate_camera(d.motion);
    }
  });

   pile->loco.window.add_buttons_callback([&](const auto& d) { 
     if (editor.hovered() == false && cursor_mode == 1) {
       return;
     }
     switch (d.button) {
     case fan::mouse_right: {
      cursor_mode = !!!cursor_mode;
      pile->loco.window.set_cursor(cursor_mode);
      break;
     }
     }
  });

  pile->loco.camera_set_position(pile->loco.perspective_camera.camera, { 0.0426, 0.2974, 0.5768 });
  camera.m_yaw = 178.59;
  camera.m_pitch = -13.579;
  camera.update_view();


  pile->loco.loop([&] {
    camera.move(100);

    editor.begin_render();

    static fan::quat r;
    fan::vec3 angs;
    r.to_angles(angs);
    ImGui::DragFloat3("angle", angs.data(), 0.01);
    r = fan::quat::from_angles(angs);
    auto& bt = model->animation_list[model->active_anim].bone_transform_tracks;
    auto& bones = fan_3d::model::fms_t::bone_names_default;
    fan::quat parent = bt[bones.left_arm].rotations[0];
   //bt[bones.left_arm].rotations[0] = parent.inverse() * fan::quat(fan::quat::from_angles({0, 1, 0}));

    fan::graphics::text(
      camera.position.to_string() + " " +
      std::to_string(camera.get_yaw()) + " " +
      std::to_string(camera.get_pitch()),
      fan::vec2(0, menu_height)
    );

    editor.end_render();

    motion = 0;
  });
  delete pile;
}