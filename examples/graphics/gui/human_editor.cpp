#include <fan/pch.h>

struct bone_t : fan::graphics::bone_t{
  bool is_dragging = false;
};

struct human_editor_t {
  human_editor_t() {
    content_browser.current_directory /= "characters";
    content_browser.update_directory_cache();
  }

  void render_content_browser() {
    namespace fs = std::filesystem;
    content_browser.render();

    if (content_browser.item_right_clicked) {
      ImGui::OpenPopup("content_browser_right_click");      
      current_directory = (fs::absolute(content_browser.current_directory) / content_browser.item_right_clicked_name).string();
    }

    if (ImGui::BeginPopup("content_browser_right_click")) {
      if (ImGui::MenuItem("Open in editor")) {
        loco_t::image_load_properties_t lp;
        lp.min_filter = loco_t::image_filter::nearest;
        lp.mag_filter = loco_t::image_filter::nearest;
        lp.visual_output = loco_t::image_sampler_address_mode::repeat;
        bone_images = fan::graphics::human_t::load_character_images(current_directory, lp);
        
        fan::graphics::human_t human;
        human.load(0, human_scale, bone_images);
        
        fan::vec2 size = 0;
        int boneid = 0;
        for (const auto& i : human.bones) {
          *dynamic_cast<fan::graphics::bone_t*>(&bones[boneid]) = i;
          size = size.max(i.size * i.scale);
          ++boneid;
        }
        boneid = 0;
        int body_skip = 0;
        for (auto& i : bones) {
            if (// display only one side
            body_skip == fan::graphics::bone_e::lower_right_arm ||
            body_skip == fan::graphics::bone_e::upper_right_arm ||
            body_skip == fan::graphics::bone_e::lower_right_leg ||
            body_skip == fan::graphics::bone_e::upper_right_leg
            ) {
            ++body_skip;
            continue;
          }

          *(fan::vec2*)&i.position = size * fan::vec2(0 + boneid % 600, 2 + boneid / 600);
          ++boneid;
          ++body_skip;
        }
        human.erase();
        for (auto& i : bones) {
          i.visual.erase();
        }
      }
      ImGui::EndPopup();
    }
  }

  void render_menubar() {
    if (ImGui::BeginMenuBar()) {
      if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("Open..", "Ctrl+O")) {
          open_file_dialog.load("json;fmm", &fn);
        }
        if (ImGui::MenuItem("Save", "Ctrl+S")) {
          //fout(previous_file_name);
        }
        if (ImGui::MenuItem("Save as", "Ctrl+Shift+S")) {
          save_file_dialog.save("json;fmm", &fn);
        }
        if (ImGui::MenuItem("Quit")) {

        }
        ImGui::EndMenu();
      }
      ImGui::EndMenuBar();
    }
    if (open_file_dialog.is_finished()) {
      if (fn.size() != 0) {
        fin(fn);
      }
      open_file_dialog.finished = false;
    }
    if (save_file_dialog.is_finished()) {
      if (fn.size() != 0) {
        fout(fn);
      }
      save_file_dialog.finished = false;
    }
  }

  void render() {
    render_content_browser();

    ImGui::Begin("Editor", 0, ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoBackground);
    render_menubar();


    std::vector<int> indices(bones.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
      return bones[a].position.z < bones[b].position.z;
    });

    if (ImGui::Button("test")) {
      human = new fan::graphics::human_t;
      human->load_bones(0, .5f, human->bones);
      for (int i = 0; i < std::size(human->bones); ++i) {
        if (i == fan::graphics::bone_e::lower_right_arm) {
        bones[i].position = bones[fan::graphics::bone_e::lower_left_arm].position;
        bones[i].offset = bones[fan::graphics::bone_e::lower_left_arm].offset;
        bones[i].scale = bones[fan::graphics::bone_e::lower_left_arm].scale;
        bones[i].size = bones[fan::graphics::bone_e::lower_left_arm].size;
      }
      if (i == fan::graphics::bone_e::upper_right_arm) {
        bones[i].position = bones[fan::graphics::bone_e::upper_left_arm].position;
        bones[i].offset = bones[fan::graphics::bone_e::upper_left_arm].offset;
        bones[i].scale = bones[fan::graphics::bone_e::upper_left_arm].scale;
        bones[i].size = bones[fan::graphics::bone_e::upper_left_arm].size;
      }
      if (i == fan::graphics::bone_e::lower_right_leg) {
        bones[i].position = bones[fan::graphics::bone_e::lower_left_leg].position;
        bones[i].offset = bones[fan::graphics::bone_e::lower_left_leg].offset;
        bones[i].scale = bones[fan::graphics::bone_e::lower_left_leg].scale;
        bones[i].size = bones[fan::graphics::bone_e::lower_left_leg].size;
      }
      if (i == fan::graphics::bone_e::upper_right_leg) {
        bones[i].position = bones[fan::graphics::bone_e::upper_left_leg].position;
        bones[i].offset = bones[fan::graphics::bone_e::upper_left_leg].offset;
        bones[i].scale = bones[fan::graphics::bone_e::upper_left_leg].scale;
        bones[i].size = bones[fan::graphics::bone_e::upper_left_leg].size;
      }
        human->bones[i].position = fan::vec2(bones[i].position / fan::physics::length_units_per_meter + bones[i].offset / fan::physics::length_units_per_meter) + (bones[i].size*bones[i].scale/2)/fan::physics::length_units_per_meter;
        human->bones[i].offset = 0;
        human->bones[i].scale = bones[i].scale;
      }
      human->load_preset(fan::vec2(0, 0), 1.0f, bone_images, human->bones);
    }


    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.5, 0.5, 0.5, 0.5));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.5, 0.5, 0.5, 0.5));

    fan::vec2 window_size = ImGui::GetWindowSize();
    fan::vec2 scale_factor = window_size / (fan::window_t::resolutions::r_1920x1080 / 2);

    static bool flip_x = 0;
    f32_t center_x = bones[fan::graphics::bone_e::hip].position.x;
    fan::vec2 image_size = 0;
    bool dragging_bone = false;
    for (int index : indices) {
      const auto& bone = bones[index];

      if (bone_images[index].iic()) {
        continue;
      }
      if (// display only one side
        index == fan::graphics::bone_e::lower_right_arm ||
        index == fan::graphics::bone_e::upper_right_arm ||
        index == fan::graphics::bone_e::lower_right_leg ||
        index == fan::graphics::bone_e::upper_right_leg
        ) {
        continue;
      }

      ImGui::PushID(index);

      fan::vec2 scaled_size = (bone.size * bone.scale) * scale_factor.y * fan::physics::length_units_per_meter;
      fan::vec2 final_pos = bone.position;
      if (flip_x) {
        final_pos.x += (center_x - bone.position.x) * scale_factor.y;
      }
      final_pos = (final_pos + bone.offset) * scale_factor.y * fan::physics::length_units_per_meter;

      image_size = scaled_size.max(scaled_size);
      ImGui::SetCursorPos(final_pos - ((bone.size * bone.scale)/2)*scale_factor.y*fan::physics::length_units_per_meter);
      ImGui::SetNextItemAllowOverlap();
      ImGui::ImageButton("", bone_images[index], scaled_size, flip_x == 0 ? fan::vec2(0, 0) : fan::vec2(1, 0), flip_x == 0 ? fan::vec2(1, 1) : fan::vec2(0, 1));

      if (!is_dragging_pivot && ImGui::IsItemActive()) {
        dragging_bone = true;
        bones[index].is_dragging = true;
        bones[index].position = fan::vec2(ImGui::GetMousePos() / scale_factor.y / fan::physics::length_units_per_meter);
        selected_bone = index;
      }
      ImGui::PopID();
    }

    if (image_size > 0) {
      fan::vec2 pivot_screen_pos = global_pivot_position * scale_factor.y;

      ImGui::SetCursorPos(pivot_screen_pos * fan::physics::length_units_per_meter - fan::vec2(10) * scale_factor.y);
      ImGui::InvisibleButton("pivot hitbox", fan::vec2(20)*scale_factor.y);
      //fan::print(global_pivot_position);
      if (!dragging_bone && ImGui::IsMouseDown(0) && ImGui::IsItemActive()) {
        global_pivot_position = ImGui::GetMousePos() / scale_factor.y / fan::physics::length_units_per_meter;

        is_dragging_pivot = true;
      }
      else {
        is_dragging_pivot = false;
      }

      ImGui::GetWindowDrawList()->AddRectFilled(
        (pivot_screen_pos)*fan::physics::length_units_per_meter - fan::vec2(10) * scale_factor.y,
        (pivot_screen_pos)*fan::physics::length_units_per_meter + fan::vec2(10) * scale_factor.y,
        0xffffffff
      );
    }

    ImGui::PopStyleColor(3);
    ImGui::End();

    ImGui::Begin("Bone settings");
    if (selected_bone != -1) {
      ImGui::Text(fan::graphics::bone_to_string(selected_bone).c_str());
      ImGui::Text(("parent:" + fan::graphics::bone_to_string(bones[selected_bone].parent_index)).c_str());

      fan::vec2 position = bones[selected_bone].position;
      if (fan_imgui_dragfloat1(position, 1)) {
        bones[selected_bone].position = position;
      }

      f32_t scale = bones[selected_bone].scale;
      if (fan_imgui_dragfloat1(scale, 0.01)) {
        bones[selected_bone].scale = scale;
      }
    }

    if (ImGui::Button("Flip x")) {
      flip_x = !flip_x;
    }

    if (ImGui::Combo("Select Bone", &selected_pivot_bone , fan::graphics::bone_names + 1, std::size(fan::graphics::bone_names) - 1)) {
      global_pivot_position = bones[selected_pivot_bone  + 1].pivot;
    }
    if (ImGui::Button("Set pivot")) {
      bones[selected_pivot_bone  + 1].pivot = global_pivot_position;
    }

    ImGui::End();
  }

  void fout(const std::string& filename) {
    if (current_directory.empty()) {
      return;
    }
    fan::json human;
    fan::json arr = fan::json::array();
    int boneid = 0;
    for (const auto& bone : bones) {
      fan::json bone_info;
      bone_info = fan::json()["bone_info"];
      bone_info["friction_scale"] = bone.friction_scale;
      bone_info["parent_index"] = bone.parent_index;
      bone_info["position"] = bone.position + fan::vec3(bone.offset, 0);
      bone_info["size"] = bone.size;
      bone_info["pivot"] = bone.pivot;
      bone_info["scale"] = bone.scale;
      bone_info["lower_angle"] = bone.lower_angle;
      bone_info["upper_angle"] = bone.upper_angle;
      bone_info["reference_angle"] = bone.reference_angle;
      bone_info["boneid"] = boneid;
      arr.push_back(bone_info);
      ++boneid;
    }
    human["human"] = arr;
    human["bone_count"] = fan::graphics::bone_e::bone_count;
    human["image_directory"] = current_directory;
    fan::io::file::write(filename, human.dump(2), std::ios_base::binary);
  }
  void fin(const std::string& filename) {
    selected_bone = -1;
    std::string data;
    fan::io::file::read(filename, &data);

    fan::json human = fan::json::parse(data);
    fan::json arr = human["human"];

    if (human["bone_count"] != fan::graphics::bone_e::bone_count) {
      fan::print("invalid bone count");
      return;
    }
    current_directory = human["image_directory"];
    loco_t::image_load_properties_t lp;
    lp.min_filter = loco_t::image_filter::nearest;
    lp.mag_filter = loco_t::image_filter::nearest;
    lp.visual_output = loco_t::image_sampler_address_mode::repeat;
    bone_images = fan::graphics::human_t::load_character_images(current_directory, lp);

    for (const auto& bone_info : arr) {
      bone_t bone;
      bone.friction_scale = bone_info["friction_scale"];
      bone.parent_index = bone_info["parent_index"];
      bone.position = bone_info["position"];
      bone.size = bone_info["size"];
      bone.pivot = bone_info["pivot"];
      bone.scale = bone_info["scale"];
      bone.lower_angle = bone_info["lower_angle"];
      bone.upper_angle = bone_info["upper_angle"];
      bone.reference_angle = bone_info["reference_angle"];
      int bone_id = bone_info["boneid"];
      bones[bone_id] = bone;
    }
    global_pivot_position = bones[selected_pivot_bone + 1].pivot;
  }


  std::array<bone_t, fan::graphics::bone_e::bone_count> bones;
  std::array<loco_t::image_t, fan::graphics::bone_e::bone_count> bone_images;
  std::string current_directory;
  fan::graphics::gui::imgui_content_browser_t content_browser;

  fan::graphics::file_save_dialog_t save_file_dialog;
  fan::graphics::file_open_dialog_t open_file_dialog;
  std::string fn;
  f32_t human_scale = 1;
  fan::graphics::human_t* human = nullptr;
  int selected_bone = -1;
  fan::vec2 pivot = 400 / fan::physics::length_units_per_meter;
  bool is_dragging_pivot = false;
  fan::vec2 global_pivot_position = 400 / fan::physics::length_units_per_meter;

  int selected_pivot_bone = 0;  
};

int main() {
  fan::graphics::engine_t engine;

  human_editor_t human_editor;
  human_editor.fin("human.json");

  fan::vec2 window_size = engine.window.get_size();
  f32_t wall_thickness = 50.f;
  //auto walls = fan::graphics::physics_shapes::create_stroked_rectangle(window_size / 2, window_size / 2, wall_thickness);

  engine.loop([&] {
    human_editor.render();
  });
}