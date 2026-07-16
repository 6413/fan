#if defined(FAN_GUI)

export module fan.graphics.editor:animation_system;

#endif

#if defined(FAN_GUI)
import fan.math;
import fan.graphics.gui.base;
import :fgm_types;

export namespace fan::graphics::editor {
  enum class rotation_position_e : std::uint8_t {
    center, top_left, top_middle, top_right, middle_left, middle_right, bottom_left, bottom_middle, bottom_right
  };

  struct shape_keyframe_animation_t {
    struct keyframe_t {
      static keyframe_t lerp(const keyframe_t& a, const keyframe_t& b, f32_t t) {
        return {
          a.time + (b.time - a.time) * t,
          a.position + (b.position - a.position) * t,
          a.size + (b.size - a.size) * t,
          a.angle + (b.angle - a.angle) * t,
          a.rotation_pos
        };
      }
      f32_t time = 0.0f;
      fan::vec3 position = 0;
      fan::vec2 size = 0;
      fan::vec3 angle = 0;
      rotation_position_e rotation_pos = rotation_position_e::center;
    };

    fan::vec2 get_rotation_offset(rotation_position_e pos, const fan::vec2& half_size) const {
      switch (pos) {
        case rotation_position_e::top_left: return {-half_size.x, -half_size.y};
        case rotation_position_e::top_middle: return {0, -half_size.y};
        case rotation_position_e::top_right: return {half_size.x, -half_size.y};
        case rotation_position_e::middle_left: return {-half_size.x, 0};
        case rotation_position_e::middle_right: return {half_size.x, 0};
        case rotation_position_e::bottom_left: return {-half_size.x, half_size.y};
        case rotation_position_e::bottom_middle: return {0, half_size.y};
        case rotation_position_e::bottom_right: return {half_size.x, half_size.y};
        default: return {0, 0};
      }
    }

    void add_keyframe(const fan::vec3& pos, const fan::vec2& sz, const fan::vec3& ang, rotation_position_e rot_pos) {
      f32_t time = current_time;
      if (auto_increment_time && !keyframes.empty()) {
        time = keyframes.back().time + time_increment;
      }
      keyframe_t kf {time, pos, sz, ang, rot_pos};
      auto it = std::lower_bound(keyframes.begin(), keyframes.end(), kf, [](const keyframe_t& a, const keyframe_t& b) { return a.time < b.time; });
      keyframes.insert(it, kf);
      if (auto_increment_time) current_time = time;
    }

    void remove_keyframe(int index) {
      if (index >= 0 && index < keyframes.size()) {
        keyframes.erase(keyframes.begin() + index);
        if (selected_keyframe >= keyframes.size()) {
          selected_keyframe = keyframes.size() - 1;
        }
      }
    }

    keyframe_t get_current_frame() const {
      if (keyframes.empty()) return {};
      if (keyframes.size() == 1) return keyframes[0];

      for (std::size_t i = 0; i < keyframes.size() - 1; ++i) {
        if (current_time >= keyframes[i].time && current_time <= keyframes[i + 1].time) {
          f32_t delta = keyframes[i + 1].time - keyframes[i].time;
          return delta <= 0.0f ? keyframes[i] : keyframe_t::lerp(keyframes[i], keyframes[i + 1], (current_time - keyframes[i].time) / delta);
        }
      }

      if (slerp_to_first && loop && keyframes.size() > 1) {
        f32_t last_time = keyframes.back().time;
        if (current_time > last_time && current_time <= last_time + slerp_duration) {
          return keyframe_t::lerp(keyframes.back(), keyframes.front(), (current_time - last_time) / slerp_duration);
        }
      }
      return keyframes.back();
    }

    void update(f32_t dt) {
      if (!is_playing || keyframes.empty() || !owner_shape) return;
      current_time += dt;
      f32_t max_time = keyframes.back().time + (slerp_to_first && loop ? slerp_duration : 0.0f);
      if (current_time > max_time) {
        if (loop) current_time = 0.0f;
        else { current_time = max_time; is_playing = false; }
      }
    }

    void apply_to_shape(shapes_t::global_t* shape) {
      if (keyframes.empty() || !shape || shape != owner_shape) return;
      auto frame = get_current_frame();
      shape->set_position(frame.position);
      shape->children[0].set_size(frame.size);
      shape->children[0].set_angle(frame.angle);
      shape->children[0].set_rotation_point(get_rotation_offset(frame.rotation_pos, frame.size));
    }

    fan::json serialize() const {
      fan::json j;
      j["name"] = name;
      j["loop"] = loop;
      j["slerp_to_first"] = slerp_to_first;
      j["slerp_duration"] = slerp_duration;
      j["current_rotation_pos"] = static_cast<int>(current_rotation_pos);
      j["auto_increment_time"] = auto_increment_time;
      j["time_increment"] = time_increment;
      fan::json keyframes_json = fan::json::array();
      for (const auto& kf : keyframes) {
        fan::json kf_json;
        kf_json["time"] = kf.time;
        kf_json["position"] = kf.position;
        kf_json["size"] = kf.size;
        kf_json["angle"] = kf.angle;
        kf_json["rotation_pos"] = static_cast<int>(kf.rotation_pos);
        keyframes_json.push_back(kf_json);
      }
      j["keyframes"] = keyframes_json;
      return j;
    }

    void deserialize(const fan::json& j) {
      if (j.contains("name")) name = j["name"].get<std::string>();
      if (j.contains("loop")) loop = j["loop"].get<bool>();
      if (j.contains("slerp_to_first")) slerp_to_first = j["slerp_to_first"].get<bool>();
      if (j.contains("slerp_duration")) slerp_duration = j["slerp_duration"].get<f32_t>();
      if (j.contains("current_rotation_pos")) current_rotation_pos = static_cast<rotation_position_e>(j["current_rotation_pos"].get<int>());
      if (j.contains("auto_increment_time")) auto_increment_time = j["auto_increment_time"].get<bool>();
      if (j.contains("time_increment")) time_increment = j["time_increment"].get<f32_t>();
      
      keyframes.clear();
      selected_keyframe = -1;
      current_time = 0.0f;
      is_playing = false;

      if (j.contains("keyframes")) {
        for (const auto& kf_json : j["keyframes"]) {
          keyframes.push_back({
            kf_json["time"].get<f32_t>(),
            kf_json["position"],
            kf_json["size"],
            kf_json["angle"],
            static_cast<rotation_position_e>(kf_json["rotation_pos"].get<int>())
          });
        }
      }
    }

    void render_gui(shapes_t::global_t* shape) {
      if (owner_shape != shape) {
        gui::text("ERROR: Wrong animation! This belongs to another shape.");
        return;
      }

      gui::input_text("Name", &name);
      if (gui::button(is_playing ? "Stop" : "Play")) {
        is_playing = !is_playing;
        if (is_playing && current_time >= (keyframes.empty() ? 0 : keyframes.back().time)) current_time = 0.0f;
      }
      gui::same_line();
      if (gui::button("Reset")) current_time = 0.0f;
      gui::same_line();
      gui::checkbox("Loop", &loop);
      gui::same_line();
      gui::checkbox("Slerp to First", &slerp_to_first);

      gui::checkbox("Auto Increment Time", &auto_increment_time);
      if (auto_increment_time) gui::drag("Time Increment", &time_increment, 0.01f, 0.0f, 10.0f);
      if (slerp_to_first) gui::drag("Slerp Duration", &slerp_duration, 0.01f, 0.0f, 10.0f);

      f32_t max_time = keyframes.empty() ? 10.f : keyframes.back().time + (slerp_to_first && loop ? slerp_duration : 0.0f);
      if (gui::drag("Time##current_time", &current_time, 0.01f, 0.0f, std::max(max_time, 0.1f))) apply_to_shape(shape);

      gui::separator();
      gui::text("Keyframes: ", (int)keyframes.size());

      if (gui::button("Add Keyframe")) {
        add_keyframe(shape->get_position(), shape->children[0].get_size(), shape->children[0].get_angle(), current_rotation_pos);
      }
      gui::same_line();
      if (gui::button("Remove Selected") && selected_keyframe >= 0) remove_keyframe(selected_keyframe);

      gui::separator();
      gui::begin_child("keyframes_list", fan::vec2(0, 200), true);
      for (int i = 0; i < keyframes.size(); ++i) {
        gui::push_id(i);
        if (gui::selectable(("KF " + std::to_string(i) + ": " + std::to_string(keyframes[i].time) + "s").c_str(), i == selected_keyframe)) {
          selected_keyframe = i;
          current_time = keyframes[i].time;
          apply_to_shape(shape);
        }
        gui::pop_id();
      }
      gui::end_child();

      if (selected_keyframe >= 0 && selected_keyframe < keyframes.size()) {
        gui::separator();
        auto& kf = keyframes[selected_keyframe];

        if (gui::drag("Time", &kf.time, 0.01f, 0.0f, std::numeric_limits<f32_t>::max())) {
          std::sort(keyframes.begin(), keyframes.end(), [](const keyframe_t& a, const keyframe_t& b) { return a.time < b.time; });
        }
        if (gui::drag("Position", &kf.position, 0.1f) && current_time == kf.time) shape->set_position(kf.position);
        if (gui::drag("Size", &kf.size, 0.1f) && current_time == kf.time) shape->children[0].set_size(kf.size);
        
        fan::vec3 angle_deg = fan::math::degrees(kf.angle);
        if (gui::drag("Angle", &angle_deg, 1.0f)) {
          kf.angle = fan::math::radians(angle_deg);
          if (current_time == kf.time) shape->children[0].set_angle(kf.angle);
        }

        int rot_pos_idx = static_cast<int>(kf.rotation_pos);
        const char* rot_names[] = { "Center", "Top Left", "Top Middle", "Top Right", "Middle Left", "Middle Right", "Bottom Left", "Bottom Middle", "Bottom Right" };
        if (gui::combo("Rotation Position", &rot_pos_idx, rot_names, 9)) {
          kf.rotation_pos = static_cast<rotation_position_e>(rot_pos_idx);
          if (current_time == kf.time) apply_to_shape(shape);
        }
      }

      gui::separator();
      int global_rot_pos = static_cast<int>(current_rotation_pos);
      const char* global_rot_names[] = { "Center", "Top Left", "Top Middle", "Top Right", "Middle Left", "Middle Right", "Bottom Left", "Bottom Middle", "Bottom Right" };
      if (gui::combo("Default Rotation Position", &global_rot_pos, global_rot_names, 9)) {
        current_rotation_pos = static_cast<rotation_position_e>(global_rot_pos);
      }
    }

    std::vector<keyframe_t> keyframes;
    std::string name = "Animation";
    f32_t current_time = 0.0f;
    f32_t slerp_duration = 0.5f;
    f32_t time_increment = 0.5f;
    int selected_keyframe = -1;
    rotation_position_e current_rotation_pos = rotation_position_e::center;
    shapes_t::global_t* owner_shape = nullptr;
    bool is_playing = false;
    bool loop = true;
    bool slerp_to_first = true;
    bool auto_increment_time = true;
  };
}

#endif