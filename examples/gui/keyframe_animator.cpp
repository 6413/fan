#include "keyframe_animator.h"
#include fan_pch

// 1s / 10.f
inline f32_t time_divider = 10.f;

struct key_frame_t {
  f32_t time = 0;
  fan::vec3 position = 0;
  fan::vec2 size = 400;
  f32_t angle = 0;
  fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
};

// per object
struct animation_t {
  void push_frame(const key_frame_t& key_frame) {
    key_frames.push_back(key_frame);
    update_key_frame_value();
  }

  void update_key_frame_render_value() {
    if (key_frames.empty()) {
      return;
    }
    if (frame_index >= key_frames.size()) {
      return;
    }
    current_frame = key_frames[frame_index];
    if (current_frame.time == 0) {
      current_frame.time = 1;
    }
  }

  void update_key_frame_value() {
    if (key_frames.empty()) {
      return;
    }
    if (frame_index + 1 >= key_frames.size()) {
      return;
    }
    current_frame = key_frames[frame_index];
    current_frame.time = key_frames[frame_index + 1].time - key_frames[frame_index].time;
    if (current_frame.time == 0) {
      current_frame.time = 1;
    }
  }

  void update_seek() {
    if (key_frames.empty()) {
      return;
    }
    if (frame_index + 1 >= key_frames.size()) {
      return;
    }
    auto& frame_src = key_frames[frame_index];
    auto& frame_dst = key_frames[frame_index + 1];
    current_frame.position = frame_src.position.lerp(frame_dst.position, current_frame.time);
    current_frame.size = frame_src.size.lerp(frame_dst.size, current_frame.time);
    current_frame.angle = fan::math::lerp(frame_src.angle, frame_dst.angle, current_frame.time);
    current_frame.rotation_vector = frame_src.rotation_vector.lerp(frame_dst.rotation_vector, current_frame.time);
  }

  void update(f32_t dt) {
    if (key_frames.empty()) {
      return;
    }
    if (frame_index + 1 >= key_frames.size()) {
      return;
    }
    auto& frame_src = current_frame;
    auto& frame_dst = key_frames[frame_index + 1];
    if (current_frame.time >= 0) {
      if (current_frame.time == 0) {
        current_frame = key_frames[frame_index];
      }
      else {
        current_frame.position += (frame_dst.position - frame_src.position) / current_frame.time * dt;
        current_frame.size += (frame_dst.size - frame_src.size) / current_frame.time * dt;
        current_frame.angle += (frame_dst.angle - frame_src.angle) / current_frame.time * dt;
        current_frame.rotation_vector += (frame_dst.rotation_vector - frame_src.rotation_vector) / current_frame.time * dt;
        current_frame.time -= dt;
      }
    }
    else {
      frame_index++;
      current_frame = key_frames[frame_index];
      update_key_frame_value();
    }
  }
  int frame_index = 0;
  std::vector<key_frame_t> key_frames;
  key_frame_t current_frame;
  std::unique_ptr<fan::graphics::vfi_root_t> sprite;
  f32_t time = 0;
};

std::vector<animation_t> objects;

template <typename T>
bool make_imgui_element(const char* label, T& value) {
  if constexpr (std::is_same_v<T, f32_t>) {
    return ImGui::DragFloat(label, &value, .1, -10000, 10000);
  }
  else if constexpr (std::is_same_v<T, fan::vec2>) {
    return ImGui::DragFloat2(label, value.data(), .1, -10000, 10000);
  }
  else if constexpr (std::is_same_v<T, fan::vec3>) {
    return ImGui::DragFloat3(label, value.data(), .1, -10000, 10000);
  }
}

struct controls_t {
  bool playing = false;
  bool loop = true;
  f32_t time = 0;
  f32_t max_time = 0;
}controls;

struct timeline_t {
  int32_t current_frame = 0;
  int32_t start_frame = 0;
  int32_t end_frame = 256;
  std::vector<ImGui::FrameIndexType> frames;
  bool do_delete = false;
}timeline;



void play_from_begin() {
  // seek to begin
  controls.time = 0;
  for (auto& obj : objects) {
    obj.frame_index = 0;
    obj.update_key_frame_value();
  }
  controls.playing = true;
}

void handle_imgui() {
  auto& style = ImGui::GetStyle();
  ImVec4* colors = style.Colors;
  const ImVec4 bgColor = ImVec4(0.1, 0.1, 0.1, 0.1);
  colors[ImGuiCol_WindowBg].w = bgColor.w;
  colors[ImGuiCol_ChildBg].w = bgColor.w;
  colors[ImGuiCol_TitleBg].w = bgColor.w;

  ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
  ImGui::PushStyleColor(ImGuiCol_DockingEmptyBg, ImVec4(0, 0, 0, 0));
  ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());
  ImGui::PopStyleColor(2);
  ImGui::Begin("Editor");
  fan::vec2 window_size = gloco->window.get_size();
  fan::vec2 viewport_size = ImGui::GetContentRegionAvail();
  fan::vec2 viewport_pos = fan::vec2(ImGui::GetWindowPos() + fan::vec2(0, ImGui::GetFontSize() + ImGui::GetStyle().FramePadding.y * 2));
  fan::vec2 offset = viewport_size - viewport_size;
  fan::vec2 s = viewport_size;
  gloco->default_camera->camera.set_ortho(
    fan::vec2(-s.x, s.x),
    fan::vec2(-s.y, s.y)
  );
  gloco->default_camera->viewport.set(viewport_pos, viewport_size, window_size);
  ImGui::End();
  
  static int active_object = 0;

  static std::vector<const char*> v;
  ImGui::Begin("Key frames");
  if (ImGui::Button("+")) {
    objects.resize(objects.size() + 1);

    fan::graphics::sprite_t temp{{
      .position = viewport_size / 2,
      .size = 100
    }};

    loco_t::shapes_t::vfi_t::properties_t vfip;
    vfip.shape.rectangle->position = temp.get_position();
    vfip.shape.rectangle->position.z += 1;
    vfip.shape.rectangle->size = temp.get_size();
    vfip.mouse_button_cb = [](const auto& d) -> int {
      return 0;
    };
    vfip.mouse_move_cb = [](const auto& d) -> int {
      /*if (root->erasing) {
        return 0;
      }*/
      return 0;
    };
    objects.back().sprite = std::make_unique<fan::graphics::vfi_root_t>();
    objects.back().sprite.get()->set_root(vfip);
    objects.back().sprite.get()->push_child(std::move(temp));
    v.resize(v.size() + 1, "object");
  }
  ImGui::SameLine();
  if (ImGui::Button("Insert keyframe")) {
    if (objects.size()) {
      for (auto& obj : objects) {
        auto& child = obj.sprite.get()->children[0];
        key_frame_t kf{
          .time = (float)timeline.current_frame / time_divider,
          .position = child.get_position(),
          .size = child.get_size(),
          .angle = child.get_angle(),
          .rotation_vector = child.get_rotation_vector()
        };
        obj.push_frame(kf);
      }
      timeline.frames.push_back(timeline.current_frame);
    }
  }
  {
    ImGui::ListBox("##listbox_keyframes", &active_object, v.data(), v.size());
  }

  ImGui::End();
  //
  ImGui::Begin("Key frame properties");
  if (v.size()) {
    auto& child = objects[active_object].sprite.get()->children[0];
    fan::mp_t<key_frame_t> mp(key_frame_t{
      .position = child.get_position(),
      .size = child.get_size(),
      .angle = fan::math::degrees(child.get_angle()),
      .rotation_vector = child.get_rotation_vector()
    });
    {
      static constexpr const char* names[]{ "time", "position", "size", "angle", "rotation vector" };
      bool edit = false;
      mp.iterate([&]<auto i>(auto & v) {
        if (make_imgui_element(names[i], v)) {
          edit = true;
        }
      });//////
      if (edit) {
        objects[active_object].sprite->set_position(mp.operator key_frame_t().position);
        objects[active_object].sprite->set_size(mp.operator key_frame_t().size);
        child.set_angle(fan::math::radians(mp.operator key_frame_t().angle));
        child.set_rotation_vector(mp.operator key_frame_t().rotation_vector);
      }
      static fan::string input;
      input.resize(30);
      if (ImGui::InputText("Image", input.data(), input.size(), ImGuiInputTextFlags_EnterReturnsTrue)) {
        input = input.c_str();
        static loco_t::image_t image;
        if (image.load(input)) {
          fan::print_warning("failed to load image:"+ input);
        }
        else {
          objects[active_object].sprite->children[0].set_image(&image);
        }
      }
    }
  }
  ImGui::End();

  ImGui::Begin("Controls");
  if (ImGui::Button("Play")) {
    play_from_begin();
  }
  ImGui::SameLine();
  if (ImGui::Button("Continue")) {
    controls.playing = true;
  }
  ImGui::SameLine();
  if (ImGui::Button("Pause")) {
    controls.playing = false;
  }
  ImGui::Checkbox("Loop", &controls.loop);
  if (controls.playing) {
    for (auto& obj : objects) {
      obj.update(gloco->delta_time);
      key_frame_t kf = obj.current_frame;
      obj.sprite.get()->set_position(kf.position);
      obj.sprite.get()->set_size(kf.size);
      obj.sprite.get()->children[0].set_angle(kf.angle);
      obj.sprite.get()->children[0].set_rotation_vector(kf.rotation_vector);
    }
    timeline.current_frame = controls.time * time_divider;
    controls.time += gloco->delta_time;
    if (controls.loop && controls.max_time <= controls.time) {
      play_from_begin();
    }
  }

  if (ImGui::BeginNeoSequencer("Sequencer", &timeline.current_frame, &timeline.start_frame, &timeline.end_frame, { 0, 300 },
    ImGuiNeoSequencerFlags_EnableSelection |
    ImGuiNeoSequencerFlags_Selection_EnableDragging |
    ImGuiNeoSequencerFlags_Selection_EnableDeletion | 
    ImGuiNeoSequencerFlags_AllowLengthChanging))
  {
    static bool group_open = true;
    if (ImGui::BeginNeoGroup("", &group_open))
    {

      if (ImGui::BeginNeoTimelineEx("Transform"))
      {
        {
          int idx = 0;
          controls.max_time = 0;
          for (auto&& v : timeline.frames)
          {
            for (auto& obj : objects) {
              obj.key_frames[idx].time = v / time_divider;
              controls.max_time = std::max(controls.max_time, obj.key_frames[idx].time);
            }
            ++idx;
            ImGui::NeoKeyframe(&v);
          }
        }

        static int prev_frame = timeline.current_frame;
        // scrolling time
        if (!controls.playing && objects.size() && prev_frame != timeline.current_frame && objects.size() > active_object) {
          for (auto& obj : objects) {
            auto& frames = obj.key_frames;

            f32_t closest_greater = std::numeric_limits<f32_t>::max();
            int frame_index = 0;
            int idx = 0;
            for (const auto& item : frames) {
              if (item.time > (float)timeline.current_frame / time_divider && item.time < closest_greater) {
                closest_greater = item.time;
                frame_index = idx;
              }
              else if (closest_greater == std::numeric_limits<f32_t>::max() && idx + 1 == frames.size()) {
                frame_index = idx + 1;
              }
              idx++;
            }
            obj.frame_index = std::max(frame_index - 1, 0);
            obj.update_key_frame_render_value();
            if (obj.frame_index + 1 < obj.key_frames.size()) {
              obj.current_frame.time = fan::math::normalize(timeline.current_frame / time_divider, obj.key_frames[obj.frame_index].time, obj.key_frames[obj.frame_index + 1].time);
            }
            else {
              // todo fix
              obj.current_frame.time = 1.0 - timeline.current_frame / time_divider;
            }
            obj.update_seek();

            obj.update(gloco->delta_time);
            key_frame_t kf = obj.current_frame;

            obj.sprite.get()->set_position(kf.position);
            obj.sprite.get()->set_size(kf.size);
            obj.sprite.get()->children[0].set_angle(kf.angle);
            obj.sprite.get()->children[0].set_rotation_vector(kf.rotation_vector);
            prev_frame = timeline.current_frame;
          }
        }

        if (timeline.do_delete)
        {
          uint32_t count = ImGui::GetNeoKeyframeSelectionSize();

          ImGui::FrameIndexType* toRemove = new ImGui::FrameIndexType[count];

          ImGui::GetNeoKeyframeSelection(toRemove);
        }
        ImGui::EndNeoTimeLine();
      }
      ImGui::EndNeoGroup();
    }

    ImGui::EndNeoSequencer();
  }

  ImGui::End();
}

int main() {
  loco_t loco;
  auto&io = ImGui::GetIO();
  io.FontGlobalScale = 1.5;

  fan::graphics::imgui_element_t main_view =
    fan::graphics::imgui_element_t([&] {handle_imgui(); });

  loco.loop([&] {

  });
}