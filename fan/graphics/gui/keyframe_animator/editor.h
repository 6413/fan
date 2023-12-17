#pragma once

namespace fan {
  namespace graphics {
    struct animation_editor_t {
      fan::window_t::keys_callback_NodeReference_t keyboard_it;
      animation_editor_t() {
        keyboard_it = gloco->window.add_keys_callback([&](const auto& d) {
          auto& imio = ImGui::GetIO();
          if (imio.WantTextInput) {
            return;
          }
          if (d.state != fan::keyboard_state::press) {
            return;
          }
          switch (d.key) {
            case fan::key_space: {
              controls.playing = !controls.playing;
              break;
            }
          }
        });
      }

      // 1s / 10.f
      static constexpr f32_t time_divider = 10.f;

      struct key_frame_t {
        f32_t time = 0;
        fan::vec3 position = 0;
        fan::vec2 size = 400;
        fan::vec3 angle = 0;
        fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
      };

      struct controls_t {
        bool playing = false;
        bool loop = true;
        f32_t time = 0;
        f32_t max_time = 0;
      };
      struct object_t {
        int frame_index = 0;
        std::vector<key_frame_t> key_frames;
        key_frame_t current_frame;
        // can be either image or texturepack image name
        fan::string image_name;
        std::unique_ptr<fan::graphics::vfi_root_t> sprite;
      };

      bool update(object_t& obj) {
        if (obj.key_frames.empty()) {
          return 0;
        }
        if (obj.frame_index + 1 >= obj.key_frames.size()) {
          return 0;
        }
        if (controls.time < obj.key_frames[obj.frame_index].time) {
          return 0;
        }
        auto& frame_src = obj.key_frames[obj.frame_index];
        auto& frame_dst = obj.key_frames[obj.frame_index + 1];
        if (controls.time < frame_dst.time) {
          f32_t offset = fan::math::normalize(controls.time, frame_src.time, frame_dst.time);
          obj.current_frame.position = frame_src.position.lerp(frame_dst.position, offset);
          obj.current_frame.size = frame_src.size.lerp(frame_dst.size, offset);
          obj.current_frame.angle = fan::math::lerp(frame_src.angle, frame_dst.angle, offset);
          obj.current_frame.rotation_vector = frame_src.rotation_vector.lerp(frame_dst.rotation_vector, offset);
        }
        else {
          obj.frame_index++;
          obj.current_frame = obj.key_frames[obj.frame_index];
        }
        return 1;
      }

      void play_from_begin() {
        controls.time = 0;
        for (auto& obj : objects) {
          obj.frame_index = 0;
        }
        controls.playing = true;
      }

      static void load_image(loco_t::shape_t& shape, const fan::string& name, loco_t::image_t& image) {
        if (image.load(name)) {
          fan::print_warning("failed to load image:" + name);
        }
        else {
          shape.set_image(&image);
        }
      }

      static void load_image(loco_t::shape_t& shape, const fan::string& name, loco_t::texturepack_t& texturepack) {
        loco_t::texturepack_t::ti_t ti;
        if (texturepack.qti(name, &ti)) {
          fan::print_warning("failed to load texturepack image:" + name);
        }
        else {
          shape.set_tp(&ti);
        }
      }

      bool is_finished() {
        return controls.max_time <= controls.time;
      }

      void push_sprite(uint32_t i, auto&& temp) {
        loco_t::shapes_t::vfi_t::properties_t vfip;
        vfip.shape.rectangle->position = temp.get_position();
        vfip.shape.rectangle->position.z += 1;
        vfip.shape.rectangle->size = temp.get_size();
        objects[i].sprite = std::make_unique<fan::graphics::vfi_root_t>();
        objects[i].sprite.get()->set_root(vfip);
        objects[i].sprite.get()->push_child(std::move(temp));
        object_list_names.resize(object_list_names.size() + 1, "object");
      }

      struct timeline_t {
        int32_t current_frame = 0;
        int32_t start_frame = 0;
        int32_t end_frame = 256;
        std::vector<ImGui::FrameIndexType> frames;
        bool do_delete = false;
      }timeline;

      template <typename T>
      bool make_imgui_element(const char* label, T& value) {
        if constexpr (std::is_same_v<T, f32_t>) {
          return ImGui::DragFloat(label, &value, .5, -10000, 10000);
        }
        else if constexpr (std::is_same_v<T, fan::vec2>) {
          return ImGui::DragFloat2(label, value.data(), .5, -10000, 10000);
        }
        else if constexpr (std::is_same_v<T, fan::vec3>) {
          return ImGui::DragFloat3(label, value.data(), .5, -10000, 10000);
        }
      }

      void find_closest_frame(std::vector<key_frame_t>& frames, int& frame_index) {
        f32_t closest_greater = std::numeric_limits<f32_t>::max();
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
      }

      void update_play_frame() {
        for (auto& obj : objects) {
          if (!update(obj)) {
            continue;
          }
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

      void insert_and_update_keyframes() {
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

      void update_seek(object_t& obj) {
        if (obj.key_frames.empty()) {
          return;
        }
        if (obj.frame_index + 1 >= obj.key_frames.size()) {
          return;
        }
        auto& frame_src = obj.key_frames[obj.frame_index];
        auto& frame_dst = obj.key_frames[obj.frame_index + 1];
        // assuming controls.time is between src and dst
        f32_t offset = fan::math::normalize(controls.time, frame_src.time, frame_dst.time);
        obj.current_frame.position = frame_src.position.lerp(frame_dst.position, offset);
        obj.current_frame.size = frame_src.size.lerp(frame_dst.size, offset);
        obj.current_frame.angle = fan::math::lerp(frame_src.angle, frame_dst.angle, offset);
        obj.current_frame.rotation_vector = frame_src.rotation_vector.lerp(frame_dst.rotation_vector, offset);
      }

      void handle_timeline_seek(int& prev_frame) {
        for (auto& obj : objects) {
          auto& frames = obj.key_frames;
          if (frames.empty()) {
            continue;
          }

          int frame_index = 0;

          find_closest_frame(frames, frame_index);

          obj.frame_index = std::max(frame_index - 1, 0);

          controls.time = timeline.current_frame / time_divider;
          if (controls.time >= controls.max_time) {
            return;
          }

          update_seek(obj);
          if (!update(obj)) {
            continue;
          }

          key_frame_t kf = obj.current_frame;
          obj.sprite.get()->set_position(kf.position);
          obj.sprite.get()->set_size(kf.size);
          obj.sprite.get()->children[0].set_angle(kf.angle);
          obj.sprite.get()->children[0].set_rotation_vector(kf.rotation_vector);
        }
        prev_frame = timeline.current_frame;
      }

      void handle_imgui() {
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

        ImGui::Begin("Key frames");
        if (ImGui::Button("+")) {
          objects.resize(objects.size() + 1);
          if (objects.size() > 1) {
            objects.back().key_frames = objects[objects.size() - 2].key_frames;
          }

          fan::graphics::sprite_t temp{ {
            .position = viewport_size / 2,
            .size = 100
          } };
          push_sprite(objects.size() - 1, std::move(temp));
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
              obj.key_frames.push_back(kf);
            }
            timeline.frames.push_back(timeline.current_frame);
          }
        }
        {
          ImGui::ListBox("##listbox_keyframes", &active_object, object_list_names.data(), object_list_names.size());
        }

        static fan::string save_file_str;
        save_file_str.resize(40);
        if (ImGui::InputText("Save", save_file_str.data(), save_file_str.size(), ImGuiInputTextFlags_EnterReturnsTrue)) {
          file_save(save_file_str.c_str());
        }
        static fan::string load_file_str;
        load_file_str.resize(40);
        if (ImGui::InputText("Load", load_file_str.data(), load_file_str.size(), ImGuiInputTextFlags_EnterReturnsTrue)) {
          object_list_names.clear();
          timeline.frames.clear();
          objects.clear();
          file_load(load_file_str.c_str());
        }

        ImGui::End();
        //
        ImGui::Begin("Key frame properties");
        if (object_list_names.size()) {
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
              loco_t::texturepack_t::ti_t ti;
              if (image.load(input)) {
                fan::print_warning("failed to load image:" + input);
              }
              else {
                auto& ob = objects[active_object].sprite->children[0];
                ob.set_image(&image);
                ob.set_tc_position(0);
                ob.set_tc_size(1);
                objects[active_object].image_name = input;
              }
              if (texturepack.qti(input, &ti)) {
                fan::print_warning("failed to load texturepack image:" + input);
              }
              else {
                objects[active_object].sprite->children[0].set_tp(&ti);
                objects[active_object].image_name = input;
              }
            }
          }
        }
        ImGui::End();
        ImGui::Begin("Key frame data");
        if (active_object < objects.size()) {
          int idx = 0;
          for (auto& frame : objects[active_object].key_frames) {
            ImGui::Text(fan::format("frame {}:", std::to_string(idx++)).c_str());
            fan::mp_t<key_frame_t> mp(frame);
            {
              static const char* names[]{ "time", "position", "size", "angle", "rotation vector" };
              bool edit = false;
              mp.iterate([&]<auto i>(auto & v) {
                if (make_imgui_element(("##" + fan::to_string(idx) + fan::string(names[i])).c_str(), v)) {
                  edit = true;
                }
                ImGui::SameLine();
                ImGui::Text(names[i]);
              });//////
              if (edit) {
                frame = mp;
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
          update_play_frame();
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
              insert_and_update_keyframes();

              static int prev_frame = timeline.current_frame;
              // scrolling time
              if (!controls.playing && prev_frame != timeline.current_frame) {
                handle_timeline_seek(prev_frame);
              }

              if (ImGui::IsKeyPressed(ImGuiKey_Delete)) {
                timeline.do_delete = true;
              }
              else {
                timeline.do_delete = false;
              }

              if (timeline.do_delete)
              {
                uint32_t count = ImGui::GetNeoKeyframeSelectionSize();

                ImGui::FrameIndexType* toRemove = new ImGui::FrameIndexType[count];

                ImGui::GetNeoKeyframeSelection(toRemove);

                for (int i = 0; i < count; ++i) {
                  auto found = std::find_if(timeline.frames.begin(), timeline.frames.end(),
                    [&](int a) {
                      return a == toRemove[i];
                    });
                  if (found != timeline.frames.end()) {
                    timeline.frames.erase(found);
                    for (auto& obj : objects) {
                      obj.key_frames.erase(obj.key_frames.begin() + std::distance(timeline.frames.begin(), found));
                    }
                  }
                }
                delete toRemove;
              }
              ImGui::EndNeoTimeLine();
            }
            ImGui::EndNeoGroup();
          }

          ImGui::EndNeoSequencer();
        }

        ImGui::End();
      }

      /*
        global data
        keyframe data
      */
      void file_save(const fan::string& path) {
        fan::string ostr;
        fan::write_to_string(ostr, controls.loop);
        fan::write_to_string(ostr, controls.max_time);
        fan::write_to_string(ostr, (uint32_t)objects.size());
        for (auto& obj : objects) {
          fan::write_to_string(ostr, obj.image_name);
          fan::write_to_string(ostr, (uint32_t)obj.key_frames.size());
          if (obj.key_frames.size()) {
            ostr.append((char*)&obj.key_frames[0], sizeof(key_frame_t) * obj.key_frames.size());
          }
        }
        fan::io::file::write(path, ostr, std::ios_base::binary);
      }

      void file_load(const fan::string& path) {
        fan::string istr;
        fan::io::file::read(path, &istr);
        uint32_t off = 0;
        fan::read_from_string(istr, off, controls.loop);
        fan::read_from_string(istr, off, controls.max_time);
        uint32_t obj_size = 0;
        fan::read_from_string(istr, off, obj_size);
        objects.resize(obj_size);
        // todo remove
        static std::vector<loco_t::image_t> images(obj_size);
        int iterate_idx = 0;
        for (auto& obj : objects) {
          fan::read_from_string(istr, off, obj.image_name);
          uint32_t keyframe_size = 0;
          fan::read_from_string(istr, off, keyframe_size);
          obj.key_frames.resize(keyframe_size);
          int frame_idx = 0;
          for (auto& frame : obj.key_frames) {
            frame = ((key_frame_t*)&istr[off])[frame_idx++];
            timeline.frames.push_back(frame.time * time_divider);
          }
          memcpy(obj.key_frames.data(), &istr[off], sizeof(key_frame_t) * obj.key_frames.size());
          off += sizeof(key_frame_t) * obj.key_frames.size();
          if (obj.key_frames.size()) {
            fan::graphics::sprite_t temp{ {
              .position = obj.key_frames[0].position,
              .size = obj.key_frames[0].size,
              .angle = obj.key_frames[0].angle,
              .rotation_vector = obj.key_frames[0].rotation_vector
            } };

            load_image(temp, obj.image_name, texturepack);
            push_sprite(iterate_idx, std::move(temp));
          }
          iterate_idx += 1;
        }
      }
      f32_t time = 0;
      controls_t controls;
      std::vector<object_t> objects;
      std::vector<const char*> object_list_names;
      loco_t::texturepack_t texturepack;
    };
  }
}