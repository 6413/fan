#pragma once

struct fgm_t {

  void open(const fan::string& texturepack_name) {
    texturepack.open_compiled(texturepack_name);

    gloco->window.add_keys_callback([this](const auto& d) {
      if (d.state != fan::keyboard_state::press) {
        return;
      }
      if (ImGui::IsAnyItemActive()) {
        return;
      }

      switch (d.key) {
        case fan::key_r: {
          erase_current();
          break;
        }
      }
    });
  }
  void close() {
    texturepack.close();
  }

  static constexpr auto editor_str = "Editor";
  static constexpr auto create_str = "Create";
  static constexpr auto properties_str = "Properties";

  static constexpr auto max_depth = 0xff;
  static constexpr int max_path_input = 40;

  static constexpr int max_id_input = 20;
  static constexpr fan::vec2 default_button_size{100, 30};

  fan::string file_name = "file.fgm";

  struct shapes_t {
    struct global_t : fan::graphics::vfi_root_t, fan::graphics::imgui_element_t {

      struct shape_data {
        struct sprite_t {
          fan::string image_name;
        }sprite;
      }shape_data;

      global_t() = default;

      template <typename T>
      global_t(fgm_t* fgm, const T& obj) : fan::graphics::imgui_element_t() {
        T temp = std::move(obj);
        loco_t::shapes_t::vfi_t::properties_t vfip;
        vfip.shape.rectangle->position = temp.get_position();
        vfip.shape.rectangle->position.z += fgm->current_z++;
        vfip.shape.rectangle->size = temp.get_size();
        vfip.mouse_button_cb = [fgm, this](const auto& d) -> int {
          fgm->event_type = event_type_e::move;
          fgm->current_shape = this;
          return 0;
          };
        fan::graphics::vfi_root_t::set_root(vfip);
        temp.set_position(fan::vec3(fan::vec2(temp.get_position()), fgm->current_z - 1));
        fan::graphics::vfi_root_t::push_child(std::move(temp));

        fgm->current_shape = this;
      }

      // global data
      fan::string id;
      uint32_t group_id = 0;
    };
  };

  #include _FAN_PATH(graphics/gui/fgm/common.h)

  #define BLL_set_StoreFormat 1
  //#define BLL_set_CPP_CopyAtPointerChange
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_AreWeInsideStruct 1
  #include _FAN_PATH(fan_bll_preset.h)
  #define BLL_set_prefix shape_list
  #define BLL_set_type_node uint32_t
  #define BLL_set_NodeDataType shapes_t::global_t*
  #define BLL_set_Link 1
  #include _FAN_PATH(BLL/BLL.h)


  enum class event_type_e {
    none,
    add,
    remove,
    move,
    resize
  };


  #define make_line(T, prop) \
  { \
    T v = shape->CONCAT(get_, prop)(); \
 \
    auto str = v.to_string(); \
 \
    str.resize(str.size() + 10); \
 \
    ImGui::Text(STRINGIFY_DEFINE(prop)); \
    ImGui::SameLine(); \
    if (ImGui::InputText("##hidden_label0" STRINGIFY_DEFINE(prop), str.data(), str.size())) { \
      T p = fan::string_to<T>(str); \
      v = p; \
    } \
 \
    ImGui::Indent();\
    ImGui::Text("x"); \
    ImGui::SameLine(); \
    ImGui::SliderFloat("##hidden_label1" STRINGIFY_DEFINE(prop), &v[0], 0, editor_size.x); \
    if constexpr (T::size() > 1) { \
      \
        ImGui::Text("y"); \
        ImGui::SameLine(); \
        ImGui::SliderFloat("##hidden_label2" STRINGIFY_DEFINE(prop), &v[1], 0, editor_size.y); \
    } \
      if constexpr (T::size() > 2) {\
        \
          ImGui::Text("z"); \
          ImGui::SameLine(); \
          ImGui::SliderFloat("##hidden_label3" STRINGIFY_DEFINE(prop), &v[2], 0, max_depth); \
      } \
        ImGui::Unindent(); \
        \
          shape->CONCAT(set_, prop)(v); \
  }

  bool id_exists(const fan::string& id) {
    auto it = shape_list.GetNodeFirst();
    while (it != shape_list.dst) {
      if (shape_list[it]->id == id) {
        return true;
      }
      it = it.Next(&shape_list);
    }
    return false;
  }

  void open_properties(fgm_t::shapes_t::global_t* shape, const fan::vec2& editor_size) {

    fan::string shape_str = fan::string("Shape name:") + gloco->shape_names[shape->children[0]->shape_type];
    ImGui::Text(shape_str.c_str());

    make_line(fan::vec3, position);
    make_line(fan::vec2, size);

    {
      ImGui::Text("angle");
      ImGui::SameLine();
      fan::vec3 angle = shape->children[0].get_angle();
      angle.x = fan::math::degrees(angle.x);
      angle.y = fan::math::degrees(angle.y);
      angle.z = fan::math::degrees(angle.z);
      ImGui::SliderFloat3("##hidden_label1" "angle", angle.data(), 0, 360);
      angle = fan::math::radians(angle);
      shape->children[0].set_angle(angle);

    }

    {
      fan::string& id = current_shape->id;
      fan::string str = id;
      str.resize(max_id_input);
      ImGui::Text("id");
      ImGui::SameLine();
      if (ImGui::InputText("##hidden_label0" "id", str.data(), str.size())) {
        \
          if (ImGui::IsItemDeactivatedAfterEdit()) {
            fan::string new_id = str.substr(0, std::strlen(str.c_str()));
            if (!id_exists(new_id)) {
              id = new_id;
            }
          }
      }
    }
    {
      fan::string id = std::to_string(current_shape->group_id);
      fan::string str = id;
      str.resize(max_id_input);
      ImGui::Text("group id");
      ImGui::SameLine();
      if (ImGui::InputText("##hidden_label0" "group id", str.data(), str.size())) {
        \
          if (ImGui::IsItemDeactivatedAfterEdit()) {
            current_shape->group_id = std::stoul(str);
          }
      }
    }
    switch ((loco_t::shape_type_t)shape->children[0]->shape_type) {
      case loco_t::shape_type_t::unlit_sprite:
      case loco_t::shape_type_t::sprite: {
        fan::string& current = shape->shape_data.sprite.image_name;
        fan::string str = current;
        str.resize(max_path_input);
        ImGui::Text("image name");
        ImGui::SameLine();
        if (ImGui::InputText("##hidden_label4", str.data(), str.size())) {
          if (ImGui::IsItemDeactivatedAfterEdit()) {
            loco_t::texturepack_t::ti_t ti;
            if (texturepack.qti(str, &ti)) {
              fan::print_no_space("failed to load texture:", str);
            }
            else {
              current = str.substr(0, std::strlen(str.c_str()));
              auto& data = texturepack.get_pixel_data(ti.pack_id);
              if ((loco_t::shape_type_t)shape->children[0]->shape_type == loco_t::shape_type_t::sprite) {
                gloco->shapes.sprite.load_tp(shape->children[0], &ti);
              }
              else if ((loco_t::shape_type_t)shape->children[0]->shape_type == loco_t::shape_type_t::unlit_sprite) {
                gloco->shapes.unlit_sprite.load_tp(shape->children[0], &ti);
              }
            }
          }
        }
        break;
      }
    }
  }

  void push_shape(loco_t::shape_type_t shape_type, const fan::vec2& pos) {
    auto nr = shape_list.NewNodeLast();

    static fan::mp_t<current_version_t::shapes_t> mp;
    mp.iterate([&]<auto i, typename T> (T& v) {
      if (shape_type == v.shape_type) {
        shape_list[nr] = new shapes_t::global_t{this, typename T::type_t{{
            .position = pos,
            .size = 100
          }}};
      }
    });
  }

  fan::graphics::imgui_element_t main_view =
    fan::graphics::imgui_element_t(
    [&] {
      fan::vec2 editor_size;

      if (ImGui::Begin(editor_str, nullptr)) {
        fan::vec2 window_size = gloco->window.get_size();
        fan::vec2 viewport_size = ImGui::GetWindowSize();
        fan::vec2 ratio = viewport_size / viewport_size.max();
        gloco->default_camera->camera.set_ortho(
          fan::vec2(0, viewport_size.x),
          fan::vec2(0, viewport_size.y)
        );
        gloco->default_camera->viewport.set(ImGui::GetWindowPos(), viewport_size, window_size);
        editor_size = ImGui::GetContentRegionAvail();
      }

      if (ImGui::IsWindowHovered()) {
        if (ImGui::IsMouseClicked(0) &&
       event_type == event_type_e::add) {
          ImVec2 pos = ImGui::GetMousePos();
          push_shape(selected_shape_type, pos);
        }
      }

      ImGui::End();

      if (ImGui::Begin(create_str, nullptr)) {
        static fan::mp_t<current_version_t::shapes_t> mp;
        mp.iterate([&]<auto i> (auto& v) {
          if (ImGui::Button(gloco->shape_names[(std::underlying_type_t<loco_t::shape_type_t>)v.shape_type])) {
            event_type = event_type_e::add;
            selected_shape_type = v.shape_type;
          }
        });
        {
          fan::vec2 window_size = ImGui::GetWindowSize();
          fan::vec2 cursor_pos(
            window_size.x - default_button_size.x - ImGui::GetStyle().WindowPadding.x,
            window_size.y - default_button_size.y - ImGui::GetStyle().WindowPadding.y
          );
          ImGui::SetCursorPos(cursor_pos);
          if (ImGui::Button("Save")) {
            fout(file_name);
          }
          cursor_pos.x += default_button_size.x / 2;
          ImGui::SetCursorPos(cursor_pos);
          if (ImGui::Button("Quit")) {
            auto it = shape_list.GetNodeFirst();
            while (it != shape_list.dst) {
              delete shape_list[it];
              it = it.Next(&shape_list);
            }

            close_cb();
            ImGui::End();
            return;
          }
        }
      }

      ImGui::End();

      if (ImGui::Begin(properties_str, nullptr)) {
        if (current_shape != nullptr) {
          open_properties(current_shape, editor_size);
        }
      }

      ImGui::End();
  });
  /*
  header 4 byte
  shape_type 2 byte
  struct size x byte
  data{
    ...
  }
  */
  void fout(const fan::string& filename) {

    fan::string ostr;
    ostr.append((char*)&current_version, sizeof(current_version));
    fan::mp_t<current_version_t::shapes_t> shapes;
    auto it = shape_list.GetNodeFirst();
    while (it != shape_list.dst) {
      shapes.iterate([&]<auto i0, typename T>(T & l) {
        auto shape_type = shape_list[it]->children[0]->shape_type;
        //!
        if (!((loco_t::shape_type_t)shape_type == loco_t::shape_type_t::rectangle && T::shape_type == loco_t::shape_type_t::mark)) {
          if ((loco_t::shape_type_t)shape_type != T::shape_type) {
            return;
          }
        }
        

        ostr.append((char*)&shape_type, sizeof(T::shape_type));

        fan::mp_t<T> shape;
        shape.init(shape_list[it]);

        fan::string shape_str;
        shape.iterate([&]<auto i1, typename T2>(T2 & v) {
          if constexpr (std::is_same_v<T2, fan::string>) {
            uint64_t string_length = v.size();
            shape_str.append((char*)&string_length, sizeof(string_length));
            shape_str.append(v);
          }
          else {
            shape_str.append((char*)&v, sizeof(T2));
          }
        });

        uint32_t struct_size = shape_str.size();
        ostr.append((char*)&struct_size, sizeof(struct_size));

        ostr +=shape_str;
      });
      it = it.Next(&shape_list);
    }

    fan::io::file::write(filename, ostr, std::ios_base::binary);
    fan::print("file saved to:" + filename);
  }
  /*
  header - 4 byte
  shape_type - 2 byte
  struct size - x byte
  data{
    ...
  }
  */
  void fin(const fan::string& filename) {
    #include _FAN_PATH(graphics/gui/stage_maker/loader_versions/1.h)
  }

  void invalidate_current() {
    current_shape = nullptr;
    selected_shape_type = loco_t::shape_type_t::invalid;
  }

  void erase_current() {
    if (current_shape == nullptr) {
      return;
    }

    auto it = shape_list.GetNodeFirst();
    while (it != shape_list.dst) {
      if (current_shape == shape_list[it]) {
        delete shape_list[it];
        shape_list.unlrec(it);
        invalidate_current();
        break;
      }
      it = it.Next(&shape_list);
    }
  }

  event_type_e event_type = event_type_e::none;
  loco_t::shape_type_t selected_shape_type = loco_t::shape_type_t::invalid;
  shapes_t::global_t* current_shape = nullptr;
  shape_list_t shape_list;

  f32_t current_z = 0;
  uint32_t current_id = 0;

  loco_t::texturepack_t texturepack;

  fan::function_t<void()> close_cb = [] {};
};
