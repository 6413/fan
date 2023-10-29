#include fan_pch

struct fgm_t {

  static constexpr uint32_t version_001 = 1;

  fgm_t(const fan::string& texturepack_name) {
    texturepack.open_compiled(texturepack_name);
  }

  static constexpr auto editor_str = "Editor";
  static constexpr auto create_str = "Create";
  static constexpr auto properties_str = "Properties";

  static constexpr auto max_depth = 0xff;
  static constexpr int max_path_input = 40;

  static constexpr int max_id_input = 10;
  static constexpr fan::vec2 default_button_size{100, 30};

  fan::string file_name = "file.fgm";

  struct shapes_t{
    struct global_t : fan::graphics::vfi_root_t, fan::graphics::imgui_element_t {

      struct shape_data {
        struct sprite_t {
          fan::string image_name;
        }sprite;
      }shape_data;

      global_t() = default;

      template <typename T>
      global_t(fgm_t* fgm, const T& obj) : fan::graphics::imgui_element_t(){
        T temp = std::move(obj);
        loco_t::vfi_t::properties_t vfip;
        vfip.shape.rectangle->position = temp.get_position();
        vfip.shape.rectangle->position.z += fgm->current_z++;
        vfip.shape.rectangle->size = temp.get_size();
        vfip.mouse_button_cb = [fgm, this] (const auto& d) -> int {
          fgm->event_type = event_type_e::move;
          fgm->current_shape = this;
          return 0;
        };
        vfi_root_t::set_root(vfip);
        temp.set_position(fan::vec3(fan::vec2(temp.get_position()), fgm->current_z - 1));
        vfi_root_t::push_child(std::move(temp));

        fgm->current_shape = this;
      }

      // global data
      fan::string id;
      uint32_t group_id = 0;
    };
  };

  struct version_001_t {

    struct sprite_t {

      static constexpr typename loco_t::shape_type_t::_t shape_type = loco_t::shape_type_t::sprite;

      fan::vec3 position;
      fan::vec2 size;
      fan::color color;
      fan::string image_name;

      // global
      fan::string id;
      uint32_t group_id;

      void init(loco_t::shape_t& shape) {
        position = shape.get_position();
        size = shape.get_size();
        color = shape.get_color();
      }

      void from_string(const fan::string& str, uint64_t& off) {
        position = fan::read_data<fan::vec3>(str, off);
        size = fan::read_data<fan::vec3>(str, off);
        color = fan::read_data<fan::color>(str, off);
        position = fan::read_data<fan::vec3>(str, off);
      }

      shapes_t::global_t* get_shape(fgm_t* fgm) {
        fgm_t::shapes_t::global_t* ret = new fgm_t::shapes_t::global_t(
          fgm,
         fan::graphics::sprite_t{{
             .position = position,
             .size = size,
             .color = color
          }}
        );

        ret->shape_data.sprite.image_name = image_name;
        ret->id = id;
        ret->group_id = group_id;

        if (image_name.empty()) {
          return ret;
        }

        loco_t::texturepack_t::ti_t ti;
        if (fgm->texturepack.qti(image_name, &ti)) {
          fan::print_no_space("failed to load texture:", image_name);
        }
        else {
          auto& data = fgm->texturepack.get_pixel_data(ti.pack_id);
          gloco->sprite.load_tp(ret->children[0], &ti);
        }
        
        return ret;
      }
    };

    struct shapes_t {
      sprite_t sprite;
    };
  };

  static constexpr uint32_t current_version = version_001;
  using current_version_t = version_001_t;

  #define BLL_set_StoreFormat 1
  //#define BLL_set_CPP_CopyAtPointerChange
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_BaseLibrary 1
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
    fan::vec3 v = shape->CONCAT(get_, prop)(); \
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
    make_line(fan::vec3, position);
    make_line(fan::vec2, size);

    {
      ImGui::Text("angle");
      ImGui::SameLine();
      f32_t angle = shape->children[0].get_angle();
      angle = fan::math::degrees(angle);
      ImGui::SliderFloat("##hidden_label1" "angle", &angle, 0, 360);
      angle = fan::math::radians(angle);
      shape->children[0].set_angle(angle);

    }

    {
      fan::string& id = current_shape->id;
      fan::string str = id;
      str.resize(max_id_input);
      ImGui::Text("id");
      ImGui::SameLine();
      if (ImGui::InputText("##hidden_label0" "id", str.data(), str.size())) { \
        if (ImGui::IsItemDeactivatedAfterEdit()) {
          fan::string new_id = str;
          if (!id_exists(new_id)) {
            id = new_id;
          }
        }
      } 
    }
    switch (shape->children[0]->shape_type) {
      case loco_t::shape_type_t::sprite:{
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
            else{
              current = str.substr(0, std::strlen(str.c_str()));
              auto& data = texturepack.get_pixel_data(ti.pack_id);
              gloco->sprite.load_tp(shape->children[0], &ti);
            }
          }
        }
        break;
      }
    }
  }


  void push_rectangle(const fan::vec2& pos) {
    auto nr = shape_list.NewNodeLast();

    shape_list[nr] = new shapes_t::global_t{this, fan::graphics::rectangle_t{{
        .position = pos,
        .size = 100,
        .color = fan::colors::black
    }}};
    // has to be updated after because you cannot loop bll inside temp obj
    while (id_exists(std::to_string(current_id))) { ++current_id; }
    shape_list[nr]->id = std::to_string(current_id);
  }
  void push_sprite(const fan::vec2& pos) {
    auto nr = shape_list.NewNodeLast();

    shape_list[nr] = new shapes_t::global_t{this, fan::graphics::sprite_t{{
      .position = pos,
      .size = 100,
      .blending = true
    }}};
    while (id_exists(std::to_string(current_id))) { ++current_id; }
    shape_list[nr]->id = std::to_string(current_id);
  }

  fan::graphics::imgui_element_t main_view = 
    fan::graphics::imgui_element_t(
    [&]{
      auto& style = ImGui::GetStyle();
      ImVec4* colors = style.Colors;

      const ImVec4 bgColor = ImVec4(0.1, 0.1, 0.1, 0.1);
      colors[ImGuiCol_WindowBg] = bgColor;
      colors[ImGuiCol_ChildBg] = bgColor;
      colors[ImGuiCol_TitleBg] = bgColor;

      ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
      ImGui::PushStyleColor(ImGuiCol_DockingEmptyBg, ImVec4(0, 0, 0, 0));
      ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());
      ImGui::PopStyleColor(2);

      fan::vec2 editor_size;

      //// Create a window that docks to the top
      if (ImGui::Begin(editor_str, nullptr, ImGuiWindowFlags_DockNodeHost)) {
        //ImGuiViewport* viewport = ;
        fan::vec2 window_size = gloco->get_window()->get_size();
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
          // Get the position of the mouse click
          ImVec2 pos = ImGui::GetMousePos();
          switch (selected_shape_type) {
            case loco_t::shape_type_t::rectangle: {
              push_rectangle(pos);
              break;
            }
            case loco_t::shape_type_t::sprite: {
              push_sprite(pos);
              break;
            }
          }
        }
      }

      ImGui::End();

      if (ImGui::Begin(create_str, nullptr, ImGuiWindowFlags_DockNodeHost)) {
        if (ImGui::Button("Rectangle")) {
          event_type = event_type_e::add;
          selected_shape_type = loco_t::shape_type_t::rectangle;
        }
        else if (ImGui::Button("Sprite")) {
          event_type = event_type_e::add;
          selected_shape_type = loco_t::shape_type_t::sprite;
        }
        {
          fan::vec2 window_size = ImGui::GetWindowSize();
          ImGui::SetCursorPos(fan::vec2(
            window_size.x - default_button_size.x - ImGui::GetStyle().WindowPadding.x, 
            window_size.y - default_button_size.y - ImGui::GetStyle().WindowPadding.y)
          );
          if (ImGui::Button("Save")) {
            fout(file_name);
          }
        }
      }

      ImGui::End();

      if (ImGui::Begin(properties_str, nullptr, ImGuiWindowFlags_DockNodeHost)) {
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
    while(it != shape_list.dst) {
      shapes.iterate([&]<auto i0, typename T>(T& l) {
        if (shape_list[it]->children[0]->shape_type != T::shape_type) {
          return;
        }

        ostr.append((char*)&shape_list[it]->children[0]->shape_type, sizeof(T::shape_type));

        fan::mp_t<T> shape;
        shape.init(shape_list[it]->children[0]);

        if constexpr (std::is_same_v<T, current_version_t::sprite_t>) {
          shape.image_name = shape_list[it]->shape_data.sprite.image_name;
          shape.id = shape_list[it]->id;
          shape.group_id = shape_list[it]->group_id;
        }
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

        ostr += shape_str;
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
    fan::string in;
    fan::io::file::read(filename, &in);
    uint64_t off = 0;
    uint32_t version = fan::read_data<uint32_t>(in, off);
    if (version != current_version) {
      fan::print_format("invalid file version, file:{}, current:{}", version, current_version);
      return;
    }
    fan::mp_t<current_version_t::shapes_t> shapes;
    while (off != in.size()) {
      bool ignore = true;
      uint32_t byte_count = 0;
      shapes.iterate([&]<auto i0, typename T>(T& v0) {
        uint16_t shape_type = fan::read_data<uint16_t>(in, off);
        byte_count = fan::read_data<uint32_t>(in, off);
        if (shape_type != T::shape_type) {
          return;
        }
        ignore = false;

        auto it = shape_list.NewNodeLast();

        fan::mp_t<T> shape;
        shape.iterate([&]<auto i, typename T2>(T2& v) {
          v = fan::read_data<T2>(in, off);
        });

        shape_list[it] = shape.get_shape(this);
      });
      // if shape is not part of version
      if (ignore) {
        off += byte_count;
      }
    }
  }

  event_type_e event_type = event_type_e::none;
  loco_t::shape_type_t::_t selected_shape_type = loco_t::shape_type_t::invalid;
  shapes_t::global_t* current_shape = nullptr;
  shape_list_t shape_list;

  f32_t current_z = 0;
  uint32_t current_id = 0;

  loco_t::texturepack_t texturepack;
};

int main() {
  loco_t loco;

  fgm_t fgm("TexturePack");
  fgm.fin("file.fgm");

  loco.loop([&] {

  });

  return 0;
}
