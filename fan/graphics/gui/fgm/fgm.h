#pragma once

#include <fan/graphics/file_dialog.h>

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

  }

  static constexpr auto editor_str = "Editor";
  static constexpr auto create_str = "Create";
  static constexpr auto properties_str = "Properties";

  static constexpr auto max_depth = 0xff;
  static constexpr int max_path_input = 40;

  static constexpr int max_id_input = 20;
  static constexpr fan::vec2 default_button_size{ 100, 30 };

  fan::string previous_file_name;

  struct shapes_t {
    struct global_t : fan::graphics::vfi_root_t, fan::graphics::imgui_element_t {

      struct shape_data {
        struct sprite_t {
          fan::string image_name;
        }sprite;
      }shape_data;

      global_t() = default;

      uint16_t shape_type = 0;

      template <typename T>
      global_t(uint16_t shape_type, fgm_t* fgm, const T& obj) : fan::graphics::imgui_element_t() {
        T temp = obj;
        this->shape_type = shape_type;
        typename loco_t::vfi_t::properties_t vfip;
        vfip.shape.rectangle->position = temp.get_position();
        vfip.shape.rectangle->position.z += fgm->current_z++;
        vfip.shape.rectangle->size = temp.get_size();
        vfip.shape.rectangle->angle = 0;
        vfip.shape.rectangle->rotation_point = 0;
        vfip.mouse_button_cb = [fgm, this](const auto& d) -> int {
          fgm->event_type = event_type_e::move;
          fgm->current_shape = this;
          return 0;
          };
        fan::graphics::vfi_root_t::set_root(vfip);
        temp.set_position(fan::vec3(fan::vec2(temp.get_position()), fgm->current_z - 1));
        fan::graphics::vfi_root_t::push_child(temp);

        fgm->current_shape = this;
      }

      // global data
      fan::string id;
      uint32_t group_id = 0;
    };
  };
  //
#include _FAN_PATH(graphics/gui/fgm/common.h)

//#define BLL_set_StoreFormat 1
//#define BLL_set_CPP_CopyAtPointerChange 1
#define BLL_set_AreWeInsideStruct 1
#include <fan/fan_bll_preset.h>
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
    static auto str = v.to_string(); \
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
    fan_imgui_dragfloat_named(STRINGIFY_DEFINE(prop), v, 0.1, -1, -1); \
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

    fan::string shape_str = fan::string("Shape name:") + gloco->shape_names[shape->children[0].get_shape_type()];
    ImGui::Text("%s", shape_str.c_str());

    make_line(fan::vec3, position);
    make_line(fan::vec2, size);
    fan::color c = shape->get_color();

    if (ImGui::ColorEdit4("color", c.data())) {
      shape->set_color(c);
    }

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
    switch (shape->children[0].get_shape_type()) {
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
          if (texturepack.qti(str.c_str(), &ti)) {

            fan::print_no_space("failed to load texture:", str);
          }
          else {
            current = str.substr(0, std::strlen(str.c_str()));
            auto& data = texturepack.get_pixel_data(ti.pack_id);
            if (shape->children[0].get_shape_type() == loco_t::shape_type_t::sprite) {
              shape->children[0].load_tp(&ti);
            }
            else if (shape->children[0].get_shape_type() == loco_t::shape_type_t::unlit_sprite) {
              shape->children[0].load_tp(&ti);
            }
          }
        }
      }
      break;
    }
    }
  }

  void push_shape(uint16_t shape_type, const fan::vec2& pos) {
    auto nr = shape_list.NewNodeLast();

    switch (shape_type) {
    case loco_t::shape_type_t::sprite: {
      shape_list[nr] = new shapes_t::global_t{
        loco_t::shape_type_t::sprite,
        this, fan::graphics::sprite_t{{
          .position = pos,
          .size = 100
        }} };
      break;
    }
    case loco_t::shape_type_t::unlit_sprite: {
      shape_list[nr] = new shapes_t::global_t{
        loco_t::shape_type_t::unlit_sprite,
        this, fan::graphics::unlit_sprite_t{{
          .position = pos,
          .size = 100
        }} };
      break;
    }
    case loco_t::shape_type_t::rectangle: {
      shape_list[nr] = new shapes_t::global_t{
        loco_t::shape_type_t::rectangle,
        this, fan::graphics::rectangle_t{{
          .position = pos,
          .size = 100
        }} };
      break;
    }
    case loco_t::shape_type_t::light: {
      shape_list[nr] = new shapes_t::global_t{
        loco_t::shape_type_t::light,
        this, fan::graphics::light_t{{
          .position = pos,
          .size = 100
        }} };
        shape_list[nr]->push_child(fan::graphics::circle_t{ {
          .position = fan::vec3(pos, current_z - 1),
          .radius = 100,
          .color = fan::color(1, 1, 1, 0.5),
          .blending = true
        } });
      break;
    }
    }
  }

  fan::graphics::imgui_element_t main_view =
    fan::graphics::imgui_element_t(
      [&] {
        fan::vec2 editor_size;

        
        if (ImGui::Begin(editor_str, nullptr, ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoBackground)) {
          fan::vec2 window_size = gloco->window.get_size();
          fan::vec2 viewport_size = ImGui::GetWindowSize();
          fan::vec2 ratio = viewport_size / viewport_size.max();
          fan::vec2 s = viewport_size;

          //gloco->camera_set_ortho(
          //  gloco->orthographic_camera.camera,
          //  fan::vec2(0, viewport_size.x),
          //  fan::vec2(0, viewport_size.y)
          //);

          gloco->camera_set_ortho(
            gloco->orthographic_camera.camera,
            fan::vec2(0, viewport_size.x),
            fan::vec2(0, viewport_size.y)
          );
          gloco->viewport_set(
            gloco->orthographic_camera.viewport,
            ImGui::GetWindowPos(), viewport_size, window_size
          );
          editor_size = ImGui::GetContentRegionAvail();
        }

        static fan::graphics::file_save_dialog_t save_file_dialog;
        static fan::graphics::file_open_dialog_t open_file_dialog;
        static std::string fn;
        if (ImGui::BeginMenuBar())
        {
          if (ImGui::BeginMenu("File"))
          {

            if (ImGui::MenuItem("Open..", "Ctrl+O")) { 

              open_file_dialog.load("json;fmm", &fn);
            }
            if (ImGui::MenuItem("Save", "Ctrl+S")) {
              fout(previous_file_name);
            }
            if (ImGui::MenuItem("Save as", "Ctrl+Shift+S")) {
              save_file_dialog.save("json;fmm", &fn);
            }
            if (ImGui::MenuItem("Quit")) {
              auto it = shape_list.GetNodeFirst();
              while (it != shape_list.dst) {
                delete shape_list[it];
                it = it.Next(&shape_list);
              }
              shape_list.Clear();

              close_cb();
              ImGui::End();
            }
            ImGui::EndMenu();
          }
          ImGui::EndMenuBar();
        }
        if (open_file_dialog.is_finished()) {
          if (fn.size() != 0) {
            auto it = shape_list.GetNodeFirst();
            while (it != shape_list.dst) {
              delete shape_list[it];
              it = it.Next(&shape_list);
            }
            shape_list.Clear();
            fin(fn);
          }
          open_file_dialog.finished = false;
          return;
        }
        if (save_file_dialog.is_finished()) {
          if (fn.size() != 0) {
            fout(fn);
          }
          save_file_dialog.finished = false;
        }


        if (ImGui::IsWindowHovered()) {
          if (ImGui::IsMouseClicked(0) &&
            event_type == event_type_e::add) {
            ImVec2 pos = ImGui::GetMousePos();
            push_shape(selected_shape_type, pos);
          }
        }

        ImGui::End();

        if (ImGui::Begin("lighting settings")) {
          float arr[3];
          arr[0] = gloco->lighting.ambient.data()[0];
          arr[1] = gloco->lighting.ambient.data()[1];
          arr[2] = gloco->lighting.ambient.data()[2];
          //fan::print("suffering", (void*)gloco.loco, &gloco.loco->lighting, (void*)((uint8_t*)&gloco.loco->lighting - (uint8_t*)gloco.loco), sizeof(*gloco.loco), arr[0], arr[1], arr[2]);
          if (ImGui::ColorEdit3("ambient", gloco->lighting.ambient.data())) {

          }
        }
        ImGui::End();

        if (ImGui::Begin(properties_str, nullptr)) {
          if (current_shape != nullptr) {
            open_properties(current_shape, editor_size);
          }
        }

        ImGui::End();

         if (ImGui::Begin(create_str, nullptr)) {

          if (ImGui::Button(gloco->shape_names[loco_t::shape_type_t::sprite])) {
            event_type = event_type_e::add;
            selected_shape_type = loco_t::shape_type_t::sprite;
          }
          else if (ImGui::Button(gloco->shape_names[loco_t::shape_type_t::unlit_sprite])) {
            event_type = event_type_e::add;
            selected_shape_type = loco_t::shape_type_t::unlit_sprite;
          }
          else if (ImGui::Button(gloco->shape_names[loco_t::shape_type_t::rectangle])) {
            event_type = event_type_e::add;
            selected_shape_type = loco_t::shape_type_t::rectangle;
          }
          else if (ImGui::Button(gloco->shape_names[loco_t::shape_type_t::light])) {
            event_type = event_type_e::add;
            selected_shape_type = loco_t::shape_type_t::light;
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
    previous_file_name = filename;

    auto format = filename.substr(filename.find_last_of(".") + 1);
    if (format == "fmm") {
      fan::string ostr;
      ostr.append((char*)&current_version, sizeof(current_version));
      auto it = shape_list.GetNodeFirst();
      while (it != shape_list.dst) {
        auto& shape_instance = shape_list[it];
        auto& shape = shape_instance->children[0];
        switch (shape_instance->shape_type) {
        case loco_t::shape_type_t::sprite: {
          fan::graphics::shape_serialize(shape, &ostr);
          fan::write_to_string(ostr, shape_instance->id);
          fan::write_to_string(ostr, shape_instance->group_id);
          fan::write_to_string(ostr, shape_instance->shape_data.sprite.image_name);
          break;
        }
        case loco_t::shape_type_t::unlit_sprite: {
          fan::graphics::shape_serialize(shape, &ostr);
          fan::write_to_string(ostr, shape_instance->id);
          fan::write_to_string(ostr, shape_instance->group_id);
          fan::write_to_string(ostr, shape_instance->shape_data.sprite.image_name);
          break;
        }
        case loco_t::shape_type_t::rectangle: {
          fan::graphics::shape_serialize(shape, &ostr);
          fan::write_to_string(ostr, shape_instance->id);
          fan::write_to_string(ostr, shape_instance->group_id);
          break;
        }
        case loco_t::shape_type_t::light: {
          fan::graphics::shape_serialize(shape, &ostr);
          fan::write_to_string(ostr, shape_instance->id);
          fan::write_to_string(ostr, shape_instance->group_id);
          break;
        }
        default: {
          fan::print("unimplemented shape type");
          break;
        }
        }
        it = it.Next(&shape_list);
      }
      fan::io::file::write(filename, ostr, std::ios_base::binary);
    }
    else if (format == "json") {
      fan::json ostr;
      ostr["version"] = current_version;
      fan::json shapes = fan::json::array();
      auto it = shape_list.GetNodeFirst();
      while (it != shape_list.dst) {
        auto& shape_instance = shape_list[it];
        auto& shape = shape_instance->children[0];

        fan::json shape_json;

        switch (shape_instance->shape_type) {
        case loco_t::shape_type_t::sprite: {
          fan::graphics::shape_serialize(shape, &shape_json);
          shape_json["id"] = shape_instance->id;
          shape_json["group_id"] = shape_instance->group_id;
          shape_json["image_name"] = shape_instance->shape_data.sprite.image_name;
          break;
        }
        case loco_t::shape_type_t::unlit_sprite: {
          fan::graphics::shape_serialize(shape, &shape_json);
          shape_json["id"] = shape_instance->id;
          shape_json["group_id"] = shape_instance->group_id;
          shape_json["image_name"] = shape_instance->shape_data.sprite.image_name;
          break;
        }
        case loco_t::shape_type_t::rectangle: {
          fan::graphics::shape_serialize(shape, &shape_json);
          shape_json["id"] = shape_instance->id;
          shape_json["group_id"] = shape_instance->group_id;
          break;
        }
        case loco_t::shape_type_t::light: {
          fan::graphics::shape_serialize(shape, &shape_json);
          shape_json["id"] = shape_instance->id;
          shape_json["group_id"] = shape_instance->group_id;
          break;
        }
        default: {
          fan::print("unimplemented shape type");
          break;
        }
        }
        shapes.push_back(shape_json);
        it = it.Next(&shape_list);
      }
      ostr["shapes"] = shapes;
      fan::io::file::write(filename, ostr.dump(2), std::ios_base::binary);
    }
    else {
      fan::print("invalid format:" +format);
    }
    fan::print("file saved to:" + filename);
  }

  void load_tp(fgm_t::shape_list_NodeData_t& node) {
    loco_t::texturepack_t::ti_t ti;
    if (texturepack.qti(node->shape_data.sprite.image_name, &ti)) {
      fan::print_no_space("failed to load texture:", node->shape_data.sprite.image_name);
    }
    else {
      auto& data = texturepack.get_pixel_data(ti.pack_id);
      node->children[0].load_tp(&ti);
    }
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

    previous_file_name = filename;

    auto format = filename.substr(filename.find_last_of(".") + 1);
    if (format == "fmm") {

      fan::string in;
      fan::io::file::read(filename, &in);
      uint64_t off = 0;
      auto version = fan::read_data<decltype(current_version)>(in, off);
      if (version != current_version) {
        fan::print("invalid file version, file:", version, "current:",  current_version);
        return;
      }

      fan::graphics::shape_deserialize_t iterator;
      loco_t::shape_t shape;
      int i = 0;
      while (iterator.iterate(in, &shape)) {
        auto it = shape_list.NewNodeLast();
        auto& node = shape_list[it];
        switch (shape.get_shape_type()) {
        case loco_t::shape_type_t::sprite: {
          node = new fgm_t::shapes_t::global_t(
            loco_t::shape_type_t::sprite,
            this,
            shape
          );
          node->id = fan::read_data<fan::string>(in, iterator.data.offset);
          node->group_id = fan::read_data<uint32_t>(in, iterator.data.offset);
          node->shape_data.sprite.image_name = fan::read_data<fan::string>(in, iterator.data.offset);

          load_tp(node);
          break;
        }
        case loco_t::shape_type_t::unlit_sprite: {
          node = new fgm_t::shapes_t::global_t(
            loco_t::shape_type_t::unlit_sprite,
            this,
            shape
          );
          node->id = fan::read_data<fan::string>(in, iterator.data.offset);
          node->group_id = fan::read_data<uint32_t>(in, iterator.data.offset);
          node->shape_data.sprite.image_name = fan::read_data<fan::string>(in, iterator.data.offset);

          load_tp(node);
          break;
        }
        case loco_t::shape_type_t::rectangle: {
          node = new fgm_t::shapes_t::global_t(
            loco_t::shape_type_t::rectangle,
            this,
            shape
          );
          node->id = fan::read_data<fan::string>(in, iterator.data.offset);
          node->group_id = fan::read_data<uint32_t>(in, iterator.data.offset);
          break;
        }
        case loco_t::shape_type_t::light: {
          node = new fgm_t::shapes_t::global_t(
            loco_t::shape_type_t::light,
            this,
            shape
          );
          node->push_child(fan::graphics::circle_t{ {
            .position = shape.get_position(),
            .radius = shape.get_size().x,
            .color = shape.get_color(),
            .blending = true
          } });
          node->id = fan::read_data<fan::string>(in, iterator.data.offset);
          node->group_id = fan::read_data<uint32_t>(in, iterator.data.offset);
          break;
        }
        default: {
          fan::print("unimplemented shape type");
          break;
        }
        }
      }
    }
    else if (format == "json") {
      fan::string in;
      fan::io::file::read(filename, &in);
      fan::json json_in = nlohmann::json::parse(in);
      auto version = json_in["version"].get<decltype(current_version)>();
      if (version != current_version) {
        fan::print("invalid file version, file:", version, "current:", current_version);
        return;
      }
      fan::graphics::shape_deserialize_t iterator;
      loco_t::shape_t shape;
      int i = 0;
      while (iterator.iterate(json_in["shapes"], &shape)) {
        auto it = shape_list.NewNodeLast();
        auto& node = shape_list[it];
        switch (shape.get_shape_type()) {
        case loco_t::shape_type_t::sprite: {
          node = new fgm_t::shapes_t::global_t(
            loco_t::shape_type_t::sprite,
            this,
            shape
          );
          const auto& shape_json = *(iterator.data.it - 1);
          node->id = shape_json["id"].get<fan::string>();
          node->group_id = shape_json["group_id"].get<uint32_t>();
          node->shape_data.sprite.image_name = shape_json["image_name"].get<fan::string>();

          load_tp(node);
          break;
        }
        case loco_t::shape_type_t::unlit_sprite: {
          node = new fgm_t::shapes_t::global_t(
            loco_t::shape_type_t::unlit_sprite,
            this,
            shape
          );
          const auto& shape_json = *(iterator.data.it - 1);
          node->id = shape_json["id"].get<fan::string>();
          node->group_id = shape_json["group_id"].get<uint32_t>();
          node->shape_data.sprite.image_name = shape_json["image_name"].get<fan::string>();

          load_tp(node);
          break;
        }
        case loco_t::shape_type_t::rectangle: {
          node = new fgm_t::shapes_t::global_t(
            loco_t::shape_type_t::rectangle,
            this,
            shape
          );
          const auto& shape_json = *(iterator.data.it - 1);
          node->id = shape_json["id"].get<fan::string>();
          node->group_id = shape_json["group_id"].get<uint32_t>();
          break;
        }
        case loco_t::shape_type_t::light: {
          node = new fgm_t::shapes_t::global_t(
            loco_t::shape_type_t::light,
            this,
            shape
          );
          node->push_child(fan::graphics::circle_t{ {
            .position = shape.get_position(),
            .radius = shape.get_size().x,
            .color =  shape.get_color(),
            .blending = true
          } });
          const auto& shape_json = *(iterator.data.it - 1);
          node->id = shape_json["id"].get<fan::string>();
          node->group_id = shape_json["group_id"].get<uint32_t>();
          break;
        }
        default: {
          fan::print("unimplemented shape type");
          break;
        }
        }
      }
      }
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
    uint16_t selected_shape_type = loco_t::shape_type_t::invalid;
    shapes_t::global_t* current_shape = nullptr;
    shape_list_t shape_list;

    f32_t current_z = 0;
    uint32_t current_id = 0;

    loco_t::texturepack_t texturepack;

    fan::function_t<void()> close_cb = [] {};
  };
