#pragma once

#include <fan/graphics/file_dialog.h>


struct fgm_t {

  static constexpr f32_t scroll_speed = 1.2;

  loco_t::shape_t xy_lines[2];

  void open(const fan::string& texturepack_name) {

    camera.camera = gloco->open_camera(
      fan::vec2(0, 1),
      fan::vec2(0, 1)
    );

    camera.viewport = gloco->open_viewport(
      fan::vec2(0),
      fan::vec2(1)
    );

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
    gloco->input_action.add_keycombo({ fan::input::key_left_control, fan::input::key_space }, "toggle_content_browser");
    gloco->input_action.add_keycombo({ fan::input::key_left_control, fan::input::key_f }, "set_windowed_fullscreen");

    gloco->window.add_mouse_move_callback([this](const auto& d) {
      if (viewport_settings.move) {
        fan::vec2 move_off = (d.position - viewport_settings.offset) / viewport_settings.zoom;
        gloco->camera_set_position(camera.camera, viewport_settings.pos - move_off);
      }
    });

    gloco->window.add_buttons_callback([this](const auto& d) {

      f32_t old_zoom = viewport_settings.zoom;

      switch (d.button) {
      case fan::mouse_middle: {
        viewport_settings.move = (bool)d.state;
        fan::vec2 old_pos = viewport_settings.pos;
        viewport_settings.offset = gloco->get_mouse_position();
        viewport_settings.pos = gloco->camera_get_position(camera.camera);
        break;
      }
      case fan::mouse_scroll_up: {
        if (viewport_settings.editor_hovered) {
          viewport_settings.zoom *= scroll_speed;
        }
        return;
      }
      case fan::mouse_scroll_down: {
        if (viewport_settings.editor_hovered) {
          viewport_settings.zoom /= scroll_speed;
        }
        return;
      }
      }
    });
    xy_lines[0] = fan::graphics::line_t{ {
      .camera = &camera,
      .src = fan::vec2(-0xffffff, 0),
      .dst = fan::vec2(0xffffff, 0),
      .color = fan::colors::red / 2
    } };

    xy_lines[1] = fan::graphics::line_t{ {
        .camera = &camera,
        .src = fan::vec2(0, -0xffffff),
        .dst = fan::vec2(0, 0xffffff),
        .color = fan::colors::green / 2
    } };

    drag_select = fan::graphics::rectangle_t{ {
        .camera = &camera,
        .position = fan::vec3(0, 0, 0xffff - 0xff),
        .size = 0,
        .color = fan::color::hex(0x3eb9ff44),
        .blending = true
    } };

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
      global_t(uint16_t shape_type, fgm_t* fgm, const T& obj, bool shape_add = true) : fan::graphics::imgui_element_t() {
        T temp = obj;
        this->shape_type = shape_type;
        typename loco_t::vfi_t::properties_t vfip;
        vfip.shape.rectangle->position = temp.get_position();
        vfip.shape.rectangle->position.z += shape_add ? fgm->current_z++ : 0;
        vfip.shape.rectangle->size = temp.get_size();
        vfip.shape.rectangle->angle = 0;
        vfip.shape.rectangle->rotation_point = 0;
        vfip.shape.rectangle->camera = fgm->camera.camera;
        vfip.shape.rectangle->viewport = fgm->camera.viewport;
        vfip.mouse_button_cb = [fgm, this](const auto& d) -> int {
          fgm->event_type = event_type_e::move;
          fgm->current_shape = this;
          return 0;
          };
        fan::graphics::vfi_root_t::set_root(vfip);
        if (shape_add) {
          temp.set_position(fan::vec3(fan::vec2(temp.get_position()), fgm->current_z - 1));
        }
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
#include <BLL/BLL.h>


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
    ImGui::Text(shape_str.c_str());

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
      auto current_image = shape->children[0].get_image();
      fan::vec2 uv0 = shape->children[0].get_tc_position(), uv1 = shape->children[0].get_tc_size();
      uv1 += uv0;
      ImGui::Image(current_image, fan::vec2(64), uv0, uv1);
      if (ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("CONTENT_BROWSER_ITEM")) {
          const wchar_t* path = (const wchar_t*)payload->Data;
          if (current_image != gloco->default_texture) {
            gloco->image_unload(current_image);
          }
          shape->children[0].set_image(gloco->image_load(std::filesystem::absolute(std::filesystem::path(path)).string()));
          shape->children[0].set_tc_position(0);
          shape->children[0].set_tc_size(1);
          //fan::print(std::filesystem::path(path));
        }
        ImGui::EndDragDropTarget();
      }

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

  shape_list_t::nr_t push_shape(uint16_t shape_type, const fan::vec2& pos) {
    auto nr = shape_list.NewNodeLast();

    switch (shape_type) {
    case loco_t::shape_type_t::sprite: {
      shape_list[nr] = new shapes_t::global_t{
        loco_t::shape_type_t::sprite,
        this, fan::graphics::sprite_t{{
          .camera = &camera,
          .position = pos,
          .size = 128
        }} };
      break;
    }
    case loco_t::shape_type_t::unlit_sprite: {
      shape_list[nr] = new shapes_t::global_t{
        loco_t::shape_type_t::unlit_sprite,
        this, fan::graphics::unlit_sprite_t{{
          .camera = &camera,
          .position = pos,
          .size = 128
        }} };
      break;
    }
    case loco_t::shape_type_t::rectangle: {
      shape_list[nr] = new shapes_t::global_t{
        loco_t::shape_type_t::rectangle,
        this, fan::graphics::rectangle_t{{
          .camera = &camera,
          .position = pos,
          .size = 128
        }} };
      break;
    }
    case loco_t::shape_type_t::light: {
      shape_list[nr] = new shapes_t::global_t{
        loco_t::shape_type_t::light,
        this, fan::graphics::light_t{{
          .camera = &camera,
          .position = pos,
          .size = 128
        }} };
        shape_list[nr]->push_child(fan::graphics::circle_t{ {
          .camera = &camera,
          .position = fan::vec3(pos, current_z),
          .radius = 100,
          .color = fan::color(1, 1, 1, 0.5),
          .blending = true
        } });
      break;
    }
    }
    return nr;
  }

  void DrawTextBottomRight(const char* text, uint32_t reverse_yoffset = 0)
  {
    // Retrieve the current window draw list
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    // Retrieve the current window position and size
    ImVec2 window_pos = ImGui::GetWindowPos();
    ImVec2 window_size = ImGui::GetWindowSize();

    // Calculate the size of the text
    ImVec2 text_size = ImGui::CalcTextSize(text);

    // Calculate the position to draw the text (bottom-right corner)
    ImVec2 text_pos;
    text_pos.x = window_pos.x + window_size.x - text_size.x - ImGui::GetStyle().WindowPadding.x;
    text_pos.y = window_pos.y + window_size.y - text_size.y - ImGui::GetStyle().WindowPadding.y;

    text_pos.y -= reverse_yoffset * ImGui::GetTextLineHeightWithSpacing();

    // Draw the text at the calculated position
    draw_list->AddText(text_pos, IM_COL32(255, 255, 255, 255), text);
  }

  fan::vec2 get_mouse_position() {
    auto& style = ImGui::GetStyle();
    fan::vec2 pos = fan::vec2((ImGui::GetMousePos() - viewport_settings.start_pos + style.WindowPadding) - viewport_settings.size / 2);
    pos = fan::vec2(viewport_settings.pos) + pos / viewport_settings.zoom;
    return pos;
  }

  fan::graphics::imgui_element_t main_view =
    fan::graphics::imgui_element_t(
      [&] {

        if (ImGui::IsMouseClicked(0) && !fan::graphics::vfi_root_t::moving_object) {
          drag_start = get_mouse_position();
        }
        else if (ImGui::IsMouseDown(0) && !fan::graphics::vfi_root_t::moving_object) {
          fan::vec2 size = get_mouse_position() - drag_start;

          drag_select.set_position(drag_start + size / 2);
          drag_select.set_size(size / 2);
        }
        else if (ImGui::IsMouseReleased(0) && !fan::graphics::vfi_root_t::moving_object && 
          (drag_select.get_size().x >= 1 && drag_select.get_size().y >= 1)) {
          auto it = shape_list.GetNodeFirst();
          int i = 0;
          fan::graphics::vfi_root_t::selected_objects.clear();
          while (it != shape_list.dst) {
            auto& shape = shape_list[it];
            if (fan_2d::collision::rectangle::check_collision(
              drag_select.get_position(),
              drag_select.get_size(),
              shape->children[0].get_position(),
              shape->children[0].get_size()
            )) {
              shape->create_highlight();
              fan::graphics::vfi_root_t::selected_objects.push_back(shape);
            }
            else {
              shape->disable_highlight();
            }
            it = it.Next(&shape_list);
          }

          drag_select.set_size(0);
        }

        fan::vec2 editor_size;

        if (gloco->input_action.is_active("set_windowed_fullscreen")) {
          gloco->window.set_windowed_fullscreen();
        }

        if (gloco->input_action.is_active("toggle_content_browser")) {
          render_content_browser = !render_content_browser;
        }

        if (ImGui::BeginMainMenuBar()) {

          ImGui::EndMainMenuBar();
        }

        


        if (render_content_browser) {
          content_browser.render();
        }

        if (ImGui::Begin(editor_str, nullptr, ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoBackground)) {
          fan::vec2 window_size = gloco->window.get_size();
          fan::vec2 viewport_size = ImGui::GetWindowSize();
          fan::vec2 ratio = viewport_size / viewport_size.max();
          fan::vec2 s = viewport_size;

          if (ImGui::IsMouseClicked(0)) {
            fan::graphics::vfi_root_t::g_ignore_mouse = !ImGui::IsWindowHovered();
          }
          viewport_settings.editor_hovered = ImGui::IsWindowHovered();

          auto& style = ImGui::GetStyle();
          fan::vec2 frame_padding = style.FramePadding;
          fan::vec2 viewport_pos = fan::vec2(0, frame_padding.y * 2);

          ImVec2 vMin = ImGui::GetWindowContentRegionMin();
          ImVec2 vMax = ImGui::GetWindowContentRegionMax();

          vMin.x += ImGui::GetWindowPos().x;
          vMin.y += ImGui::GetWindowPos().y;
          vMax.x += ImGui::GetWindowPos().x;
          vMax.y += ImGui::GetWindowPos().y;

          viewport_size = vMax - vMin + style.WindowPadding * 2;

          gloco->viewport_set(
            camera.viewport,
            vMin - style.WindowPadding,
            viewport_size,
            window_size
          );

          gloco->camera_set_ortho(
            camera.camera,
            fan::vec2(-viewport_size.x / 2, viewport_size.x / 2) / viewport_settings.zoom,
            fan::vec2(-viewport_size.y / 2, viewport_size.y / 2) / viewport_settings.zoom
          );


          viewport_settings.size = viewport_size;
          viewport_settings.start_pos = vMin;

          {
            std::string str = fan::to_string(viewport_settings.zoom * 100, 1);
            str += " %";
            DrawTextBottomRight(str.c_str(), 1);
          }

          {
            fan::vec2 cursor_pos = ((ImGui::GetMousePos() - viewport_settings.start_pos + style.WindowPadding) - viewport_settings.size / 2);
            std::string cursor_pos_str = cursor_pos.to_string();
            std::string  str = cursor_pos_str.substr(1, cursor_pos_str.size() - 2);
            DrawTextBottomRight(str.c_str(), 0);
          }
        
          ImGui::SetCursorPos(ImGui::GetCursorStartPos());
          

          content_browser.receive_drag_drop_target([&](const std::filesystem::path& fs) {

            auto file = fs.string();
            auto extension = fan::io::file::extension(file);
            if (extension == ".json") {
              fin(file);
            }
            else if (extension == ".webp") {
              auto nr = push_shape(loco_t::shape_type_t::sprite, get_mouse_position());
              shape_list[nr]->children[0].set_image(gloco->image_load(std::filesystem::absolute(fs).string()));
            }
          });

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
            push_shape(selected_shape_type, get_mouse_position());
          }
        }

        //ImPlot::PushStyleColor(ImPlotCol_PlotBg, IM_COL32(0, 0, 0, 0)); // Make plot background transparent
        //ImPlot::PushStyleColor(ImPlotCol_FrameBg, IM_COL32(0, 0, 0, 0)); // Optional: Make frame background transparent
        //ImPlot::PushStyleColor(ImPlotCol_PlotBorder, IM_COL32(0, 0, 0, 0)); // Make plot border transparent

        //ImVec2 regionAvail = ImGui::GetContentRegionAvail();
        //ImGui::SetCursorPos(ImGui::GetCursorStartPos());
        //if (ImPlot::BeginPlot("##Coordinate System", regionAvail)) {
        //  // Set plot limits
        //  ImPlot::SetupAxisLimits(ImAxis_X1, -10, 10);
        //  ImPlot::SetupAxisLimits(ImAxis_Y1, -10, 10);

        //  //float x_axis[2] = { -10.0f, 10.0f };
        //  //float y_axis[2] = { 0.0f, 0.0f };
        //  //float y_axis_x[2] = { 0.0f, 0.0f };
        //  //float y_axis_y[2] = { -10.0f, 10.0f };

        //  //ImPlot::PlotLine("X-Axis", x_axis, y_axis, 2);
        //  //ImPlot::PlotLine("Y-Axis", y_axis_x, y_axis_y, 2);


        //  ImPlot::EndPlot();
        //}
        //ImPlot::PopStyleColor(3);

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
      shape.set_camera(camera.camera);
      shape.set_viewport(camera.viewport);

      auto it = shape_list.NewNodeLast();
      auto& node = shape_list[it];
      switch (shape.get_shape_type()) {
      case loco_t::shape_type_t::sprite: {
        node = new fgm_t::shapes_t::global_t(
          loco_t::shape_type_t::sprite,
          this,
          shape,
          false
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
          shape,
          false
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
          shape,
          false
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
          shape,
          false
        );
        node->push_child(fan::graphics::circle_t{ {
          .camera = &camera,
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

    fan::graphics::camera_t camera;

    fan::graphics::imgui_content_browser_t content_browser;

    event_type_e event_type = event_type_e::none;
    uint16_t selected_shape_type = loco_t::shape_type_t::invalid;
    shapes_t::global_t* current_shape = nullptr;
    shape_list_t shape_list;

    f32_t current_z = 0;
    uint32_t current_id = 0;

    loco_t::texturepack_t texturepack;
    bool render_content_browser = true;

    fan::function_t<void()> close_cb = [] {};

    fan::vec2 drag_start;
    loco_t::shape_t drag_select;

    struct {
      f32_t zoom = 1;
      bool move = false;
      fan::vec2 pos = 0;
      fan::vec2 size = 0;
      fan::vec2 start_pos = 0;
      fan::vec2 offset = 0;
      bool editor_hovered = false;
    }viewport_settings;
  };
