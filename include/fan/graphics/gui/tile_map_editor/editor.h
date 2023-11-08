#pragma once

struct ftme_t {

  static constexpr int current_version = 001;

  static constexpr auto editor_str = "Editor";
  static constexpr auto create_str = "Create";
  static constexpr auto properties_str = "Properties";

  static constexpr int max_path_input = 40;

  static constexpr fan::vec2 default_button_size{100, 30};

  fan::string file_name = "file.ftme";

  // can have like sensor, etc
  struct mesh_property_t : fan::any_type_wrap_t<uint8_t> {
    using fan::any_type_wrap_t<uint8_t>::any_type_wrap_t;
    static constexpr uint8_t none = 0;
    static constexpr uint8_t collision_solid = 1;
  };

  struct shapes_t {
    struct global_t : fan::graphics::vfi_root_t, fan::graphics::imgui_element_t {

      struct shape_data {
        struct cell_t {
          mesh_property_t mesh_property = mesh_property_t::none;
        }cell;
      }shape_data;

      global_t() = default;

      template <typename T>
      global_t(ftme_t* root, const T& obj) : fan::graphics::imgui_element_t() {
        vfi_root_t::move_and_resize_auto = false;
        T temp = std::move(obj);
        loco_t::shapes_t::vfi_t::properties_t vfip;
        vfip.shape.rectangle->position = temp.get_position();
        vfip.shape.rectangle->position.z += 1;
        vfip.shape.rectangle->size = temp.get_size();
        vfip.mouse_button_cb = [root, this](const auto& d) -> int {
          //root->event_type = event_type_e::move;
          //root->current_shape = this;
          return 0;
        };
        vfip.mouse_move_cb = [root, this](const auto& d) -> int {
          if (root->erasing) {
            return 0;
          }

          if (d.mouse_stage == loco_t::shapes_t::vfi_t::mouse_stage_e::inside) {
            if (root->current_tile != this) {
              if (root->current_tile != nullptr) {
                root->current_tile->children[0].set_color(fan::color(1, 1, 1));
              }
              if (children.size()) {
                children[0].set_color(fan::color(0.5, 0.5, 1));
                root->current_tile = this;
              }
            }
          }
          //root->event_type = event_type_e::move;
          //root->current_shape = this;
          return 0;
        };
        vfi_root_t::set_root(vfip);
        vfi_root_t::push_child(std::move(temp));
        //root->current_shape = this;
      }

      // global data
      fan::string id;
      uint32_t group_id = 0;
    };
  };

  enum class event_type_e {
    none,
    add,
    remove
  };

  void resize_map() {
    if (map_size == 0) {
      map_tiles.clear();
      return;
    }

    map_tiles.resize(map_size.y);
    for (auto& i : map_tiles) {
      i.resize(map_size.x);
    }

    {
      erasing = true;
      static loco_t::image_t* textures[2]{&texture_light_gray, &texture_dark_gray};
      uint32_t y = 0;
      uint8_t offset = 0; // creates grid pattern
      for (auto& i : map_tiles) {
        uint32_t x = 0;
        for (auto& j : i) {
          j = std::make_unique<shapes_t::global_t>(this, fan::graphics::sprite_t{{
              .position = fan::vec3(tile_size + fan::vec2(x, y) * tile_size * 2, 0),
              .size = tile_size,
              .image = textures[(y * map_size.x + x + offset) & 1]
            }});
          ++x;
        }
        ++y;
        offset = ((offset + 1) & 1);
      }
      current_tile = nullptr;
      erasing = false;
    }
  }

  void open(const fan::string& texturepack_name) {
    texturepack.open_compiled(texturepack_name);

    gloco->get_window()->add_mouse_move_callback([this](const auto& d) {
      if (viewport_settings.move) {
        gloco->default_camera->camera.set_camera_position(viewport_settings.pos - (d.position -
          viewport_settings.offset) * viewport_settings.zoom);
        fan::print("o", d.position);
      }
    });

    gloco->get_window()->add_buttons_callback([this](const auto& d) {
      if (ImGui::IsAnyItemActive()) {
        return;
      }

      {// handle camera movement
        switch (d.button) {
          case fan::mouse_middle: { break;}
          case fan::mouse_scroll_up: { viewport_settings.zoom -= 0.1; return; }
          case fan::mouse_scroll_down: { viewport_settings.zoom += 0.1; return; }
          default: {return;} //?
        };
        viewport_settings.move = (bool)d.state;
        viewport_settings.pos = gloco->default_camera->camera.get_camera_position();
        viewport_settings.offset = gloco->get_mouse_position();
      }// handle camera movement
   });

    gloco->get_window()->add_keys_callback([this](const auto& d) {
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

    // transparent pattern
    texture_light_gray.create(fan::color::rgb(60, 60, 60, 255), fan::vec2(1, 1));
    texture_dark_gray.create(fan::color::rgb(40, 40, 40, 255), fan::vec2(1, 1));

    viewport_settings.pos = map_size * tile_size;
    gloco->default_camera->camera.set_camera_position(viewport_settings.pos);

    resize_map();
  }
  void close() {
    texturepack.close();
  }

  void open_properties(shapes_t::global_t* shape, const fan::vec2& editor_size) {

   /* fan::string shape_str = fan::string("Shape name:") + gloco->shape_names[shape->children[0]->shape_type];
    ImGui::Text(shape_str.c_str());

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
    }*/
  }

  void push_shape(loco_t::shape_type_t shape_type, const fan::vec2& pos) {
  //  auto nr = shape_list.NewNodeLast();

   /* static fan::mp_t<current_version_t::shapes_t> mp;
    mp.iterate([&]<auto i, typename T> (T & v) {
      if (shape_type == v.shape_type) {
        shape_list[nr] = new shapes_t::global_t{this, typename T::type_t{{
            .position = pos,
            .size = 100
          }}};
      }
    });*/
  }

  fan::graphics::imgui_element_t main_view =
    fan::graphics::imgui_element_t(
    [&] {
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

      fan::vec2 editor_size;

      if (ImGui::Begin(editor_str, nullptr, ImGuiWindowFlags_DockNodeHost)) {
        fan::vec2 window_size = gloco->get_window()->get_size();
        fan::vec2 viewport_size = ImGui::GetWindowSize();
        fan::vec2 viewport_pos = fan::vec2(ImGui::GetWindowPos() + fan::vec2(0, ImGui::GetFontSize() + ImGui::GetStyle().FramePadding.y * 2));
        fan::vec2 offset = viewport_size - viewport_size * viewport_settings.zoom;
        gloco->default_camera->camera.set_ortho(
          fan::vec2(offset.x, viewport_size.x - offset.x),
          fan::vec2(offset.y, viewport_size.y - offset.y)
        );
        gloco->default_camera->camera.set_camera_zoom(viewport_settings.zoom);
        gloco->default_camera->viewport.set(viewport_pos, viewport_size, window_size);
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

      if (ImGui::Begin(create_str, nullptr, ImGuiWindowFlags_DockNodeHost)) {
        {
          static auto make_setting_ii2 = [&](const char* title, auto& value, auto todo){
            auto i = value;
            if (ImGui::InputInt2(title, (int*)i.data())) {
              if (ImGui::IsItemDeactivatedAfterEdit()) {
                value = i;
                todo();
              }
            }
          };

          make_setting_ii2("map size", map_size, [this] { resize_map(); });
          make_setting_ii2("tile size", tile_size, [] {});

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
            
            return;
          }
        }
      }

      ImGui::End();

      if (ImGui::Begin(properties_str, nullptr, ImGuiWindowFlags_DockNodeHost)) {
        if (current_tile != nullptr) {
          open_properties(current_tile, editor_size);
        }
      }

      ImGui::End();
  });

  void fout(const fan::string& filename) {

  }

  void fin(const fan::string& filename) {

  }

  void invalidate_current() {
    current_tile = nullptr;
    selected_shape_type = loco_t::shape_type_t::invalid;
  }

  void erase_current() {
   /* if (current_shape == nullptr) {
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
    }*/
  }

  fan::vec2ui map_size{32, 32};
  fan::vec2ui tile_size{32, 32};

  event_type_e event_type = event_type_e::none;
  loco_t::shape_type_t selected_shape_type = loco_t::shape_type_t::invalid;
  shapes_t::global_t* current_tile = nullptr;

  uint32_t current_id = 0;
  std::vector<std::vector<std::unique_ptr<shapes_t::global_t>>> map_tiles;

  loco_t::texturepack_t texturepack;
  // tile pattern
  loco_t::image_t texture_light_gray, texture_dark_gray;

  fan::function_t<void()> close_cb = [] {};

  struct {
    f32_t zoom = 1;
    bool move = false;
    fan::vec2 pos;
    fan::vec2 offset = 0;
  }viewport_settings;
  // very bad fix to prevent mouse move cb when erasing vfi
  bool erasing = false;
};
