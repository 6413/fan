#pragma once

struct ftme_t {

  // can have like sensor, etc
  struct mesh_property_t : fan::any_type_wrap_t<uint8_t> {
    using fan::any_type_wrap_t<uint8_t>::any_type_wrap_t;
    static constexpr uint8_t none = 0;
    static constexpr uint8_t collision_solid = 1;
  };

  static constexpr auto editor_str = "Editor";
  static constexpr auto editor_settings_str = "Editor settings";
  static constexpr auto properties_str = "Properties";

  static constexpr int max_path_input = 40;

  static constexpr fan::vec2 default_button_size{100, 30};
  static constexpr fan::vec2 tile_viewer_sprite_size{64, 64};
  static constexpr fan::color highlighted_tile_color = fan::color(0.5, 0.5, 1);

  static constexpr f32_t scroll_speed = 0.05;

  fan::string file_name = "file.ftme";

  struct shapes_t {
    struct global_t : fan::graphics::vfi_root_t, fan::graphics::imgui_element_t {

      struct shape_data {
        struct cell_t {
          uint8_t mesh_property = mesh_property_t::none;
          uint64_t image_hash = 0;
          uint8_t color_idx = 0;
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
                children[0].set_color(highlighted_tile_color);
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

  #include "common.h"

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
      uint32_t y = 0;
      uint8_t offset = 0; // creates grid pattern
      for (auto& i : map_tiles) {
        uint32_t x = 0;
        for (auto& j : i) {
          uint32_t idx = (y * map_size.x + x + ((map_size.x & 1 & y & 1) ? 0 : offset)) & 1;
          if (j) {
            j->set_position(fan::vec3(tile_size + fan::vec2(x, y) * tile_size * 2, 0));
            j->set_size(tile_size);
            if (j->children.size()) {
              j->children[0].set_color(1);
              if (j->children[0].get_image() == &texture_gray_shades[0] || j->children[0].get_image() == &texture_gray_shades[1]) {
                j->children[0].set_image(&texture_gray_shades[idx]);
              }
            }
            ++x;
            continue;
          }
          j = std::make_unique<shapes_t::global_t>(this, fan::graphics::sprite_t{{
            .position = fan::vec3(tile_size + fan::vec2(x, y) * tile_size * 2, 0),
            .size = tile_size,
            .image = &texture_gray_shades[idx]
          }});
          j->shape_data.cell.color_idx = idx;
          ++x;
        }
        ++y;
        offset = ((offset + 1) & 1);
      }
      current_tile = nullptr;
      erasing = false;
    }
  }

  void reset_map() {
    map_tiles.clear();
    resize_map();
  }

  void open(const fan::string& texturepack_name) {
    texturepack.open_compiled(texturepack_name);

    gloco->get_window()->add_mouse_move_callback([this](const auto& d) {
      if (viewport_settings.move) {
        fan::vec2 viewport_size = gloco->default_camera->viewport.get_size();
        fan::vec2 scaler = (viewport_size / viewport_settings.zoom / viewport_size);
        gloco->default_camera->camera.set_camera_position(viewport_settings.pos - ((d.position -
          viewport_settings.offset) * scaler));
      }
    });

    gloco->get_window()->add_buttons_callback([this](const auto& d) {
      if (ImGui::IsAnyItemActive()) {
        return;
      }
      if (!editor_settings.hovered) {
        return;
      }
      

      {// handle camera movement
        f32_t old_zoom = viewport_settings.zoom;

        auto set_camera_center = [&] {
          fan::vec2 scaler = (viewport_settings.size / old_zoom / viewport_settings.size);
          fan::vec2 scaler1 = (viewport_settings.size / viewport_settings.zoom / viewport_settings.size);
          viewport_settings.pos -= scaler - scaler1;
          //viewport_settings.pos += (viewport_settings.size * viewport_settings.zoom);
          gloco->default_camera->camera.set_camera_position(viewport_settings.pos);
        };

        switch (d.button) {
          case fan::mouse_middle: { break;}
          case fan::mouse_scroll_up: {
            viewport_settings.zoom += scroll_speed; 
            set_camera_center();
            return; 
          }
          case fan::mouse_scroll_down: { 
            viewport_settings.zoom -= scroll_speed; 
            set_camera_center();
            return; 
          }
          default: {return;} //?
        };
        viewport_settings.move = (bool)d.state;
        fan::vec2 old_pos = viewport_settings.pos;
        viewport_settings.offset = gloco->get_mouse_position();
        viewport_settings.pos = gloco->default_camera->camera.get_camera_position();

      }// handle camera movement
   });

    gloco->get_window()->add_keys_callback([this](const auto& d) {
      if (d.state != fan::keyboard_state::press) {
        return;
      }

      switch (d.key) {
        case fan::key_delete: {
          if (gloco->get_window()->key_pressed(fan::key_left_control)) {
            reset_map();
          }
          break;
        }
      }
    });

    // transparent pattern
    texture_gray_shades[0].create(fan::color::rgb(60, 60, 60, 255), fan::vec2(1, 1));
    texture_gray_shades[1].create(fan::color::rgb(40, 40, 40, 255), fan::vec2(1, 1));

    viewport_settings.size = 0;

    resize_map();

    texturepack_images.reserve(texturepack.texture_list.size());

    texturepack.iterate_loaded_images([this](auto& image, uint32_t pack_id) {
      image_info_t ii;
      ii.ti = loco_t::texturepack_t::ti_t{
        .pack_id = pack_id,
        .position = image.position,
        .size = image.size,
        .image = &texturepack.get_pixel_data(pack_id).image
      };

      ii.ti.position /= ii.ti.image->size;
      ii.ti.size /= ii.ti.image->size;
      ii.image_hash = image.hash;

      texturepack_images.push_back(ii);
    });
  }
  void close() {
    texturepack.close();
  }

  void open_properties(shapes_t::global_t* shape, const fan::vec2& editor_size) {

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

      if (ImGui::Begin(editor_str, nullptr)) {
        fan::vec2 window_size = gloco->get_window()->get_size();
        fan::vec2 viewport_size = ImGui::GetWindowSize();
        fan::vec2 viewport_pos = fan::vec2(ImGui::GetWindowPos() + fan::vec2(0, ImGui::GetFontSize() + ImGui::GetStyle().FramePadding.y * 2));
        fan::vec2 offset = viewport_size - viewport_size / viewport_settings.zoom;
        gloco->default_camera->camera.set_ortho(
          fan::vec2(0, viewport_size.x),
          fan::vec2(0, viewport_size.y)
        );

        gloco->default_camera->camera.set_camera_zoom(viewport_settings.zoom);
        gloco->default_camera->viewport.set(viewport_pos, viewport_size, window_size);
        editor_size = ImGui::GetContentRegionAvail();
        viewport_settings.size = editor_size;
      }

      editor_settings.hovered = ImGui::IsWindowHovered();

      // add texture
      if (editor_settings.hovered) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
          if (current_tile != nullptr && current_tile_image.ti.image != nullptr && current_tile->children.size()) {
            gloco->shapes.sprite.set(
              current_tile->children[0],
              &loco_t::shapes_t::sprite_t::vi_t::tc_position,
              current_tile_image.ti.position
            );
            gloco->shapes.sprite.set(
              current_tile->children[0],
              &loco_t::shapes_t::sprite_t::vi_t::tc_size,
              current_tile_image.ti.size
            );
            current_tile->children[0].set_color(1);
            current_tile->children[0].set_image(current_tile_image.ti.image);
            current_tile->shape_data.cell.image_hash = current_tile_image.image_hash;
          }
        }
        //remove texture
        if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
          if (current_tile != nullptr && current_tile->children.size()) {
            gloco->shapes.sprite.set(
              current_tile->children[0],
              &loco_t::shapes_t::sprite_t::vi_t::tc_position,
              0
            );
            gloco->shapes.sprite.set(
              current_tile->children[0],
              &loco_t::shapes_t::sprite_t::vi_t::tc_size,
              1
            );
            current_tile->children[0].set_color(highlighted_tile_color);
            current_tile->children[0].set_image(&texture_gray_shades[current_tile->shape_data.cell.color_idx]);
          }
        }
      }

      ImGui::End();

      if (ImGui::Begin(editor_settings_str, nullptr)) {
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
          make_setting_ii2("tile size", tile_size, [this] { resize_map(); });

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

      if (ImGui::Begin(properties_str, nullptr, ImGuiWindowFlags_HorizontalScrollbar)) {

        {
          int images_per_row = 4;
          static uint32_t current_image_idx = -1;
          for (uint32_t i = 0; i < texturepack_images.size(); i++) {
            auto& node = texturepack_images[i];

            bool selected = false;
            if (current_image_idx == i) {
              ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(1.0f, 1.0f, 0.0f, 0.5f));
              ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(1.0f, 1.0f, 0.0f, 0.5f));
              ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 1.0f, 0.0f, 0.5f));
              selected = true;
            }

            if (ImGui::ImageButton(
              (fan::string("##ibutton") + std::to_string(i)).c_str(),
              (void*)(intptr_t)*node.ti.image->get_texture(),
              tile_viewer_sprite_size,
              node.ti.position,
              node.ti.position + node.ti.size
            )) {
              current_image_idx = i;
              current_tile_image = node;
            }

            if (selected) {
              ImGui::PopStyleColor(3);
            }

            if ((i + 1) % images_per_row != 0)
              ImGui::SameLine();
          }
        }

        if (current_tile != nullptr) {
          //open_properties(current_tile, editor_size);
        }
      }

      ImGui::End();
  });

  /*
  * header
  header version 4 byte
  map size 8 byte
  tile size 8 byte
  struct size x byte
  shape data{
    ...
  }
  */
  void fout(const fan::string& filename) {
    fan::string ostr;
    ostr.append((char*)&current_version, sizeof(current_version));
    ostr.append((char*)map_size.data(), sizeof(map_size));
    ostr.append((char*)tile_size.data(), sizeof(tile_size));

    for (auto& i : map_tiles) {
      for (auto& j : i) {
        if (j->children.empty() || 
          j->children[0].get_image() == &texture_gray_shades[0] ||
          j->children[0].get_image() == &texture_gray_shades[1] ||
          j->shape_data.cell.image_hash == 0
          ) {
          continue;
        }

        fan::mp_t<current_version_t::shapes_t> shapes;

        shapes.iterate([&]<auto i0, typename T>(T & l) {
          fan::mp_t<T> shape;
          shape.init(j.get());

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
      }
    }
    fan::io::file::write(filename, ostr, std::ios_base::binary);
    fan::print("file saved to:" + filename);
  }

    /*
  * header
  header version 4 byte
  map size 8 byte
  tile size 8 byte
  struct size x byte
  shape data{
    ...
  }
  */
  void fin(const fan::string& filename) {
    #include _FAN_PATH(graphics/gui/tile_map_editor/loader_versions/1.h)
  }

  fan::vec2ui map_size{32, 32};
  fan::vec2ui tile_size{32, 32};

  event_type_e event_type = event_type_e::none;
  loco_t::shape_type_t selected_shape_type = loco_t::shape_type_t::invalid;
  shapes_t::global_t* current_tile = nullptr;

  struct image_info_t {
    loco_t::texturepack_t::ti_t ti;
    uint64_t image_hash;
  };

  image_info_t current_tile_image;

  uint32_t current_id = 0;
  std::vector<std::vector<std::unique_ptr<shapes_t::global_t>>> map_tiles;

  loco_t::texturepack_t texturepack;
  // tile pattern
  loco_t::image_t texture_gray_shades[2];

  fan::function_t<void()> close_cb = [] {};

  std::vector<image_info_t> texturepack_images;

  struct {
    f32_t zoom = 1;
    bool move = false;
    fan::vec2 pos = 0;
    fan::vec2 size = 0;
    fan::vec2 offset = 0;
  }viewport_settings;

  struct {
    bool hovered = false;
  }editor_settings;

  // very bad fix to prevent mouse move cb when erasing vfi
  bool erasing = false;
};
