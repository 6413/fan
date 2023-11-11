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

  static constexpr f32_t scroll_speed = 1.2;

  fan::string file_name = "file.ftme";

  struct tile_t {
    fan::vec3ui position;
    uint64_t image_hash;
    uint8_t mesh_property;
  };

  struct shapes_t {
    struct global_t : fan::graphics::vfi_root_custom_t<tile_t>, fan::graphics::imgui_element_t {

      using vfi_root_t = fan::graphics::vfi_root_custom_t<tile_t>;

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
                uint32_t found = root->prev_highlight_index;
                if (found >= root->current_tile->children.size()) {
                  return 0;
                }
                root->current_tile->children[found].set_color(fan::color(1, 1, 1));
              }
              if (children.size()) {
                uint32_t found = root->find_top_layer_shape(children);
                if (found >= children.size()) {
                  return 0;
                }
                children[found].set_color(highlighted_tile_color);
                root->current_tile = this;
                root->prev_highlight_index = found;
              }
            }
          }
          //root->event_type = event_type_e::move;
          //root->current_shape = this;
          return 0;
        };
        vfi_root_t::set_root(vfip);
        vfi_root_t::push_child(std::move(temp));
       /* vfi_root_t::push_child(fan::graphics::rectangle_t{{
            .position = fan::vec3(fan::vec2(temp.get_position()), vfip.shape.rectangle->position.z),
            .size = root->tile_size,
            .color = fan::color(0, 1, 0, 0),
            .blending = true
        }});*/
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

  uint32_t find_top_layer_shape(auto& children) {
    uint32_t found = -1;
    int64_t depth = -1;
    for (int i = 0; i < children.size(); ++i) {
      if (children[i].get_position().z > depth) {
        depth = children[i].get_position().z;
        found = i;
      }
    }
    return found;
  };

  uint32_t find_layer_shape(){
    if (current_tile == nullptr) {
      return -1;
    }
    uint32_t found = -1;
    for (int i = 0; i < current_tile->children.size(); ++i) {
      if (current_tile->children[i].get_position().z == brush.depth) {
        found = i;
        break;
      }
    }
    return found;
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
              j->children[0].set_image(&texture_gray_shades[idx]);
            }
            ++x;
            continue;
          }
          j = std::make_unique<shapes_t::global_t>(this, fan::graphics::sprite_t{{
            .position = fan::vec3(tile_size + fan::vec2(x, y) * tile_size * 2, 0),
            .size = tile_size,
            .image = &texture_gray_shades[idx]
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

  void draw_collisions() {
    //for (auto& i : map_tiles) {
    //  for (auto& j : i) {
    //    if (j->shape_data.tile.mesh_property != 0) {
    //      // dont hardcode color
    //      j->children[1].set_color(fan::color(0, 1, 0, 0.1));
    //    }
    //  }
    //}
  }

  void undraw_collisions() {
    //for (auto& i : map_tiles) {
    //  for (auto& j : i) {
    //    if (j->shape_data.tile.mesh_property != 0) {
    //      // dont hardcode color
    //      j->children[1].set_color(fan::color(0, 1, 0, 0));
    //    }
    //  }
    //}
  }

  void reset_map() {
    map_tiles.clear();
    resize_map();
  }

  void open(const fan::string& texturepack_name) {
    texturepack.open_compiled(texturepack_name);

    gloco->get_window()->add_mouse_move_callback([this](const auto& d) {
      if (viewport_settings.move) {
        fan::vec2 move_off = (d.position - viewport_settings.offset) / viewport_settings.zoom * 2;
        gloco->default_camera->camera.set_camera_position(viewport_settings.pos - move_off);
      }
    });

    gloco->get_window()->add_buttons_callback([this](const auto& d) {
      if (!editor_settings.hovered && d.state != fan::mouse_state::release) {
        return;
      }
      

      {// handle camera movement
        f32_t old_zoom = viewport_settings.zoom;

        switch (d.button) {
          case fan::mouse_middle: { break;}
          case fan::mouse_scroll_up: {
            if (gloco->get_window()->key_pressed(fan::key_left_control)) {
              brush.depth += 1;
            }
            else {
              viewport_settings.zoom *= scroll_speed;
            }
            return; 
          }
          case fan::mouse_scroll_down: { 
            if (gloco->get_window()->key_pressed(fan::key_left_control)) {
              brush.depth -= 1;
            }
            else {
              viewport_settings.zoom /= scroll_speed; 
            }
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
      if (ImGui::IsAnyItemActive()) {
        return;
      }

      switch (d.key) {
        case fan::key_delete: {
          if (gloco->get_window()->key_pressed(fan::key_left_control)) {
            reset_map();
          }
          break;
        }
          // change this
        case fan::key_e: {
          render_collisions = !render_collisions;
          if (render_collisions) {
            draw_collisions();
          }
          else {
            undraw_collisions();
          }
          break;
        }
      }
    });

    // transparent pattern
    texture_gray_shades[0].create(fan::color::rgb(60, 60, 60, 255), fan::vec2(1, 1));
    texture_gray_shades[1].create(fan::color::rgb(40, 40, 40, 255), fan::vec2(1, 1));

    viewport_settings.size = 0;

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

    viewport_settings.pos = gloco->get_window()->get_size() / 2;
    gloco->default_camera->camera.set_camera_position(viewport_settings.pos);

    gloco->process_frame();
    resize_map();
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

      if (ImGui::Begin(editor_str)) {
        fan::vec2 window_size = gloco->get_window()->get_size();
        fan::vec2 viewport_size = ImGui::GetWindowSize();
        fan::vec2 viewport_pos = fan::vec2(ImGui::GetWindowPos() + fan::vec2(0, ImGui::GetFontSize() + ImGui::GetStyle().FramePadding.y * 2));
        fan::vec2 offset = viewport_size - viewport_size / viewport_settings.zoom;
        fan::vec2 s = viewport_size;
        gloco->default_camera->camera.set_ortho(
          fan::vec2(-s.x, s.x) / viewport_settings.zoom,
          fan::vec2(-s.y, s.y) / viewport_settings.zoom
        );

        //gloco->default_camera->camera.set_camera_zoom(viewport_settings.zoom);
        gloco->default_camera->viewport.set(viewport_pos, viewport_size, window_size);
        editor_size = ImGui::GetContentRegionAvail();
        viewport_settings.size = editor_size;
        ImGui::SetWindowFontScale(1.5);
        if (render_collisions) {
          ImGui::TextColored(ImVec4(1, 0, 0, 1), "rendering collisions");
        }
        ImGui::TextColored(ImVec4(1, 1, 1, 1), fan::format("brush depth:{}", (int)brush.depth).c_str());


        if (ImGui::Begin("Layer window")) {
          struct data_t {
            fan::string text;
          };
          static std::vector<data_t> layers{{.text = "default"}};
          for (int i = 0; i < layers.size(); ++i) {
            layers[i].text.resize(32);
            ImGui::Text(fan::format("Layer {}", i).c_str());
            ImGui::SameLine();
            ImGui::InputText(fan::format("##layer{}", i).c_str(), layers[i].text.data(), layers[i].text.size());
          }
          if (ImGui::Button("+")) {
            layers.push_back(data_t{.text = "default"});
          }
          ImGui::SameLine();
          if (ImGui::Button("-")) {
            layers.pop_back();
          }
        }

        ImGui::End();
      }

      editor_settings.hovered = ImGui::IsWindowHovered();

      // add texture
      if (editor_settings.hovered) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
          if (render_collisions && current_tile != nullptr && current_tile->children.size()) {
            //current_tile->children[1].set_color(fan::color(0, 1, 0, 0.1));
            ///current_tile->shape_data.tile.mesh_property = 1;
          }
          else {
            if (current_tile != nullptr && current_tile_image.ti.image != nullptr && current_tile->children.size()) {
              fan::graphics::vfi_root_custom_t<tile_t>::child_data_t* current_shape = 0;
              uint32_t found = find_layer_shape();
              if (found == (uint32_t)-1) {
                // copy initial
                current_tile->children.push_back(current_tile->children[0]);
                fan::vec3 parent_pos = current_tile->children[0].get_position();
                current_tile->children.back().set_position(fan::vec3(parent_pos.x, parent_pos.y, brush.depth));
                //current_tile->children.set_image(current_tile->)
                  // implement meshproprty 
                current_tile->children.back().mesh_property = 1;
                current_shape = &current_tile->children.back();
                current_shape->set_color(1);
              }
              else {
                current_shape = &current_tile->children[found];
              }
              gloco->shapes.sprite.set(
                *current_shape,
                &loco_t::shapes_t::sprite_t::vi_t::tc_position,
                current_tile_image.ti.position
              );
              gloco->shapes.sprite.set(
                *current_shape,
                &loco_t::shapes_t::sprite_t::vi_t::tc_size,
                current_tile_image.ti.size
              );
              //current_tile->children[0].set_color(1);
              current_shape->set_image(current_tile_image.ti.image);
              current_shape->image_hash = current_tile_image.image_hash;
            }
          }
        }
        //remove texture
        if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
          if (render_collisions && current_tile != nullptr && current_tile->children.size()) {
           /* current_tile->children[1].set_color(fan::color(0, 1, 0, 0));
            current_tile->mesh_property = 0;*/
          }
          else {
            if (current_tile != nullptr && current_tile->children.size()) {
              uint32_t found = find_layer_shape();
              if (found == (uint32_t)-1) {
                goto g_skip;
              }
              current_tile->children.erase(current_tile->children.begin() + found);
              prev_highlight_index = -1;
              current_tile = nullptr;
            }
          }
        }
      g_skip:;
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
  element count
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
        fan::mp_t<current_version_t::shapes_t> shapes;

        shapes.iterate([&]<auto i0, typename T>(T & l) {
          fan::mp_t<T> shape;
          shape.init(this, j.get());

          fan::string shape_str;
          shape.iterate([&]<auto i1, typename T2>(T2 & v) {
            if constexpr (std::is_same_v<T2, fan::string>) {
              uint64_t string_length = v.size();
              shape_str.append((char*)&string_length, sizeof(string_length));
              shape_str.append(v);
            }
            else if constexpr (fan_requires_rule(T2, typename T2::value_type)) {
              if constexpr (std::is_same_v<T2, std::vector<typename T2::value_type>>) {
                uint32_t len = v.size();
                shape_str.append((char*)&len, sizeof(uint32_t));
                for (auto& ob : v) {
                  shape_str.append((char*)&ob, sizeof(ob));
                }
              }
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
    #include _FAN_PATH(graphics/gui/tilemap_editor/loader_versions/1.h)
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

  bool render_collisions = false;

  struct {
    f32_t depth = 1;
  }brush;

  struct {
    f32_t zoom = 0.5;
    bool move = false;
    fan::vec2 pos = 0;
    fan::vec2 size = 0;
    fan::vec2 offset = 0;
  }viewport_settings;

  struct {
    bool hovered = false;
  }editor_settings;

  uint32_t prev_highlight_index = -1;

  // very bad fix to prevent mouse move cb when erasing vfi
  bool erasing = false;
};
