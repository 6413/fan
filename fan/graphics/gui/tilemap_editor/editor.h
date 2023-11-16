#pragma once

struct fte_t {

  static constexpr auto editor_str = "Editor";
  static constexpr auto editor_settings_str = "Editor settings";
  static constexpr auto properties_str = "Properties";

  static constexpr int max_path_input = 40;

  static constexpr fan::vec2 default_button_size{100, 30};
  static constexpr fan::vec2 tile_viewer_sprite_size{64, 64};
  static constexpr fan::color highlighted_tile_color = fan::color(0.5, 0.5, 1);
  static constexpr fan::color highlighted_selected_tile_color = fan::color(0.5, 0, 0, 0.5);

  static constexpr f32_t scroll_speed = 1.2;
  static constexpr uint32_t invalid = -1;

  struct shape_depths_t {
    static constexpr uint32_t max_layer_depth = 0xffff;
    static constexpr int cursor_highlight_depth = 10000;
  };

  fan::string file_name = "file.fte";

  #include "common2.h"

  struct shapes_t {
    struct global_t {

      global_t() = default;

      template <typename T>
      global_t(fte_t* root, const T& obj) {
        layers.push_back(obj);
      }

      struct layer_t {
        tile_t tile;
        loco_t::shape_t shape;
      };
      std::vector<layer_t> layers;
    };
  };

  #include "common.h"

  enum class event_type_e {
    none,
    add,
    remove
  };

  uint32_t find_top_layer_shape(const auto& vec) {
    uint32_t found = -1;
    int64_t depth = -1;
    for (int i = 0; i < vec.size(); ++i) {
      if (vec[i].tile.position.z > depth) {
        depth = vec[i].tile.position.z;
        found = i;
      }
    }
    return found;
  };

  uint32_t find_layer_shape(const auto& vec){
    uint32_t found = -1;
    for (int i = 0; i < vec.size(); ++i) {
      if (vec[i].tile.position.z == brush.depth) {
        found = i;
        break;
      }
    }
    return found;
  };

  void resize_map() {
    grid_visualize.background.set_size(tile_size * map_size);
    grid_visualize.background.set_tc_size(fan::vec2(0.5) * map_size);
    gloco->shapes.line_grid.sb_set_vi(
      grid_visualize.line_grid,
      &loco_t::shapes_t::line_grid_t::vi_t::grid_size,
      map_size
    );
    grid_visualize.line_grid.set_size(map_size * (tile_size / 2) * 2);

    fan::vec2 s = grid_visualize.highlight_hover.get_size();
    fan::vec2 sp = fan::vec2(grid_visualize.highlight_hover.get_position());
    fan::vec2 p = tile_size * ((sp / s));
    grid_visualize.highlight_hover.set_position(p);
    grid_visualize.highlight_hover.set_size(tile_size);
    grid_visualize.highlight_selected.set_position(p);
    if (current_tile.shape != nullptr) {
      grid_visualize.highlight_selected.set_size(tile_size);
    }
  }

  void reset_map() {
    map_tiles.clear();
    resize_map();
  }

  bool window_relative_to_grid(const fan::vec2& window_relative_position, fan::vec2i* in) {
    fan::vec2 p = gloco->translate_position(window_relative_position) / 2 + gloco->default_camera->camera.get_position() / 2;
    fan::vec2 ws = gloco->window.get_size();
    p = (p / tile_size).floor() * tile_size * 2 + tile_size;
    *in = p;
    return fan_2d::collision::rectangle::point_inside_no_rotation(p / tile_size, 0, map_size);
  }

  void open(const fan::string& texturepack_name) {
    texturepack.open_compiled(texturepack_name);

    gloco->window.add_mouse_move_callback([this](const auto& d) {
      if (viewport_settings.move) {
        fan::vec2 move_off = (d.position - viewport_settings.offset) / viewport_settings.zoom * 2;
        gloco->default_camera->camera.set_position(viewport_settings.pos - move_off);
      }
      fan::vec2i p;
      {
        if (window_relative_to_grid(d.position, &p)) {
          fan::vec2i grid_position = p;
          grid_position /= fan::vec2i(tile_size);
          grid_visualize.highlight_hover.set_position(fan::vec2(p));
          grid_visualize.highlight_hover.set_color(fan::color(1, 1, 1, 0.6));
          //uint32_t idx = find_layer_shape(map_tiles[fan::vec2i(grid_position.x, grid_position.y)].layers);
          //if (idx != invalid) {
            //grid_visualize.highlight_hover.set_depth(map_tiles[fan::vec2i(grid_position.x, grid_position.y)].layers
            //  [idx].shape.get_position().z);
          //}
        }
        else {
          grid_visualize.highlight_hover.set_color(fan::colors::transparent);
        }
      }
    });

    gloco->window.add_buttons_callback([this](const auto& d) {
      if (!editor_settings.hovered && d.state != fan::mouse_state::release) {
        return;
      }

      {// handle camera movement
        f32_t old_zoom = viewport_settings.zoom;

        switch (d.button) {
          case fan::mouse_middle: { 
            viewport_settings.move = (bool)d.state;
            fan::vec2 old_pos = viewport_settings.pos;
            viewport_settings.offset = gloco->get_mouse_position();
            viewport_settings.pos = gloco->default_camera->camera.get_position();
            break;
          }
          case fan::mouse_scroll_up: {
            if (gloco->window.key_pressed(fan::key_left_control)) {
              brush.depth += 1;
              brush.depth = std::min((uint32_t)brush.depth, shape_depths_t::max_layer_depth);
            }
            else {
              viewport_settings.zoom *= scroll_speed;
            }
            return; 
          }
          case fan::mouse_scroll_down: { 
            if (gloco->window.key_pressed(fan::key_left_control)) {
              brush.depth -= 1;
              brush.depth = std::max((uint32_t)brush.depth, (uint32_t)1);
            }
            else {
              viewport_settings.zoom /= scroll_speed; 
            }
            return; 
          }
          default: {return;} //?
        };
      }// handle camera movement
   });

    gloco->window.add_keys_callback([this](const auto& d) {
      if (d.state != fan::keyboard_state::press) {
        return;
      }
      if (ImGui::IsAnyItemActive()) {
        return;
      }

      switch (d.key) {
        case fan::key_delete: {
          if (gloco->window.key_pressed(fan::key_left_control)) {
            reset_map();
          }
          break;
        }
          // change this
        case fan::key_e: {
          render_collisions = !render_collisions;
         /* if (render_collisions) {
            draw_collisions();
          }
          else {
            undraw_collisions();
          }*/
          break;
        }
      }
    });

    viewport_settings.size = 0;

    texturepack_images.reserve(texturepack.texture_list.size());

    // loaded texturepack
    texturepack.iterate_loaded_images([this](auto& image, uint32_t pack_id) {
      tile_info_t ii;
      ii.ti = loco_t::texturepack_t::ti_t{
        .pack_id = pack_id,
        .position = image.position,
        .size = image.size,
        .image = &texturepack.get_pixel_data(pack_id).image
      };

      ii.image_hash = image.hash;

      texturepack_images.push_back(ii);
    });

    grid_visualize.background = fan::graphics::sprite_t{{
      .position = fan::vec3(viewport_settings.pos, 0),
      .size = 0,
      .image = &gloco->transparent_texture
    }};

    grid_visualize.highlight_color.create(fan::colors::red, 1);
    grid_visualize.highlight_hover = fan::graphics::sprite_t{{
      .position = fan::vec3(viewport_settings.pos, shape_depths_t::cursor_highlight_depth),
      .size = tile_size,
      .image = &grid_visualize.highlight_color,
      .blending = true
    }};
    grid_visualize.highlight_selected = fan::graphics::rectangle_t{{
      .position = fan::vec3(viewport_settings.pos, shape_depths_t::cursor_highlight_depth - 1),
      .size = 0,
      .color = highlighted_selected_tile_color,
      .blending = true
    }};

    {
      fan::vec2 p = 0;
      p = ((p - tile_size) / tile_size).floor() * tile_size;
      grid_visualize.highlight_hover.set_position(p);
    }

    // update viewport sizes
    gloco->process_frame();

    gloco->default_camera->camera.set_position(viewport_settings.pos);

    loco_t::shapes_t::line_grid_t::properties_t p;
    p.position = fan::vec3(0, 0, shape_depths_t::cursor_highlight_depth + 1);
    p.size = 0;
    p.color = fan::color::rgb(0, 128, 255);

    grid_visualize.line_grid = p;

    resize_map();
  }
  void close() {
    texturepack.close();
  }

  bool is_in_constrains(fan::vec2i& position, int& j, int& i)
  {
    position += (-brush.size / 2) * tile_size * 2 + tile_size * 2 * fan::vec2(j, i) +
      fan::vec2(1 ? 0 : 0, 1 ? 0 : 0);
    if (position.x > map_size.x * tile_size.x || position.x < -(map_size.x * tile_size.x)) {
      return false;
    }
    if (position.y > map_size.y * tile_size.y || position.y < -(map_size.y * tile_size.y)) {
      return false;
    }
    return true;
  }

  bool handle_tile_push(fan::vec2i& position, int& j, int& i)
  {
    if (!is_in_constrains(position, j, i)) {
      return true;
    }
    // fan::vec2(0, (-brush_sizey / 2)* tile_size.y * 2 + i * tile_size.y * 2);
    fan::vec2i grid_position = position;
    grid_position /= fan::vec2i(tile_size);
    auto& layers = map_tiles[fan::vec2i(grid_position.x, grid_position.y)].layers;
    uint32_t idx = find_top_layer_shape(layers);
    if ((idx == invalid || idx + 1 < brush.depth) && current_tile_image.ti.valid()) {
      layers.resize(layers.size() + 1);
      layers.back().tile.position = fan::vec3(position, brush.depth);
      layers.back().tile.image_hash = current_tile_image.image_hash;
      // todo fix
      layers.back().tile.mesh_property = mesh_property_t::none;
      layers.back().shape = fan::graphics::sprite_t{{
          .position = fan::vec3(position, brush.depth),
          .size = tile_size
        }};
      if (layers.back().shape.set_tp(&current_tile_image.ti)) {
        fan::print("failed to load image");
      }
      current_tile.shape = &layers.back().shape;
      current_tile.tile_info.ti = current_tile_image.ti;
    }
    else if (idx != invalid && current_tile_image.ti.valid()) {
      fan::vec2i grid_position = position;
      grid_position /= fan::vec2i(tile_size);
      auto found = map_tiles.find(fan::vec2i(grid_position.x, grid_position.y));
      if (found != map_tiles.end()) {
        auto& layers = map_tiles[fan::vec2i(grid_position.x, grid_position.y)].layers;
        idx = find_layer_shape(layers);
        if (idx != invalid || idx < layers.size()) {
          layers[idx].shape.set_tp(&current_tile_image.ti);
        }
      }
    }
    return false;
  }

  bool handle_tile_erase(fan::vec2i& position, int& j, int& i) {
    if (!is_in_constrains(position, j, i)) {
      return true;
    }
    fan::vec2i grid_position = position;
    grid_position /= fan::vec2i(tile_size);
    auto found = map_tiles.find(fan::vec2i(grid_position.x, grid_position.y));
    if (found != map_tiles.end()) {
      auto& layers = map_tiles[fan::vec2i(grid_position.x, grid_position.y)].layers;
      uint32_t idx = find_layer_shape(layers);
      if (idx != invalid || idx < brush.depth) {
        layers.erase(layers.begin() + idx);
      }
    }
  }

  void handle_tile_brush() {
    if (editor_settings.hovered &&
  ImGui::IsMouseDown(fan::window_input::fan_to_imguikey(fan::mouse_left)) &&
  !gloco->window.key_pressed(fan::key_left_control) &&
  !gloco->window.key_pressed(fan::key_left_shift)) {
      fan::vec2i position;
      // if inside grids
      for (int i = 0; i < brush.size.y; ++i) {
        for (int j = 0; j < brush.size.x; ++j) {
          if (window_relative_to_grid(gloco->get_mouse_position(), &position)) {
            if (handle_tile_push(position, j, i)) {
              continue;
            }
          }
          else {
            if (editor_settings.hovered) {
              if (current_tile.shape != nullptr) {
                grid_visualize.highlight_selected.set_size(0);
                current_tile.shape = nullptr;
              }
            }
          }
        }
      }
    }
    if (ImGui::IsMouseDown(fan::window_input::fan_to_imguikey(fan::mouse_right))) {
      fan::vec2i position;
      for (int i = 0; i < brush.size.y; ++i) {
        for (int j = 0; j < brush.size.x; ++j) {
          if (window_relative_to_grid(gloco->get_mouse_position(), &position)) {
            if (handle_tile_erase(position, j, i)) {
              continue;
            }
          }
        }
      }
    }
  }

  void handle_editor_window(fan::vec2& editor_size) {
    if (ImGui::Begin(editor_str)) {
      fan::vec2 window_size = gloco->window.get_size();
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
    ImGui::End();
  }

  bool handle_editor_settings_window() {
    if (ImGui::Begin(editor_settings_str)) {
      {
        if (ImGui::SliderInt2("brush size", brush.size.data(), 1, 100)) {
          grid_visualize.highlight_hover.set_size(tile_size * brush.size);
        }

        if (ImGui::InputInt2("map size", map_size.data())) {
          if (ImGui::IsItemDeactivatedAfterEdit()) {
            resize_map();
          }
        }
        if (ImGui::InputInt2("tile size", tile_size.data())) {
          resize_map();
          for (auto& i : map_tiles) {
            for (auto& j : i.second.layers) {
              fan::vec2 s = j.shape.get_size();
              fan::vec2 sp = fan::vec2(j.shape.get_position());
              fan::vec2 p = tile_size * ((sp / s));
              j.shape.set_position(p);
              j.shape.set_size(tile_size);
            }
          }
        }

        if (ImGui::Checkbox("render grid", &grid_visualize.render_grid)) {
          if (grid_visualize.render_grid) {
            grid_visualize.line_grid.set_size(map_size * (tile_size / 2) * 2);
          }
          else {
            grid_visualize.line_grid.set_size(0);
          }
        }

        // use ImGui::Dummy here
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
          return true;
        }
      }
    }
    ImGui::End();
    return false;
  }

  void handle_properties_window() {
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
            (void*)(intptr_t)node.ti.image->get_texture(),
            tile_viewer_sprite_size,
            node.ti.position / node.ti.image->size,
            node.ti.position / node.ti.image->size + node.ti.size / node.ti.image->size
          )) {
            current_image_idx = i;
            current_tile_image = node;
            grid_visualize.highlight_hover.set_tp(&current_tile_image.ti);
          }

          if (selected) {
            ImGui::PopStyleColor(3);
          }

          if ((i + 1) % images_per_row != 0)
            ImGui::SameLine();
        }
      }
    }
  }

  void handle_tile_settings_window()
  {
    if (ImGui::Begin("Tile settings")) {
      if (current_tile.shape != nullptr) {
        {
          f32_t angle = current_tile.shape->get_angle();
          if (ImGui::SliderAngle("angle", &angle)) {
            current_tile.shape->set_angle(angle);
          }
        }
        {
          int mesh_property = (int)current_tile.tile_info.mesh_property;
          if (ImGui::SliderInt("mesh flags", &mesh_property, 0, (int)mesh_property_t::size - 1)) {
            current_tile.tile_info.mesh_property = (mesh_property_t)mesh_property;
          }
        }
      }
    }
    ImGui::End();
  }

  void handle_pick_tile() {
    fan::vec2i position;
    if (window_relative_to_grid(gloco->get_mouse_position(), &position)) {
      fan::vec2i grid_position = position;
      grid_position /= fan::vec2i(tile_size);
      auto found = map_tiles.find(fan::vec2i(grid_position.x, grid_position.y));
      if (found != map_tiles.end()) {
        auto& layers = found->second.layers;
        uint32_t idx = find_layer_shape(layers);
        if (idx == invalid) {
          idx = find_top_layer_shape(layers);
        }
        if (idx != invalid || idx < brush.depth) {
          current_tile_image.ti = layers[idx].shape.get_tp();
          current_tile_image.image_hash = layers[idx].tile.image_hash;
          grid_visualize.highlight_hover = layers[idx].shape;
          grid_visualize.highlight_hover.set_depth(brush.depth);
        }
      }
    }
  }

  void handle_select_tile() {
    fan::vec2i position;
    if (window_relative_to_grid(gloco->get_mouse_position(), &position)) {
      fan::vec2i grid_position = position;
      grid_position /= fan::vec2i(tile_size);
      auto found = map_tiles.find(fan::vec2i(grid_position.x, grid_position.y));
      if (found != map_tiles.end()) {
        auto& layers = found->second.layers;
        uint32_t idx = find_top_layer_shape(layers);
        if ((idx != invalid || idx < brush.depth)) {
          current_tile.shape = &layers[idx].shape;
          current_tile.tile_info.mesh_property = layers[idx].tile.mesh_property;
          current_tile.tile_info.image_hash = layers[idx].tile.image_hash;
          grid_visualize.highlight_selected.set_position(fan::vec2(position));
          grid_visualize.highlight_selected.set_size(tile_size);
        }
      }
    }
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

    fan::vec2 editor_size;

    handle_editor_window(editor_size);

    if (handle_editor_settings_window()) {
      return;
    }

    handle_properties_window();

    handle_tile_settings_window();

    if (editor_settings.hovered && ImGui::IsMouseDown(fan::window_input::fan_to_imguikey(fan::mouse_left))) {
      if (gloco->window.key_pressed(fan::key_left_control)) {
        handle_pick_tile();
      }
      else if (gloco->window.key_pressed(fan::key_left_shift)) {
        handle_select_tile();
      }
    }

    handle_tile_brush();

    ImGui::End();
  }

  fan::graphics::imgui_element_t main_view =
    fan::graphics::imgui_element_t([&] {handle_imgui(); });

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
      fan::mp_t<current_version_t::shapes_t> shapes;

      shapes.iterate([&]<auto i0, typename T>(T & l) {
        fan::mp_t<T> shape;
        shape.init(this, &i.second);

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

  fan::vec2i map_size{64, 64};
  fan::vec2i tile_size{32, 32};

  struct tile_info_t {
    loco_t::texturepack_t::ti_t ti;
    uint64_t image_hash;
    mesh_property_t mesh_property = mesh_property_t::none;
  };

  struct current_tile_t {
    loco_t::shape_t* shape = nullptr;
    tile_info_t tile_info;
  };

  current_tile_t current_tile;
  current_tile_t current_hover_tile;
  tile_info_t current_tile_image;

  struct vec2i_hasher
  {
    std::size_t operator()(const fan::vec2i& k) const
    {
      std::hash<int> hasher;
      std::size_t hash_value = 17;
      hash_value = hash_value * 31 + hasher(k.x);
      hash_value = hash_value * 31 + hasher(k.y);
      return hash_value;
    }
  };

  std::unordered_map<fan::vec2i, shapes_t::global_t, vec2i_hasher> map_tiles;

  loco_t::texturepack_t texturepack;

  std::vector<tile_info_t> texturepack_images;

  struct {
    loco_t::shape_t background;
    loco_t::shape_t highlight_selected;
    loco_t::shape_t highlight_hover;
    loco_t::image_t highlight_color;
    loco_t::shape_t line_grid;
    bool render_grid = true;
  }grid_visualize;

  bool render_collisions = false;

  struct {
    fan::vec2i size = 1;
    f32_t depth = 1;
  }brush;

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
};
