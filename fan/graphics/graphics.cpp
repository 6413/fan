#include <fan/graphics/graphics.h>

#if defined(fan_gui)
  #include <fan/imgui/imgui_internal.h>
  #include <fan/graphics/gui/imgui_themes.h>
#endif

loco_t::polygon_t::properties_t fan::graphics::create_hexagon(f32_t radius, const fan::color& color) {
  loco_t::polygon_t::properties_t pp;
  // for triangle strip
  for (int i = 0; i < 6; ++i) {
    pp.vertices.push_back(fan::graphics::vertex_t{ fan::vec3(0, 0, 0), color });

    f32_t angle1 = 2 * fan::math::pi * i / 6;
    f32_t x1 = radius * std::cos(angle1);
    f32_t y1 = radius * std::sin(angle1);
    pp.vertices.push_back(fan::graphics::vertex_t{ fan::vec3(fan::vec2(x1, y1), 0), color });

    f32_t angle2 = 2 * fan::math::pi * ((i + 1) % 6) / 6;
    f32_t x2 = radius * std::cos(angle2);
    f32_t y2 = radius * std::sin(angle2);
    pp.vertices.push_back(fan::graphics::vertex_t{ fan::vec3(fan::vec2(x2, y2), 0), color });
  }

  return pp;
}

fan::line3 fan::graphics::get_highlight_positions(const fan::vec3& op_, const fan::vec2& os, int index) {
  fan::line3 positions;
  fan::vec2 op = op_;
  switch (index) {
  case 0:
    positions[0] = fan::vec3(op - os, op_.z + 1);
    positions[1] = fan::vec3(op + fan::vec2(os.x, -os.y), op_.z + 1);
    break;
  case 1:
    positions[0] = fan::vec3(op + fan::vec2(os.x, -os.y), op_.z + 1);
    positions[1] = fan::vec3(op + os, op_.z + 1);
    break;
  case 2:
    positions[0] = fan::vec3(op + os, op_.z + 1);
    positions[1] = fan::vec3(op + fan::vec2(-os.x, os.y), op_.z + 1);
    break;
  case 3:
    positions[0] = fan::vec3(op + fan::vec2(-os.x, os.y), op_.z + 1);
    positions[1] = fan::vec3(op - os, op_.z + 1);
    break;
  default:
    // Handle invalid index if necessary
    break;
  }

  return positions;
}

fan::vec2 fan::graphics::get_mouse_position() {
  return gloco->get_mouse_position();
}

fan::vec2 fan::graphics::get_mouse_position(const loco_t::camera_t& camera, const loco_t::viewport_t& viewport) {
  return loco_t::transform_position(gloco->get_mouse_position(), viewport, camera);
}

#if defined(fan_gui)

void fan::graphics::gui::image(loco_t::image_t img, const fan::vec2& size, const fan::vec2& uv0, const fan::vec2& uv1, const fan::color& tint_col, const fan::color& border_col) {
  ImGui::Image((ImTextureID)gloco->image_get_handle(img), size, uv0, uv1, tint_col, border_col);
}

bool fan::graphics::gui::image_button(const std::string& str_id, loco_t::image_t img, const fan::vec2& size, const fan::vec2& uv0, const fan::vec2& uv1, int frame_padding, const fan::color& bg_col, const fan::color& tint_col) {
  return ImGui::ImageButton(str_id.c_str(), (ImTextureID)gloco->image_get_handle(img), size, uv0, uv1, bg_col, tint_col);
}

bool fan::graphics::gui::image_text_button(loco_t::image_t img, const std::string& text, const fan::color& color, const fan::vec2& size, const fan::vec2& uv0, const fan::vec2& uv1, int frame_padding, const fan::color& bg_col, const fan::color& tint_col) {
  bool ret = ImGui::ImageButton(text.c_str(), (ImTextureID)gloco->image_get_handle(img), size, uv0, uv1, bg_col, tint_col);
  ImVec2 text_size = ImGui::CalcTextSize(text.c_str());
  ImVec2 pos = ImGui::GetItemRectMin(); 
  pos.x += (size.x - text_size.x) * 0.5f;
  pos.y += (size.y - text_size.y) * 0.5f;
  ImGui::GetWindowDrawList()->AddText(pos, ImGui::GetColorU32(color), text.c_str());
  return ret;
}

bool fan::graphics::gui::toggle_button(const std::string& str, bool* v) {
  ImGui::Text("%s", str.c_str());
  ImGui::SameLine();

  ImVec2 p = ImGui::GetCursorScreenPos();
  ImDrawList* draw_list = ImGui::GetWindowDrawList();

  float height = ImGui::GetFrameHeight();
  float width = height * 1.55f;
  float radius = height * 0.50f;

  bool changed = ImGui::InvisibleButton(("##" + str).c_str(), ImVec2(width, height));
  if (changed)
    *v = !*v;
  ImU32 col_bg;
  if (ImGui::IsItemHovered())
    col_bg = *v ? IM_COL32(145 + 20, 211, 68 + 20, 255) : IM_COL32(218 - 20, 218 - 20, 218 - 20, 255);
  else
    col_bg = *v ? IM_COL32(145, 211, 68, 255) : IM_COL32(218, 218, 218, 255);

  draw_list->AddRectFilled(p, ImVec2(p.x + width, p.y + height), col_bg, height * 0.5f);
  draw_list->AddCircleFilled(ImVec2(*v ? (p.x + width - radius) : (p.x + radius), p.y + radius), radius - 1.5f, IM_COL32(255, 255, 255, 255));

  return changed;
}

bool fan::graphics::gui::toggle_image_button(const std::string& char_id, loco_t::image_t image, const fan::vec2& size, bool* toggle) {
  bool clicked = false;

  ImVec4 tintColor = ImVec4(1, 1, 1, 1);
  if (*toggle) {
    tintColor = ImVec4(0.3f, 0.3f, 0.3f, 1.0f);
  }

  if (fan::graphics::gui::image_button(char_id, image, size, ImVec2(0, 0), ImVec2(1, 1), -1, ImVec4(0, 0, 0, 0), tintColor)) {
    *toggle = !(*toggle);
    clicked = true;
  }

  return clicked;
}

void fan::graphics::gui::text_bottom_right(const char* text, uint32_t reverse_yoffset) {
  ImDrawList* draw_list = ImGui::GetWindowDrawList();

  ImVec2 window_pos = ImGui::GetWindowPos();
  ImVec2 window_size = ImGui::GetWindowSize();

  ImVec2 text_size = ImGui::CalcTextSize(text);

  ImVec2 text_pos;
  text_pos.x = window_pos.x + window_size.x - text_size.x - ImGui::GetStyle().WindowPadding.x;
  text_pos.y = window_pos.y + window_size.y - text_size.y - ImGui::GetStyle().WindowPadding.y;

  text_pos.y -= reverse_yoffset * ImGui::GetTextLineHeightWithSpacing();

  draw_list->AddText(text_pos, IM_COL32(255, 255, 255, 255), text);
}

fan::vec2 fan::graphics::gui::get_position_bottom_corner(const std::string& text, uint32_t reverse_yoffset) {
  fan::vec2 window_pos = ImGui::GetWindowPos();
  fan::vec2 window_size = ImGui::GetWindowSize();

  fan::vec2 text_size = ImGui::CalcTextSize(text.c_str());

  fan::vec2 text_pos;
  text_pos.x = window_pos.x + window_size.x - text_size.x - ImGui::GetStyle().WindowPadding.x;
  text_pos.y = window_pos.y + window_size.y - text_size.y - ImGui::GetStyle().WindowPadding.y;

  text_pos.y -= reverse_yoffset * ImGui::GetTextLineHeightWithSpacing();

  return text_pos;
}

void fan::graphics::gui::image_rotated(loco_t::image_t image, const fan::vec2& size, int angle, const fan::vec2& uv0, const fan::vec2& uv1, const fan::color& tint_col, const fan::color& border_col) {
  IM_ASSERT(angle % 90 == 0);
  ImVec2 _uv0, _uv1, _uv2, _uv3;
  switch (angle % 360)
  {
  case 0:
    fan::graphics::gui::image(image, size, uv0, uv1, tint_col, border_col);
    return;
  case 180:
    fan::graphics::gui::image(image, size, uv1, uv0, tint_col, border_col);
    return;
  case 90:
    _uv3 = uv0;
    _uv1 = uv1;
    _uv0 = ImVec2(uv1.x, uv0.y);
    _uv2 = ImVec2(uv0.x, uv1.y);
    break;
  case 270:
    _uv1 = uv0;
    _uv3 = uv1;
    _uv0 = ImVec2(uv0.x, uv1.y);
    _uv2 = ImVec2(uv1.x, uv0.y);
    break;
  }
  ImGuiWindow* window = ImGui::GetCurrentWindow();
  if (window->SkipItems)
    return;
  ImVec2 _size(size.y, size.x);
  ImRect bb(window->DC.CursorPos, window->DC.CursorPos + _size);
  if (border_col.a > 0.0f)
    bb.Max += ImVec2(2, 2);
  ImGui::ItemSize(bb);
  if (!ImGui::ItemAdd(bb, 0))
    return;
  if (border_col.a > 0.0f)
  {
    window->DrawList->AddRect(bb.Min, bb.Max, ImGui::GetColorU32(border_col), 0.0f);
    ImVec2 x0 = bb.Min + ImVec2(1, 1);
    ImVec2 x2 = bb.Max - ImVec2(1, 1);
    ImVec2 x1 = ImVec2(x2.x, x0.y);
    ImVec2 x3 = ImVec2(x0.x, x2.y);
    window->DrawList->AddImageQuad((ImTextureID)gloco->image_get_handle(image), x0, x1, x2, x3, _uv0, _uv1, _uv2, _uv3, ImGui::GetColorU32(tint_col));
  }
  else
  {
    ImVec2 x1 = ImVec2(bb.Max.x, bb.Min.y);
    ImVec2 x3 = ImVec2(bb.Min.x, bb.Max.y);
    window->DrawList->AddImageQuad((ImTextureID)gloco->image_get_handle(image), bb.Min, x1, bb.Max, x3, _uv0, _uv1, _uv2, _uv3, ImGui::GetColorU32(tint_col));
  }
}

void fan::graphics::gui::content_browser_t::render() {
  item_right_clicked = false;
  item_right_clicked_name.clear();
  ImGuiStyle& style = ImGui::GetStyle();
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 16.0f));
  ImGuiWindowClass window_class;
  //window_class.DockNodeFlagsOverrideSet = ImGuiDockNodeFlags_NoTabBar; TODO ?
  ImGui::SetNextWindowClass(&window_class);
  if (ImGui::Begin("Content Browser", 0, ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoTitleBar)) {
    if (ImGui::BeginMenuBar()) {
      ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.f, 0.f, 0.f, 0.f));
      ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.f, 0.f, 0.f, 0.f));
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.3f, 0.3f));

      if (fan::graphics::gui::image_button("##icon_arrow_left", icon_arrow_left, fan::vec2(32))) {
        if (std::filesystem::equivalent(current_directory, asset_path) == false) {
          current_directory = current_directory.parent_path();
        }
        update_directory_cache();
      }
      ImGui::SameLine();
      fan::graphics::gui::image_button("##icon_arrow_right", icon_arrow_right, fan::vec2(32));
      ImGui::SameLine();
      ImGui::PopStyleColor(3);

      ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.f, 0.f, 0.f, 0.f));
      ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.f, 0.f, 0.f, 0.f));
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.3f, 0.3f));

      auto image_list = std::to_array({ icon_files_list, icon_files_big_thumbnail });

      fan::vec2 bc = fan::graphics::gui::get_position_bottom_corner();

      bc.x -= ImGui::GetWindowPos().x;
      ImGui::SetCursorPosX(bc.x / 2);

      fan::vec2 button_sizes = 32;

      ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - (button_sizes.x * 2 + style.ItemSpacing.x) * image_list.size());

      ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 20.0f);
      ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 7.0f));
      f32_t y_pos = ImGui::GetCursorPosY() + ImGui::GetStyle().WindowPadding.y;
      ImGui::SetCursorPosY(y_pos);


      if (ImGui::InputText("##content_browser_search", search_buffer.data(), search_buffer.size())) {

      }
      ImGui::PopStyleVar(2);

      fan::graphics::gui::toggle_image_button(image_list, button_sizes, (int*)&current_view_mode);

      ImGui::PopStyleColor(3);

      ///ImGui::InputText("Search", search_buffer.data(), search_buffer.size());

      ImGui::EndMenuBar();
    }
    switch (current_view_mode) {
    case view_mode_large_thumbnails:
      render_large_thumbnails_view();
      break;
    case view_mode_list:
      render_list_view();
      break;
    default:
      break;
    }
  }

  ImGui::PopStyleVar(1);
  ImGui::End();
}

fan::graphics::gui::content_browser_t::content_browser_t() {
  search_buffer.resize(32);
  asset_path = std::filesystem::absolute(std::filesystem::path(asset_path)).wstring();
  current_directory = std::filesystem::path(asset_path);
  update_directory_cache();
}
fan::graphics::gui::content_browser_t::content_browser_t(bool no_init) {

}
fan::graphics::gui::content_browser_t::content_browser_t(const std::wstring& path) {
  search_buffer.resize(32);
  asset_path = std::filesystem::absolute(std::filesystem::path(asset_path)).wstring();
  current_directory = asset_path / std::filesystem::path(path);
  update_directory_cache();
}

void fan::graphics::gui::content_browser_t::update_directory_cache() {
  for (auto& img : directory_cache) {
    if (img.preview_image.iic() == false) {
      gloco->image_unload(img.preview_image);
    }
  }
  directory_cache.clear();
  fan::io::iterate_directory_sorted_by_name(current_directory, [this](const std::filesystem::directory_entry& path) {
    file_info_t file_info;
    // SLOW
    auto relative_path = std::filesystem::relative(path, asset_path);
    file_info.filename = relative_path.filename().string();
    file_info.item_path = relative_path.wstring();
    file_info.is_directory = path.is_directory();
    file_info.some_path = path.path().filename();//?
    //fan::print(get_file_extension(path.path().string()));
    if (fan::io::file::extension(path.path().string()) == ".webp" || fan::io::file::extension(path.path().string()) == ".png") {
      file_info.preview_image = gloco->image_load(std::filesystem::absolute(path.path()).string());
    }
    directory_cache.push_back(file_info);
  });
}

void fan::graphics::gui::content_browser_t::render_large_thumbnails_view() {
  float thumbnail_size = 128.0f;
  float panel_width = ImGui::GetContentRegionAvail().x;
  int column_count = std::max((int)(panel_width / (thumbnail_size + padding)), 1);

  ImGui::Columns(column_count, 0, false);

  int pressed_key = -1;
  
  auto& style = ImGui::GetStyle();
  // basically bad way to check if gui is disabled. I couldn't find other way
  if (style.DisabledAlpha != style.Alpha) {
    if (ImGui::IsWindowFocused()) {
      for (int i = ImGuiKey_A; i != ImGuiKey_Z + 1; ++i) {
        if (ImGui::IsKeyPressed((ImGuiKey)i, false)) {
          pressed_key = (i - ImGuiKey_A) + 'A';
          break;
        }
      }
    }
  }

  // Render thumbnails or icons
  for (std::size_t i = 0; i < directory_cache.size(); ++i) {
    // reference somehow corrupts
    auto file_info = directory_cache[i];
    if (std::string(search_buffer.c_str()).size() && file_info.filename.find(search_buffer) == std::string::npos) {
      continue;
    }

    if (pressed_key != -1 && ImGui::IsWindowFocused()) {
      if (!file_info.filename.empty() && std::tolower(file_info.filename[0]) == std::tolower(pressed_key)) {
        ImGui::SetScrollHereY();
      }
    }

    ImGui::PushID(file_info.filename.c_str());
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    fan::graphics::gui::image_button("##" + file_info.filename, file_info.preview_image.iic() == false ? file_info.preview_image : file_info.is_directory ? icon_directory : icon_file, ImVec2(thumbnail_size, thumbnail_size));

    bool item_hovered = ImGui::IsItemHovered();
    if (item_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
      item_right_clicked = true;
      item_right_clicked_name = file_info.filename;
      item_right_clicked_name.erase(std::remove_if(item_right_clicked_name.begin(), item_right_clicked_name.end(), isspace), item_right_clicked_name.end());
    }

    // Handle drag and drop, double click, etc.
    handle_item_interaction(file_info);

    ImGui::PopStyleColor();
    ImGui::TextWrapped("%s", file_info.filename.c_str());
    ImGui::NextColumn();
    ImGui::PopID();
  }

  ImGui::Columns(1);
}

void fan::graphics::gui::content_browser_t::render_list_view() {
  if (ImGui::BeginTable("##FileTable", 1, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY
    | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV
    | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_Sortable)) {
    ImGui::TableSetupColumn("##Filename", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableHeadersRow();

    int pressed_key = -1;
    ImGuiStyle& style = ImGui::GetStyle();
    if (style.DisabledAlpha != style.Alpha) {
      if (ImGui::IsWindowFocused()) {
        for (int i = ImGuiKey_A; i != ImGuiKey_Z + 1; ++i) {
          if (ImGui::IsKeyPressed((ImGuiKey)i, false)) {
            pressed_key = (i - ImGuiKey_A) + 'A';
            break;
          }
        }
      }
    }

    // Render table view
    for (std::size_t i = 0; i < directory_cache.size(); ++i) {

      // reference somehow corrupts
      auto file_info = directory_cache[i];

      if (pressed_key != -1 && ImGui::IsWindowFocused()) {
        if (!file_info.filename.empty() && std::tolower(file_info.filename[0]) == std::tolower(pressed_key)) {
          ImGui::SetScrollHereY();
        }
      }

      if (search_buffer.size() && strstr(file_info.filename.c_str(), search_buffer.c_str()) == nullptr) {
        continue;
      }
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0); // Icon column
      fan::vec2 cursor_pos = ImGui::GetWindowPos() + ImGui::GetCursorPos() + fan::vec2(ImGui::GetScrollX(), -ImGui::GetScrollY());
      fan::vec2 image_size = ImVec2(thumbnail_size / 4, thumbnail_size / 4);
      ImGuiStyle& style = ImGui::GetStyle();
      std::string space = "";
      while (ImGui::CalcTextSize(space.c_str()).x < image_size.x) {
        space += " ";
      }
      auto str = space + file_info.filename;

      ImGui::Selectable(str.c_str());
      bool item_hovered = ImGui::IsItemHovered();
      if (item_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        item_right_clicked_name = str;
        item_right_clicked_name.erase(std::remove_if(item_right_clicked_name.begin(), item_right_clicked_name.end(), isspace), item_right_clicked_name.end());
        item_right_clicked = true;
      }
      if (file_info.preview_image.iic() == false) {
        ImGui::GetWindowDrawList()->AddImage((ImTextureID)gloco->image_get_handle(file_info.preview_image), cursor_pos, cursor_pos + image_size);
      }
      else if (file_info.is_directory) {
        ImGui::GetWindowDrawList()->AddImage((ImTextureID)gloco->image_get_handle(icon_directory), cursor_pos, cursor_pos + image_size);
      }
      else {
        ImGui::GetWindowDrawList()->AddImage((ImTextureID)gloco->image_get_handle(icon_file), cursor_pos, cursor_pos + image_size);
      }

      handle_item_interaction(file_info);
    }

    ImGui::EndTable();
  }
}

void fan::graphics::gui::content_browser_t::handle_item_interaction(const file_info_t& file_info) {
  if (file_info.is_directory == false) {

    if (ImGui::BeginDragDropSource()) {
      ImGui::SetDragDropPayload("CONTENT_BROWSER_ITEM", file_info.item_path.data(), (file_info.item_path.size() + 1) * sizeof(wchar_t));
      ImGui::Text("%s", file_info.filename.c_str());
      ImGui::EndDragDropSource();
    }
  }

  if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
    if (file_info.is_directory) {
      current_directory /= file_info.some_path;
      update_directory_cache();
    }
  }
}

#endif

bool fan::graphics::texture_packe0::push_texture(fan::graphics::image_nr_t image, const texture_properties_t& texture_properties) {

  if (texture_properties.image_name.empty()) {
    fan::print_warning("texture properties name empty");
    return 1;
  }

  for (uint32_t gti = 0; gti < texture_list.size(); gti++) {
    if (texture_list[gti].image_name == texture_properties.image_name) {
      texture_list.erase(texture_list.begin() + gti);
      break;
    }
  }

  auto data = gloco->image_get_pixel_data(image, GL_RGBA, texture_properties.uv_pos, texture_properties.uv_size);

  fan::vec2ui image_size = 0;
  auto& image_data = gloco->image_get_data(image);
  image_size = {
    (uint32_t)(image_data.size.x * texture_properties.uv_size.x),
    (uint32_t)(image_data.size.y * texture_properties.uv_size.y)
  };

  if ((int)image_size.x % 2 != 0 || (int)image_size.y % 2 != 0) {
    fan::print_warning("failed to load, image size is not divideable by 2");
    fan::print(texture_properties.image_name, image_size);
    return 1;
  }

  texture_t t;
  t.size = image_size;
  t.decoded_data.resize(t.size.multiply() * 4);
  std::memcpy(t.decoded_data.data(), data.get(), t.size.multiply() * 4);
  t.image_name = texture_properties.image_name;
  t.visual_output = texture_properties.visual_output;
  t.min_filter = texture_properties.min_filter;
  t.mag_filter = texture_properties.mag_filter;
  t.group_id = texture_properties.group_id;

  texture_list.push_back(t);
  return 0;
}

#if defined(fan_json)
void fan::graphics::texture_packe0::load_compiled(const char* filename) {
  std::ifstream file(filename);
  fan::json j;
  file >> j;

  loaded_pack.resize(j["pack_amount"]);

  std::vector<loco_t::image_t> images;

  for (std::size_t i = 0; i < j["pack_amount"]; i++) {
    loaded_pack[i].texture_list.resize(j["packs"][i]["count"]);

    for (std::size_t k = 0; k < j["packs"][i]["count"]; k++) {
      pack_t::texture_t* t = &loaded_pack[i].texture_list[k];
      std::string image_name = j["packs"][i]["textures"][k]["image_name"];
      t->position = j["packs"][i]["textures"][k]["position"];
      t->size = j["packs"][i]["textures"][k]["size"];
      t->image_name = image_name;
    }

    std::vector<uint8_t> pixel_data = j["packs"][i]["pixel_data"].get<std::vector<uint8_t>>();
    fan::image::image_info_t image_info;
    image_info.data = WebPDecodeRGBA(
      pixel_data.data(),
      pixel_data.size(),
      &image_info.size.x,
      &image_info.size.y
    );
    loaded_pack[i].pixel_data = std::vector<uint8_t>((uint8_t*)image_info.data, (uint8_t*)image_info.data + image_info.size.x * image_info.size.y * 4);


    loaded_pack[i].visual_output = j["packs"][i]["visual_output"];
    loaded_pack[i].min_filter = j["packs"][i]["min_filter"];
    loaded_pack[i].mag_filter = j["packs"][i]["mag_filter"];
    images.push_back(gloco->image_load(image_info));
    WebPFree(image_info.data);
    for (std::size_t k = 0; k < loaded_pack[i].texture_list.size(); ++k) {
      auto& tl = loaded_pack[i].texture_list[k];
      fan::graphics::texture_packe0::texture_properties_t tp;
      tp.group_id = 0;
      tp.uv_pos = fan::vec2(tl.position) / fan::vec2(image_info.size);
      tp.uv_size = fan::vec2(tl.size) / fan::vec2(image_info.size);
      tp.visual_output = loaded_pack[i].visual_output;
      tp.min_filter = loaded_pack[i].min_filter;
      tp.mag_filter = loaded_pack[i].mag_filter;
      tp.image_name = tl.image_name;
      push_texture(images.back(), tp);
    }
  }
}//

#endif

void fan::camera::move(f32_t movement_speed, f32_t friction) {
  this->velocity /= friction * gloco->delta_time + 1;
  static constexpr auto minimum_velocity = 0.001;
  if (this->velocity.x < minimum_velocity && this->velocity.x > -minimum_velocity) {
    this->velocity.x = 0;
  }
  if (this->velocity.y < minimum_velocity && this->velocity.y > -minimum_velocity) {
    this->velocity.y = 0;
  }
  if (this->velocity.z < minimum_velocity && this->velocity.z > -minimum_velocity) {
    this->velocity.z = 0;
  }
  #if defined(fan_gui)
  if (!gloco->console.input.IsFocused()) {
#endif
    if (gloco->window.key_pressed(fan::input::key_w)) {
      this->velocity += this->m_front * (movement_speed * gloco->delta_time);
    }
    if (gloco->window.key_pressed(fan::input::key_s)) {
      this->velocity -= this->m_front * (movement_speed * gloco->delta_time);
    }
    if (gloco->window.key_pressed(fan::input::key_a)) {
      this->velocity -= this->m_right * (movement_speed * gloco->delta_time);
    }
    if (gloco->window.key_pressed(fan::input::key_d)) {
      this->velocity += this->m_right * (movement_speed * gloco->delta_time);
    }

    if (gloco->window.key_pressed(fan::input::key_space)) {
      this->velocity.y += movement_speed * gloco->delta_time;
    }
    if (gloco->window.key_pressed(fan::input::key_left_shift)) {
      this->velocity.y -= movement_speed * gloco->delta_time;
    }

    if (gloco->window.key_pressed(fan::input::key_left)) {
      this->set_yaw(this->get_yaw() - sensitivity * 100 * gloco->delta_time);
    }
    if (gloco->window.key_pressed(fan::input::key_right)) {
      this->set_yaw(this->get_yaw() + sensitivity * 100 * gloco->delta_time);
    }
    if (gloco->window.key_pressed(fan::input::key_up)) {
      this->set_pitch(this->get_pitch() + sensitivity * 100 * gloco->delta_time);
    }
    if (gloco->window.key_pressed(fan::input::key_down)) {
      this->set_pitch(this->get_pitch() - sensitivity * 100 * gloco->delta_time);
    }
  #if defined(fan_gui)
  }
#endif

  this->position += this->velocity * gloco->delta_time;
  this->update_view();
}

fan::graphics::interactive_camera_t::interactive_camera_t(loco_t::camera_t camera_nr, loco_t::viewport_t viewport_nr) :
  reference_camera(camera_nr), reference_viewport(viewport_nr)
{
  auto& window = gloco->window;
  static auto update_ortho = [&] (loco_t* loco){
    fan::vec2 s = loco->viewport_get_size(reference_viewport);
    loco->camera_set_ortho(
      reference_camera,
      fan::vec2(-s.x, s.x) / zoom,
      fan::vec2(-s.y, s.y) / zoom
    );
  };

  auto it = gloco->m_update_callback.NewNodeLast();
  gloco->m_update_callback[it] = update_ortho;

  button_cb_nr = window.add_buttons_callback([&](const auto& d) {
    if (d.button == fan::mouse_scroll_up) {
      zoom *= 1.2;
    }
    else if (d.button == fan::mouse_scroll_down) {
      zoom /= 1.2;
    }
  });
}

fan::graphics::interactive_camera_t::~interactive_camera_t() {
  if (button_cb_nr.iic() == false) {
    gloco->window.remove_buttons_callback(button_cb_nr);
    button_cb_nr.sic();
  }
  if (uc_nr.iic() == false) {
    gloco->m_update_callback.unlrec(uc_nr);
    uc_nr.sic();
  }
}

#if defined(fan_gui)

// called in loop
void fan::graphics::interactive_camera_t::move_by_cursor() {
  if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
    fan::vec2 drag_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Middle);
    gloco->camera_set_position(
      gloco->orthographic_camera.camera,
      gloco->camera_get_position(gloco->orthographic_camera.camera) - (drag_delta / zoom) * 2
    );
    ImGui::ResetMouseDragDelta(ImGuiMouseButton_Middle);
  }
}

fan::graphics::dialogue_box_t::dialogue_box_t() {
  gloco->input_action.add(fan::mouse_left, "skip or continue dialog");
}

void fan::graphics::dialogue_box_t::set_cursor_position(const fan::vec2& cursor_position) {
  this->cursor_position = cursor_position;
}

fan::ev::task_t fan::graphics::dialogue_box_t::text(const std::string& text) {
  active_dialogue = text;
  render_pos = 0;
  finish_dialog = false;
  while (render_pos < active_dialogue.size() && !finish_dialog) {
    ++render_pos;
    co_await fan::co_sleep(1000 / character_per_s);
  }
  render_pos = active_dialogue.size();
}

fan::ev::task_t fan::graphics::dialogue_box_t::button(const std::string& text, const fan::vec2& position, const fan::vec2& size) {
  button_choice = -1;
  button_t button;
  button.position = position;
  button.size = size;
  button.text = text;
  buttons.push_back(button);
  co_return;
}

int fan::graphics::dialogue_box_t::get_button_choice() const {
  return button_choice;
}

fan::ev::task_t fan::graphics::dialogue_box_t::wait_user_input() {
  wait_user = true;
  fan::time::clock c;
  c.start(0.5e9);
  int prev_render = render_pos;
  while (wait_user) {
    if (c.finished()) {
      if (prev_render == render_pos) {
        render_pos = std::max(prev_render - 1, 0);
      }
      else {
        render_pos = prev_render;
      }
      c.restart();
    }
    co_await fan::co_sleep(10);
  }
  render_pos = prev_render;
}

void fan::graphics::dialogue_box_t::render(const std::string& window_name, ImFont* font, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing) {
  ImGui::PushFont(font);
  fan::vec2 root_window_size = ImGui::GetWindowSize();
  fan::vec2 next_window_pos;
  next_window_pos.x = (root_window_size.x - window_size.x) / 2.0f;
  next_window_pos.y = (root_window_size.y - window_size.y) / 1.1;
  ImGui::SetNextWindowPos(next_window_pos);

  ImGui::SetNextWindowSize(window_size);
  ImGui::Begin(window_name.c_str(), 0, 
    ImGuiWindowFlags_NoNavInputs | ImGuiWindowFlags_NoTitleBar | 
    ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar
  );
  ImGui::SetCursorPos(ImVec2(100.0f, 100.f));
  ImGui::BeginChild((window_name + "child").c_str(), fan::vec2(wrap_width, 0), 0, ImGuiWindowFlags_NoNavInputs | ImGuiWindowFlags_NoTitleBar | 
    ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoBackground);
  if (wait_user == false) {
    ImGui::SetScrollY(ImGui::GetScrollMaxY());
  }
  fan::graphics::text_partial_render(active_dialogue.c_str(), render_pos, wrap_width, line_spacing);
  ImGui::EndChild();
  if (wait_user) {
    fan::vec2 first_pos = -1;


    // calculate biggest button
    fan::vec2 button_size = 0;
    for (std::size_t i = 0; i < buttons.size(); ++i) {
      fan::vec2 text_size = ImGui::CalcTextSize(buttons[i].text.c_str());
      float padding_x = ImGui::GetStyle().FramePadding.x; 
      float padding_y = ImGui::GetStyle().FramePadding.y; 
      ImVec2 bs = ImVec2(text_size.x + padding_x * 2.0f, text_size.y + padding_y * 2.0f);
      button_size = button_size.max(fan::vec2(bs));
    }

    for (std::size_t i = 0; i < buttons.size(); ++i) {
      const auto& button = buttons[i];
      if (button.position != -1) {
        first_pos = button.position;
        ImGui::SetCursorPos((button.position * window_size) - button_size / 2);
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + ImGui::GetScrollY());
      }
      else {
        ImGui::SetCursorPosX(first_pos.x * window_size.x - button_size.x / 2);
      }
      ImGui::PushID(i);

      if (fan::graphics::gui::image_text_button(gloco->default_texture, button.text.c_str(), fan::colors::white, button.size == 0 ? button_size : button.size)) {
        button_choice = i;
        buttons.clear();
        wait_user = false;
        ImGui::PopID();
        break;
      }
      ImGui::PopID();
    }
  }
  if (gloco->input_action.is_active("skip or continue dialog") && ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows | ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem  )) {
    finish_dialog = true;
    wait_user = false;
  }
  ImGui::End();
  ImGui::PopFont();
}

#endif

#if defined(fan_physics)
bool fan::physics::is_on_sensor(fan::physics::body_id_t test_id, fan::physics::body_id_t sensor_id){
  return gloco->physics_context.is_on_sensor(test_id, sensor_id);
}

fan::physics::ray_result_t fan::physics::raycast(const fan::vec2& src, const fan::vec2& dst) {
  return gloco->physics_context.raycast(src, dst);
}
#endif

fan::vec2 fan::window::get_size(){
  return gloco->window.get_size();
}

bool fan::window::is_mouse_clicked(int button) {
  return gloco->is_mouse_clicked(button);
}

bool fan::window::is_mouse_down(int button) {
  return gloco->is_mouse_down(button);
}

bool fan::window::is_mouse_released(int button) {
  return gloco->is_mouse_released(button);
}

fan::vec2 fan::window::get_mouse_drag(int button) {
  return gloco->get_mouse_drag(button);
}

bool fan::window::is_key_pressed(int key) {
  return gloco->is_key_pressed(key);
}

bool fan::window::is_key_down(int key) {
  return gloco->is_key_down(key);
}

bool fan::window::is_key_released(int key) {
  return gloco->is_key_released(key);
}

void fan::window::set_size(const fan::vec2& size) {
  gloco->window.set_size(size);
  gloco->viewport_set(gloco->orthographic_camera.viewport, fan::vec2(0, 0), size, size);
  gloco->camera_set_ortho(
    gloco->orthographic_camera.camera, 
    fan::vec2(0, size.x),
    fan::vec2(0, size.y)
  );
  gloco->viewport_set(gloco->perspective_camera.viewport, fan::vec2(0, 0), size, size);
  gloco->camera_set_ortho(
    gloco->perspective_camera.camera, 
    fan::vec2(0, size.x),
    fan::vec2(0, size.y)
  );
}

fan::graphics::image_nr_t fan::graphics::image_load(const fan::image::image_info_t& image_info) {
  return gloco->image_load(image_info);
}

fan::graphics::image_nr_t fan::graphics::image_load(const fan::image::image_info_t& image_info, const fan::graphics::image_load_properties_t& p) {
  return gloco->image_load(image_info, p);
}

fan::graphics::image_nr_t fan::graphics::image_load(const fan::string& path) {
  return gloco->image_load(path);
}

fan::graphics::image_nr_t fan::graphics::image_load(const fan::string& path, const fan::graphics::image_load_properties_t& p) {
  return gloco->image_load(path, p);
}

fan::graphics::image_nr_t fan::graphics::image_load(fan::color* colors, const fan::vec2ui& size) {
  return gloco->image_load(colors, size);
}

fan::graphics::image_nr_t fan::graphics::image_load(fan::color* colors, const fan::vec2ui& size, const fan::graphics::image_load_properties_t& p) {
  return gloco->image_load(colors, size, p);
}

void fan::graphics::image_unload(fan::graphics::image_nr_t nr){
  return gloco->image_unload(nr);
}

#if defined(fan_gui)

static constexpr int wnd_flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | 
    ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar;

void loco_t::settings_menu_t::menu_graphics_left(settings_menu_t* menu, const fan::vec2& next_window_position, const fan::vec2& next_window_size) {

  ImGui::SetNextWindowPos(next_window_position);
  ImGui::SetNextWindowSize(next_window_size);
  ImGui::Begin("##Menu Graphics Left", 0, wnd_flags);
  {
    ImGui::TextColored(fan::color::hex(0x948c80ff) * 1.5, "DISPLAY");
    ImGui::BeginTable("settings_left_table_display", 2,
      ImGuiTableFlags_BordersInnerH |
      ImGuiTableFlags_BordersOuterH
    );
    {
      ImGui::TableNextRow();
      menu->render_display_mode();
      ImGui::TableNextRow();
      menu->render_target_fps();
      ImGui::TableNextRow();
      menu->render_resolution_dropdown();

      {
        static const char* renderers[] = {
          "OpenGL",
          "Vulkan",
        };
        ImGui::TableNextColumn();
        ImGui::Text("Renderer");
        ImGui::TableNextColumn();
        if (ImGui::BeginCombo("##Renderer", renderers[gloco->window.renderer])) {
          for (int i = 0; i < std::size(renderers); ++i) {
            bool is_selected = (gloco->window.renderer == i);
            if (ImGui::Selectable(renderers[i], is_selected)) {
              switch (i) {
              case 0: {
                if (gloco->window.renderer != fan::window_t::renderer_t::opengl) {
                  gloco->reload_renderer_to = fan::window_t::renderer_t::opengl;
                }
                break;
              }
              case 1: {
                if (gloco->window.renderer != fan::window_t::renderer_t::vulkan) {
                  gloco->reload_renderer_to = fan::window_t::renderer_t::vulkan;
                }
                break;
              }
              }
            }
            if (is_selected) {
              ImGui::SetItemDefaultFocus();
            }
          }
          ImGui::EndCombo();
        }
      }
    }

    ImGui::EndTable();
  }
  ImGui::NewLine();
  ImGui::NewLine();
  {
    ImGui::TextColored(fan::color::hex(0x948c80ff) * 1.5, "POST PROCESSING");
    ImGui::BeginTable("settings_left_table_post_processing", 2,
      ImGuiTableFlags_BordersInnerH |
      ImGuiTableFlags_BordersOuterH
    );

    {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Bloom Strength");
      ImGui::TableNextColumn();
      if (ImGui::SliderFloat("##BloomStrengthSlider", &menu->bloom_strength, 0, 1)) {
        if (gloco->window.renderer == fan::window_t::renderer_t::opengl) {
          gloco->shader_set_value(gloco->gl.m_fbo_final_shader, "bloom_strength", menu->bloom_strength);
        }
      }
    }

    ImGui::EndTable();
  }
  ImGui::NewLine();
  ImGui::NewLine();
  {
    ImGui::TextColored(fan::color::hex(0x948c80ff) * 1.5, "PERFORMANCE STATS");
    ImGui::BeginTable("settings_left_table_post_processing", 2,
      ImGuiTableFlags_BordersInnerH |
      ImGuiTableFlags_BordersOuterH
    );
    {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Show fps");
      ImGui::TableNextColumn();
      ImGui::Checkbox("##show_fps", (bool*)&gloco->toggle_fps);
    }
    {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Track OpenGL calls");
      ImGui::TableNextColumn();
      ImGui::Checkbox("##track_opengl_calls", (bool*)&fan_track_opengl_calls);
    }

    ImGui::EndTable();
  }
  ImGui::End();
}

void loco_t::settings_menu_t::menu_graphics_right(loco_t::settings_menu_t* menu, const fan::vec2& next_window_position, const fan::vec2& next_window_size) {

  ImGui::SetNextWindowPos(next_window_position);
  ImGui::SetNextWindowSize(next_window_size);
  ImGui::Begin("##Menu Graphics Right", 0, wnd_flags);

  ImGui::PushFont(gloco->fonts_bold[std::size(gloco->fonts_bold) - 3]);
  ImGui::Indent(menu->min_x);
  ImGui::TextColored(fan::color::hex(0x948c80ff) * 1.5, "Setting Info");
  ImGui::Unindent(menu->min_x);
  ImGui::PopFont();

  ImVec2 cursor_pos = ImGui::GetCursorPos();
  ImDrawList* draw_list = ImGui::GetWindowDrawList();
  ImVec2 line_start = ImGui::GetCursorScreenPos();
  line_start.x -= cursor_pos.x;
  line_start.y -= cursor_pos.y;

  ImVec2 line_end = line_start;
  line_end.y += ImGui::GetContentRegionMax().y;

  draw_list->AddLine(line_start, line_end, IM_COL32(255, 255, 255, 255));

  ImGui::End();
}

void loco_t::settings_menu_t::menu_audio_left(loco_t::settings_menu_t* menu, const fan::vec2& next_window_position, const fan::vec2& next_window_size) {

}

void loco_t::settings_menu_t::menu_audio_right(loco_t::settings_menu_t* menu, const fan::vec2& next_window_position, const fan::vec2& next_window_size) {
  loco_t::settings_menu_t::menu_graphics_right(menu, next_window_position, next_window_size);
}

void loco_t::settings_menu_t::set_settings_theme() {
  ImGuiStyle& style = ImGui::GetStyle();

  style.Alpha = 1.0f;
  style.DisabledAlpha = 0.5f;
  style.WindowPadding = ImVec2(13.0f, 10.0f);
  style.WindowRounding = 0.0f;
  style.WindowBorderSize = 1.0f;
  style.WindowMinSize = ImVec2(32.0f, 32.0f);
  style.WindowTitleAlign = ImVec2(0.5f, 0.5f);
  style.WindowMenuButtonPosition = ImGuiDir_Right;
  style.ChildRounding = 3.0f;
  style.ChildBorderSize = 1.0f;
  style.PopupRounding = 5.0f;
  style.PopupBorderSize = 1.0f;
  style.FramePadding = ImVec2(20.0f, 8.100000381469727f);
  style.FrameRounding = 2.0f;
  style.FrameBorderSize = 0.0f;
  style.ItemSpacing = ImVec2(3.0f, 3.0f);
  style.ItemInnerSpacing = ImVec2(3.0f, 8.0f);
  style.CellPadding = ImVec2(6.0f, 14.10000038146973f);
  style.IndentSpacing = 0.0f;
  style.ColumnsMinSpacing = 10.0f;
  style.ScrollbarSize = 10.0f;
  style.ScrollbarRounding = 2.0f;
  style.GrabMinSize = 12.10000038146973f;
  style.GrabRounding = 1.0f;
  style.TabRounding = 2.0f;
  style.TabBorderSize = 0.0f;
  style.TabMinWidthForCloseButton = 5.0f;
  style.ColorButtonPosition = ImGuiDir_Right;
  style.ButtonTextAlign = ImVec2(0.5f, 0.5f);
  style.SelectableTextAlign = ImVec2(0.0f, 0.0f);

  style.Colors[ImGuiCol_Text] = ImVec4(0.9803921580314636f, 0.9803921580314636f, 0.9803921580314636f, 1.0f);
  style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.4980392158031464f, 0.4980392158031464f, 0.4980392158031464f, 1.0f);
  style.Colors[ImGuiCol_WindowBg] = ImVec4(0.09411764889955521f, 0.09411764889955521f, 0.09411764889955521f, 0.9);
  style.Colors[ImGuiCol_ChildBg] = ImVec4(0.1568627506494522f, 0.1568627506494522f, 0.1568627506494522f, 1.0f);
  style.Colors[ImGuiCol_PopupBg] = ImVec4(0.09411764889955521f, 0.09411764889955521f, 0.09411764889955521f, 1.0f);
  style.Colors[ImGuiCol_Border] = ImVec4(1.0f, 1.0f, 1.0f, 0.09803921729326248f);
  style.Colors[ImGuiCol_BorderShadow] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
  style.Colors[ImGuiCol_FrameBg] = ImVec4(1.0f, 1.0f, 1.0f, 0.09803921729326248f);
  style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(1.0f, 1.0f, 1.0f, 0.1568627506494522f);
  style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.0f, 0.0f, 0.0f, 0.0470588244497776f);
  style.Colors[ImGuiCol_TitleBg] = ImVec4(0.1176470592617989f, 0.1176470592617989f, 0.1176470592617989f, 1.0f);
  style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.1568627506494522f, 0.1568627506494522f, 0.1568627506494522f, 1.0f);
  style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.1176470592617989f, 0.1176470592617989f, 0.1176470592617989f, 1.0f);
  style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.09411764889955521f, 0.09411764889955521f, 0.09411764889955521f, 1.f);
  style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.1098039224743843f);
  style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(1.0f, 1.0f, 1.0f, 0.3921568691730499f);
  style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(1.0f, 1.0f, 1.0f, 0.4705882370471954f);
  style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.0f, 0.0f, 0.0f, 0.09803921729326248f);
  style.Colors[ImGuiCol_CheckMark] = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
  style.Colors[ImGuiCol_SliderGrab] = ImVec4(1.0f, 1.0f, 1.0f, 0.3921568691730499f);
  style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(1.0f, 1.0f, 1.0f, 0.3137255012989044f);
  style.Colors[ImGuiCol_Button] = ImVec4(1.0f, 1.0f, 1.0f, 0.09803921729326248f);
  style.Colors[ImGuiCol_ButtonHovered] = ImVec4(1.0f, 1.0f, 1.0f, 0.1568627506494522f);
  style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.0f, 0.0f, 0.0f, 0.0470588244497776f);
  style.Colors[ImGuiCol_Header] = ImVec4(1.0f, 1.0f, 1.0f, 0.09803921729326248f);
  style.Colors[ImGuiCol_HeaderHovered] = ImVec4(1.0f, 1.0f, 1.0f, 0.1568627506494522f);
  style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.0f, 0.0f, 0.0f, 0.0470588244497776f);
  style.Colors[ImGuiCol_Separator] = ImVec4(1.0f, 1.0f, 1.0f, 0.1568627506494522f);
  style.Colors[ImGuiCol_SeparatorHovered] = ImVec4(1.0f, 1.0f, 1.0f, 0.2352941185235977f);
  style.Colors[ImGuiCol_SeparatorActive] = ImVec4(1.0f, 1.0f, 1.0f, 0.2352941185235977f);
  style.Colors[ImGuiCol_ResizeGrip] = ImVec4(1.0f, 1.0f, 1.0f, 0.1568627506494522f);
  style.Colors[ImGuiCol_ResizeGripHovered] = ImVec4(1.0f, 1.0f, 1.0f, 0.2352941185235977f);
  style.Colors[ImGuiCol_ResizeGripActive] = ImVec4(1.0f, 1.0f, 1.0f, 0.2352941185235977f);
  style.Colors[ImGuiCol_Tab] = ImVec4(1.0f, 1.0f, 1.0f, 0.09803921729326248f);
  style.Colors[ImGuiCol_TabHovered] = ImVec4(1.0f, 1.0f, 1.0f, 0.1568627506494522f);
  style.Colors[ImGuiCol_TabActive] = ImVec4(1.0f, 1.0f, 1.0f, 0.3137255012989044f);
  style.Colors[ImGuiCol_TabUnfocused] = ImVec4(0.0f, 0.0f, 0.0f, 0.1568627506494522f);
  style.Colors[ImGuiCol_TabUnfocusedActive] = ImVec4(1.0f, 1.0f, 1.0f, 0.2352941185235977f);
  style.Colors[ImGuiCol_PlotLines] = ImVec4(1.0f, 1.0f, 1.0f, 0.3529411852359772f);
  style.Colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
  style.Colors[ImGuiCol_PlotHistogram] = ImVec4(1.0f, 1.0f, 1.0f, 0.3529411852359772f);
  style.Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
  style.Colors[ImGuiCol_TableHeaderBg] = ImVec4(0.1568627506494522f, 0.1568627506494522f, 0.1568627506494522f, 1.0f);
  style.Colors[ImGuiCol_TableBorderStrong] = ImVec4(1.0f, 1.0f, 1.0f, 0.3137255012989044f);
  style.Colors[ImGuiCol_TableBorderLight] = ImVec4(1.0f, 1.0f, 1.0f, 0.196078434586525f);
  style.Colors[ImGuiCol_TableRowBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
  style.Colors[ImGuiCol_TableRowBgAlt] = ImVec4(1.0f, 1.0f, 1.0f, 0.01960784383118153f);
  style.Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.0f, 0.0f, 0.0f, 1.0f);
  style.Colors[ImGuiCol_DragDropTarget] = ImVec4(0.168627455830574f, 0.2313725501298904f, 0.5372549295425415f, 1.0f);
  style.Colors[ImGuiCol_NavHighlight] = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
  style.Colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.0f, 1.0f, 1.0f, 0.699999988079071f);
  style.Colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.800000011920929f, 0.800000011920929f, 0.800000011920929f, 0.2000000029802322f);
  style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.5647059082984924f);
}

void loco_t::settings_menu_t::open() {
  set_settings_theme();
  page_t page;
  {
    page.toggle = 1,
    page.name = "Graphics";
    page.page_left_render = loco_t::settings_menu_t::menu_graphics_left;
    page.page_right_render = loco_t::settings_menu_t::menu_graphics_right;
    pages.emplace_back(page);
  }
  {
    page.toggle = 0;
    page.name = "Audio";
    page.page_left_render = loco_t::settings_menu_t::menu_audio_left;
    page.page_right_render = loco_t::settings_menu_t::menu_audio_right;
    pages.emplace_back(page);
  }
  if (gloco->window.renderer == fan::window_t::renderer_t::opengl) {
    gloco->shader_set_value(gloco->gl.m_fbo_final_shader, "bloom_strength", bloom_strength);
  }
}

void loco_t::settings_menu_t::change_target_fps(int direction) {
  int index = 0;
  for (int i = 0; i < std::size(fps_values); ++i) {
    if (fps_values[i] == gloco->target_fps) {
      index = i;
      break;
    }
  }
  index = (index + direction + std::size(fps_values)) % std::size(fps_values);
  gloco->set_target_fps(fps_values[index]);
}

void loco_t::settings_menu_t::render_display_mode() {
  static const char* display_mode_names[] = {
    "Windowed",
    "Borderless",
    "Windowed Fullscreen",
    "Fullscreen",
  };
  //ImGui::GetStyle().align
  ImGui::TableNextColumn();
  ImGui::Text("Display Mode");
  ImGui::TableNextColumn();
  if (ImGui::BeginCombo("##Display Mode", display_mode_names[gloco->window.display_mode - 1])) {
    for (int i = 0; i < std::size(display_mode_names); ++i) {
      bool is_selected = (gloco->window.display_mode - 1 == i);
      if (ImGui::Selectable(display_mode_names[i], is_selected)) {
        gloco->window.set_display_mode((fan::window_t::mode)(i + 1));
      }
      if (is_selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }
}

void loco_t::settings_menu_t::render_target_fps() {
  ImGui::TableNextColumn();
  ImGui::Text("Target Framerate");
  ImGui::TableNextColumn();
  ImGui::SameLine();
  if (ImGui::ArrowButton("##left_arrow", ImGuiDir_Left)) {
    change_target_fps(-1);
  }
  ImGui::SameLine();
  ImGui::Text("%d", gloco->target_fps);
  ImGui::SameLine();
  if (ImGui::ArrowButton("##right_arrow", ImGuiDir_Right)) {
    change_target_fps(1);
  }
}

void loco_t::settings_menu_t::render_resolution_dropdown() {
  fan::vec2i current_size = gloco->window.get_size();

  int current_resolution = -1;
  for (int i = 0; i < std::size(fan::window_t::resolutions); ++i) {
    if (fan::window_t::resolutions[i] == current_size) {
      current_resolution = i;
      break;
    }
  }

  if (current_resolution == -1) {
    current_resolution = std::size(fan::window_t::resolutions);
  }

  ImGui::TableNextColumn();
  ImGui::Text("Resolution");
  ImGui::TableNextColumn();

  fan::vec2i window_size = gloco->window.get_size();
  std::string custom_res = std::to_string(window_size.x) + "x" + std::to_string(window_size.y);
  const char* current_label = (current_resolution == std::size(fan::window_t::resolutions)) ?
    custom_res.c_str() : fan::window_t::resolution_labels[current_resolution];

  if (ImGui::BeginCombo("##ResolutionCombo", current_label)) {
    for (int i = 0; i < std::size(fan::window_t::resolution_labels); ++i) {
      bool is_selected = (current_resolution == i);
      if (ImGui::Selectable(fan::window_t::resolution_labels[i], is_selected)) {
        current_resolution = i;
        gloco->window.set_size(fan::window_t::resolutions[i]);
      }
      if (is_selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    if (current_resolution == -1 && ImGui::Selectable(custom_res.c_str(), current_resolution == std::size(fan::window_t::resolutions))) {
      current_resolution = std::size(fan::window_t::resolutions);
    }
    ImGui::EndCombo();
  }
}

void loco_t::settings_menu_t::render_separator_with_margin(f32_t width, f32_t margin) {
  ImVec2 separator_start = ImGui::GetCursorScreenPos();
  ImVec2 separator_end = ImVec2(separator_start.x + width - margin * 2, separator_start.y);
  separator_start.x += margin;

  ImGui::GetWindowDrawList()->AddLine(separator_start, separator_end, ImGui::GetColorU32(ImGuiCol_Separator), 1.0f);
}

void loco_t::settings_menu_t::render_settings_left(const fan::vec2& next_window_position, const fan::vec2& next_window_size) {
  pages[current_page].page_left_render(this, next_window_position, next_window_size);
}

void loco_t::settings_menu_t::render_settings_right(const fan::vec2& next_window_position, const fan::vec2& next_window_size, f32_t min_x) {
  pages[current_page].page_right_render(this, next_window_position, next_window_size);
}

fan::vec2 loco_t::settings_menu_t::render_settings_top(f32_t min_x) {
  fan::vec2 main_window_size = gloco->window.get_size();
  ImGui::SetNextWindowPos(ImVec2(0, 0));
  ImGui::SetNextWindowSize(fan::vec2(main_window_size.x, main_window_size.y / 5));
  ImGui::Begin("##Fan Settings Nav", 0,
    ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | 
    ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar
  );

  ImGui::PushFont(gloco->fonts_bold[std::size(gloco->fonts_bold) - 2]);
  ImGui::Indent(min_x);
  ImGui::Text("Settings");
  ImGui::PopFont();

  render_separator_with_margin(ImGui::GetContentRegionAvail().x - min_x);
  f32_t options_x = 256.f;
  ImGui::Indent(options_x);
  ImGui::PushFont(gloco->fonts_bold[2]);
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(64, 5.f));
  ImGui::BeginTable("##settings_top_table", pages.size());
  ImGui::TableNextRow();
  for (std::size_t i = 0; i < std::size(pages); ++i) {
    ImGui::TableNextColumn();
    bool& is_toggled = pages[i].toggle;
    if (is_toggled) {
      ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonHovered));
    }
    else {
      ImGui::PushStyleColor(ImGuiCol_Button, fan::colors::transparent);
    }
    if (ImGui::Button(pages[i].name.c_str())) {
      pages[i].toggle = !pages[i].toggle;
      if (pages[i].toggle) {
        reset_page_selection();
        pages[i].toggle = 1;
        current_page = i;
      }
    }
    ImGui::PopStyleColor();
  }

  ImGui::EndTable();
  ImGui::PopStyleVar();

  ImGui::PopFont();
  ImGui::Unindent(options_x);
  render_separator_with_margin(ImGui::GetContentRegionAvail().x - min_x);
  fan::vec2 window_size = ImGui::GetWindowSize();
  ImGui::Unindent();
  ImGui::End();
  return window_size;
}

void loco_t::settings_menu_t::render() {
  if (gloco->reload_renderer_to != (decltype(gloco->reload_renderer_to))-1) {
    set_settings_theme();
  }

  ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.09411764889955521f, 0.09411764889955521f, 0.09411764889955521f, 0.9f));
  ImGui::PushStyleColor(ImGuiCol_Separator, ImVec4(0.8, 0.8, 0.8, 1.0f));

  fan::vec2 main_window_size = gloco->window.get_size();
  fan::vec2 window_size = render_settings_top(min_x);

  fan::vec2 next_window_position = fan::vec2(0, window_size.y);
  fan::vec2 next_window_size = fan::vec2(main_window_size.x / 2.f, main_window_size.y - next_window_position.y);
  render_settings_left(next_window_position, next_window_size);

  next_window_position = next_window_position + fan::vec2(main_window_size.x / 2.f, 0);
  next_window_size = fan::vec2(main_window_size.x / 2.f, main_window_size.y - next_window_position.y);
  render_settings_right(next_window_position, next_window_size, min_x);

  ImGui::PopStyleColor(2);
}

void loco_t::settings_menu_t::reset_page_selection() {
  for (auto& page : pages) {
    page.toggle = 0;
  }
}

#endif

loco_t::polygon_t::properties_t fan::graphics::create_sine_ground(const fan::vec2& position, f32_t amplitude, f32_t frequency, f32_t width, f32_t groundWidth) {
  loco_t::polygon_t::properties_t pp;
  // for triangle strip
  for (f32_t x = 0; x < groundWidth - width; x += width) {
    f32_t y1 = position.y / 2 + amplitude * std::sin(frequency * x);
    f32_t y2 = position.y / 2 + amplitude * std::sin(frequency * (x + width));

    // top
    pp.vertices.push_back({ fan::vec2(position.x + x, y1), fan::colors::red });
    // bottom
    pp.vertices.push_back({ fan::vec2(position.x + x, position.y), fan::colors::white });
    // next top
    pp.vertices.push_back({ fan::vec2(position.x + x + width, y2), fan::colors::red });
    // next bottom
    pp.vertices.push_back({ fan::vec2(position.x + x + width, position.y), fan::colors::white });
  }

  return pp;
}

std::vector<fan::vec2> fan::graphics::ground_points(const fan::vec2& position, f32_t amplitude, f32_t frequency, f32_t width, f32_t groundWidth) {
  std::vector<fan::vec2> outline_points;
  for (f32_t x = 0; x <= groundWidth; x += width) {
    f32_t y = position.y / 2 + amplitude * std::sin(frequency * x);
    outline_points.push_back(fan::vec2(position.x + x, y));
  }
  outline_points.push_back(fan::vec2(position.x + groundWidth, position.y));
  outline_points.push_back(fan::vec2(position.x, position.y));
  return outline_points;
}