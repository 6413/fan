#include <fan/graphics/graphics.h>

#if defined(loco_imgui)
  #include <fan/imgui/imgui_internal.h>
  #include <fan/graphics/gui/imgui_themes.h>
#endif

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

fan::vec2 fan::graphics::get_mouse_position(const loco_t::camera_t& camera, const loco_t::viewport_t& viewport) {
  return loco_t::transform_position(gloco->get_mouse_position(), viewport, camera);
}

#if defined(loco_imgui)

void fan::graphics::text(const std::string& text, const fan::vec2& position, const fan::color& color) {
  ImGui::SetCursorPos(position);
  ImGui::PushStyleColor(ImGuiCol_Text, color);
  ImGui::Text("%s", text.c_str());
  ImGui::PopStyleColor();
}
void fan::graphics::text_bottom_right(const std::string& text, const fan::color& color, const fan::vec2& offset) {
  ImVec2 text_pos;
  ImVec2 text_size = ImGui::CalcTextSize(text.c_str());
  ImVec2 window_pos = ImGui::GetWindowPos();
  ImVec2 window_size = ImGui::GetWindowSize();

  text_pos.x = window_pos.x + window_size.x - text_size.x - ImGui::GetStyle().WindowPadding.x;
  text_pos.y = window_pos.y + window_size.y - text_size.y - ImGui::GetStyle().WindowPadding.y;
  fan::graphics::text(text, text_pos + offset, color);
}
IMGUI_API void ImGui::Image(loco_t::image_t img, const ImVec2& size, const ImVec2& uv0, const ImVec2& uv1, const ImVec4& tint_col, const ImVec4& border_col) {
  ImGui::Image((ImTextureID)gloco->image_get_handle(img), size, uv0, uv1, tint_col, border_col);
}
IMGUI_API bool ImGui::ImageButton(const std::string& str_id, loco_t::image_t img, const ImVec2& size, const ImVec2& uv0, const ImVec2& uv1, int frame_padding, const ImVec4& bg_col, const ImVec4& tint_col) {
  return ImGui::ImageButton(str_id.c_str(), (ImTextureID)gloco->image_get_handle(img), size, uv0, uv1, bg_col, tint_col);
}
IMGUI_API bool ImGui::ImageTextButton(loco_t::image_t img, const std::string& text, const fan::color& color, const ImVec2& size, const ImVec2& uv0, const ImVec2& uv1, int frame_padding, const ImVec4& bg_col, const ImVec4& tint_col) {

  bool ret = ImGui::ImageButton(text.c_str(), (ImTextureID)gloco->image_get_handle(img), size, uv0, uv1, bg_col, tint_col);
  ImVec2 text_size = ImGui::CalcTextSize(text.c_str());
  ImVec2 pos = ImGui::GetItemRectMin(); 
  pos.x += (size.x - text_size.x) * 0.5f;
  pos.y += (size.y - text_size.y) * 0.5f;
  ImGui::GetWindowDrawList()->AddText(pos, ImGui::GetColorU32(color), text.c_str());
  return ret;
}
bool ImGui::ToggleButton(const std::string& str, bool* v) {

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
bool ImGui::ToggleImageButton(const std::string& char_id, loco_t::image_t image, const ImVec2& size, bool* toggle)
{
  bool clicked = false;

  ImVec4 tintColor = ImVec4(1, 1, 1, 1);
  if (*toggle) {
  //  tintColor = ImVec4(0.3f, 0.3f, 0.3f, 1.0f);
  }
  if (ImGui::IsItemHovered()) {
  //  tintColor = ImVec4(0.6f, 0.6f, 0.6f, 1.0f);
  }

  if (ImGui::ImageButton(char_id, image, size, ImVec2(0, 0), ImVec2(1, 1), -1, ImVec4(0, 0, 0, 0), tintColor)) {
    *toggle = !(*toggle);
    clicked = true;
  }

  return clicked;
}
ImVec2 ImGui::GetPositionBottomCorner(const char* text, uint32_t reverse_yoffset) {
  ImVec2 window_pos = ImGui::GetWindowPos();
  ImVec2 window_size = ImGui::GetWindowSize();

  ImVec2 text_size = ImGui::CalcTextSize(text);

  ImVec2 text_pos;
  text_pos.x = window_pos.x + window_size.x - text_size.x - ImGui::GetStyle().WindowPadding.x;
  text_pos.y = window_pos.y + window_size.y - text_size.y - ImGui::GetStyle().WindowPadding.y;

  text_pos.y -= reverse_yoffset * ImGui::GetTextLineHeightWithSpacing();

  return text_pos;
}
void ImGui::ImageRotated(ImTextureID user_texture_id, const ImVec2& size, int angle, const ImVec2& uv0, const ImVec2& uv1, const ImVec4& tint_col, const ImVec4& border_col)
{
  IM_ASSERT(angle % 90 == 0);
  ImVec2 _uv0, _uv1, _uv2, _uv3;
  switch (angle % 360)
  {
  case 0:
    Image(user_texture_id, size, uv0, uv1, tint_col, border_col);
    return;
  case 180:
    Image(user_texture_id, size, uv1, uv0, tint_col, border_col);
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
  ImGuiWindow* window = GetCurrentWindow();
  if (window->SkipItems)
    return;
  ImVec2 _size(size.y, size.x);
  ImRect bb(window->DC.CursorPos, window->DC.CursorPos + _size);
  if (border_col.w > 0.0f)
    bb.Max += ImVec2(2, 2);
  ItemSize(bb);
  if (!ItemAdd(bb, 0))
    return;
  if (border_col.w > 0.0f)
  {
    window->DrawList->AddRect(bb.Min, bb.Max, GetColorU32(border_col), 0.0f);
    ImVec2 x0 = bb.Min + ImVec2(1, 1);
    ImVec2 x2 = bb.Max - ImVec2(1, 1);
    ImVec2 x1 = ImVec2(x2.x, x0.y);
    ImVec2 x3 = ImVec2(x0.x, x2.y);
    window->DrawList->AddImageQuad(user_texture_id, x0, x1, x2, x3, _uv0, _uv1, _uv2, _uv3, GetColorU32(tint_col));
  }
  else
  {
    ImVec2 x1 = ImVec2(bb.Max.x, bb.Min.y);
    ImVec2 x3 = ImVec2(bb.Min.x, bb.Max.y);
    window->DrawList->AddImageQuad(user_texture_id, bb.Min, x1, bb.Max, x3, _uv0, _uv1, _uv2, _uv3, GetColorU32(tint_col));
  }
}
void ImGui::DrawTextBottomRight(const char* text, uint32_t reverse_yoffset) {
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
void fan::graphics::imgui_content_browser_t::render() {
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

      if (ImGui::ImageButton("##icon_arrow_left", icon_arrow_left, fan::vec2(32))) {
        if (std::filesystem::equivalent(current_directory, asset_path) == false) {
          current_directory = current_directory.parent_path();
        }
        update_directory_cache();
      }
      ImGui::SameLine();
      ImGui::ImageButton("##icon_arrow_right", icon_arrow_right, fan::vec2(32));
      ImGui::SameLine();
      ImGui::PopStyleColor(3);

      ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.f, 0.f, 0.f, 0.f));
      ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.f, 0.f, 0.f, 0.f));
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.3f, 0.3f));

      auto image_list = std::to_array({ icon_files_list, icon_files_big_thumbnail });

      fan::vec2 bc = ImGui::GetPositionBottomCorner();

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

      ImGui::ToggleImageButton(image_list, button_sizes, (int*)&current_view_mode);

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

fan::graphics::imgui_content_browser_t::imgui_content_browser_t() {
  search_buffer.resize(32);
  asset_path = std::filesystem::absolute(std::filesystem::path(asset_path)).wstring();
  current_directory = std::filesystem::path(asset_path);
  update_directory_cache();
}

void fan::graphics::imgui_content_browser_t::update_directory_cache() {
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

void fan::graphics::imgui_content_browser_t::render_large_thumbnails_view() {
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
    ImGui::ImageButton("##" + file_info.filename, file_info.preview_image.iic() == false ? file_info.preview_image : file_info.is_directory ? icon_directory : icon_file, ImVec2(thumbnail_size, thumbnail_size));

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

void fan::graphics::imgui_content_browser_t::render_list_view() {
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

void fan::graphics::imgui_content_browser_t::handle_item_interaction(const file_info_t& file_info) {
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

#if defined(loco_json)
bool fan::graphics::shape_to_json(loco_t::shape_t& shape, fan::json* json) {
  fan::json& out = *json;
  switch (shape.get_shape_type()) {
  case loco_t::shape_type_t::light: {
    out["shape"] = "light";
    out["position"] = shape.get_position();
    out["parallax_factor"] = shape.get_parallax_factor();
    out["size"] = shape.get_size();
    out["rotation_point"] = shape.get_rotation_point();
    out["color"] = shape.get_color();
    out["rotation_vector"] = shape.get_rotation_vector();
    out["flags"] = shape.get_flags();
    out["angle"] = shape.get_angle();
    break;
  }
  case loco_t::shape_type_t::line: {
    out["shape"] = "line";
    out["color"] = shape.get_color();
    out["src"] = shape.get_src();
    out["dst"] = shape.get_dst();
    break;
  }
  case loco_t::shape_type_t::rectangle: {
    out["shape"] = "rectangle";
    out["position"] = shape.get_position();
    out["size"] = shape.get_size();
    out["rotation_point"] = shape.get_rotation_point();
    out["color"] = shape.get_color();
    out["angle"] = shape.get_angle();
    break;
  }
  case loco_t::shape_type_t::sprite: {
    out["shape"] = "sprite";
    out["position"] = shape.get_position();
    out["parallax_factor"] = shape.get_parallax_factor();
    out["size"] = shape.get_size();
    out["rotation_point"] = shape.get_rotation_point();
    out["color"] = shape.get_color();
    out["angle"] = shape.get_angle();
    out["flags"] = shape.get_flags();
    out["tc_position"] = shape.get_tc_position();
    out["tc_size"] = shape.get_tc_size();
    break;
  }
  case loco_t::shape_type_t::unlit_sprite: {
    out["shape"] = "unlit_sprite";
    out["position"] = shape.get_position();
    out["parallax_factor"] = shape.get_parallax_factor();
    out["size"] = shape.get_size();
    out["rotation_point"] = shape.get_rotation_point();
    out["color"] = shape.get_color();
    out["angle"] = shape.get_angle();
    out["flags"] = shape.get_flags();
    out["tc_position"] = shape.get_tc_position();
    out["tc_size"] = shape.get_tc_size();
    break;
  }
  case loco_t::shape_type_t::text: {
    out["shape"] = "text";
    break;
  }
  case loco_t::shape_type_t::circle: {
    out["shape"] = "circle";
    out["position"] = shape.get_position();
    out["radius"] = shape.get_radius();
    out["rotation_point"] = shape.get_rotation_point();
    out["color"] = shape.get_color();
    out["rotation_vector"] = shape.get_rotation_vector();
    out["angle"] = shape.get_angle();
    break;
  }
  case loco_t::shape_type_t::grid: {
    out["shape"] = "grid";
    out["position"] = shape.get_position();
    out["size"] = shape.get_size();
    out["grid_size"] = shape.get_grid_size();
    out["rotation_point"] = shape.get_rotation_point();
    out["color"] = shape.get_color();
    out["angle"] = shape.get_angle();
    break;
  }
  case loco_t::shape_type_t::particles: {
    auto& ri = *(loco_t::particles_t::ri_t*)gloco->shaper.GetData(shape);
    out["shape"] = "particles";
    out["position"] = ri.position;
    out["size"] = ri.size;
    out["color"] = ri.color;
    out["begin_time"] = ri.begin_time;
    out["alive_time"] = ri.alive_time;
    out["respawn_time"] = ri.respawn_time;
    out["count"] = ri.count;
    out["position_velocity"] = ri.position_velocity;
    out["angle_velocity"] = ri.angle_velocity;
    out["begin_angle"] = ri.begin_angle;
    out["end_angle"] = ri.end_angle;
    out["angle"] = ri.angle;
    out["gap_size"] = ri.gap_size;
    out["max_spread_size"] = ri.max_spread_size;
    out["size_velocity"] = ri.size_velocity;
    out["particle_shape"] = ri.shape;
    out["blending"] = ri.blending;
    break;
  }
  default: {
    fan::throw_error("unimplemented shape");
  }
  }
  return false;
}
bool fan::graphics::json_to_shape(const fan::json& in, loco_t::shape_t* shape) {
  std::string shape_type = in["shape"];
  switch (fan::get_hash(shape_type.c_str())) {
  case fan::get_hash("rectangle"): {
    loco_t::rectangle_t::properties_t p;
    p.position = in["position"];
    p.size = in["size"];
    p.rotation_point = in["rotation_point"];
    p.color = in["color"];
    p.angle = in["angle"];
    *shape = p;
    break;
  }
    case fan::get_hash("light"): {
    loco_t::light_t::properties_t p;
    p.position = in["position"];
    p.parallax_factor = in["parallax_factor"];
    p.size = in["size"];
    p.rotation_point = in["rotation_point"];
    p.color = in["color"];
    p.rotation_vector = in["rotation_vector"];
    p.flags = in["flags"];
    p.angle = in["angle"];
    *shape = p;
    break;
  }
  case fan::get_hash("line"): {
    loco_t::line_t::properties_t p;
    p.color = in["color"];
    p.src = in["src"];
    p.dst = in["dst"];
    *shape = p;
    break;
  }
  case fan::get_hash("sprite"): {
    loco_t::sprite_t::properties_t p;
    p.blending = true;
    p.position = in["position"];
    p.parallax_factor = in["parallax_factor"];
    p.size = in["size"];
    p.rotation_point = in["rotation_point"];
    p.color = in["color"];
    p.angle = in["angle"];
    p.flags = in["flags"];
    p.tc_position = in["tc_position"];
    p.tc_size = in["tc_size"];
    *shape = p;
    break;
  }
  case fan::get_hash("unlit_sprite"): {
    loco_t::unlit_sprite_t::properties_t p;
    p.blending = true;
    p.position = in["position"];
    p.parallax_factor = in["parallax_factor"];
    p.size = in["size"];
    p.rotation_point = in["rotation_point"];
    p.color = in["color"];
    p.angle = in["angle"];
    p.flags = in["flags"];
    p.tc_position = in["tc_position"];
    p.tc_size = in["tc_size"];
    *shape = p;
    break;
  }
  case fan::get_hash("circle"): {
    loco_t::circle_t::properties_t p;
    p.position = in["position"];
    p.radius = in["radius"];
    p.rotation_point = in["rotation_point"];
    p.color = in["color"];
    p.rotation_vector = in["rotation_vector"];
    p.angle = in["angle"];
    *shape = p;
    break;
  }
  case fan::get_hash("grid"): {
    loco_t::grid_t::properties_t p;
    p.position = in["position"];
    p.size = in["size"];
    p.grid_size = in["grid_size"];
    p.rotation_point = in["rotation_point"];
    p.color = in["color"];
    p.angle = in["angle"];
    *shape = p;
    break;
  }
  case fan::get_hash("particles"): {
    loco_t::particles_t::properties_t p;
    p.position = in["position"];
    p.size = in["size"];
    p.color = in["color"];
    p.begin_time = in["begin_time"];
    p.alive_time = in["alive_time"];
    p.respawn_time = in["respawn_time"];
    p.count = in["count"];
    p.position_velocity = in["position_velocity"];
    p.angle_velocity = in["angle_velocity"];
    p.begin_angle = in["begin_angle"];
    p.end_angle = in["end_angle"];
    p.angle = in["angle"];
    p.gap_size = in["gap_size"];
    p.max_spread_size = in["max_spread_size"];
    p.size_velocity = in["size_velocity"];
    p.shape = in["particle_shape"];
    p.blending = in["blending"];
    *shape = p;
    break;
  }
  default: {
    fan::throw_error("unimplemented shape");
  }
  }
  return false;
}
bool fan::graphics::shape_serialize(loco_t::shape_t& shape, fan::json* out) {
  return shape_to_json(shape, out);
}
bool fan::graphics::shape_to_bin(loco_t::shape_t& shape, std::string* str) {
  std::string& out = *str;
  switch (shape.get_shape_type()) {
  case loco_t::shape_type_t::light: {
    // shape
    fan::write_to_string(out, std::string("light"));
    fan::write_to_string(out, shape.get_position());
    fan::write_to_string(out, shape.get_parallax_factor());
    fan::write_to_string(out, shape.get_size());
    fan::write_to_string(out, shape.get_rotation_point());
    fan::write_to_string(out, shape.get_color());
    fan::write_to_string(out, shape.get_rotation_vector());
    fan::write_to_string(out, shape.get_flags());
    fan::write_to_string(out, shape.get_angle());
    break;
  }
  case loco_t::shape_type_t::line: {
    fan::write_to_string(out, std::string("line"));
    fan::write_to_string(out, shape.get_color());
    fan::write_to_string(out, shape.get_src());
    fan::write_to_string(out, shape.get_dst());
    break;
    case loco_t::shape_type_t::rectangle: {
    fan::write_to_string(out, std::string("rectangle"));
    fan::write_to_string(out, shape.get_position());
    fan::write_to_string(out, shape.get_size());
    fan::write_to_string(out, shape.get_rotation_point());
    fan::write_to_string(out, shape.get_color());
    fan::write_to_string(out, shape.get_angle());
    break;
    }
    case loco_t::shape_type_t::sprite: {
    fan::write_to_string(out, std::string("sprite"));
    fan::write_to_string(out, shape.get_position());
    fan::write_to_string(out, shape.get_parallax_factor());
    fan::write_to_string(out, shape.get_size());
    fan::write_to_string(out, shape.get_rotation_point());
    fan::write_to_string(out, shape.get_color());
    fan::write_to_string(out, shape.get_angle());
    fan::write_to_string(out, shape.get_flags());
    fan::write_to_string(out, shape.get_tc_position());
    fan::write_to_string(out, shape.get_tc_size());
    break;
    }
    case loco_t::shape_type_t::unlit_sprite: {
    fan::write_to_string(out, std::string("unlit_sprite"));
    fan::write_to_string(out, shape.get_position());
    fan::write_to_string(out, shape.get_parallax_factor());
    fan::write_to_string(out, shape.get_size());
    fan::write_to_string(out, shape.get_rotation_point());
    fan::write_to_string(out, shape.get_color());
    fan::write_to_string(out, shape.get_angle());
    fan::write_to_string(out, shape.get_flags());
    fan::write_to_string(out, shape.get_tc_position());
    fan::write_to_string(out, shape.get_tc_size());
    break;
    }
    case loco_t::shape_type_t::circle: {
    fan::write_to_string(out, std::string("circle"));
    fan::write_to_string(out, shape.get_position());
    fan::write_to_string(out, shape.get_radius());
    fan::write_to_string(out, shape.get_rotation_point());
    fan::write_to_string(out, shape.get_color());
    fan::write_to_string(out, shape.get_rotation_vector());
    fan::write_to_string(out, shape.get_angle());
    break;
    }
    case loco_t::shape_type_t::grid: {
    fan::write_to_string(out, std::string("grid"));
    fan::write_to_string(out, shape.get_position());
    fan::write_to_string(out, shape.get_size());
    fan::write_to_string(out, shape.get_grid_size());
    fan::write_to_string(out, shape.get_rotation_point());
    fan::write_to_string(out, shape.get_color());
    fan::write_to_string(out, shape.get_angle());
    break;
    }
    case loco_t::shape_type_t::particles: {
    auto& ri = *(loco_t::particles_t::ri_t*)gloco->shaper.GetData(shape);
    fan::write_to_string(out, std::string("particles"));
    fan::write_to_string(out, ri.position);
    fan::write_to_string(out, ri.size);
    fan::write_to_string(out, ri.color);
    fan::write_to_string(out, ri.begin_time);
    fan::write_to_string(out, ri.alive_time);
    fan::write_to_string(out, ri.respawn_time);
    fan::write_to_string(out, ri.count);
    fan::write_to_string(out, ri.position_velocity);
    fan::write_to_string(out, ri.angle_velocity);
    fan::write_to_string(out, ri.begin_angle);
    fan::write_to_string(out, ri.end_angle);
    fan::write_to_string(out, ri.angle);
    fan::write_to_string(out, ri.gap_size);
    fan::write_to_string(out, ri.max_spread_size);
    fan::write_to_string(out, ri.size_velocity);
    fan::write_to_string(out, ri.shape);
    fan::write_to_string(out, ri.blending);
    break;
    }
  }
  default: {
    fan::throw_error("unimplemented shape");
  }
  }
  return false;
}
bool fan::graphics::bin_to_shape(const std::string& in, loco_t::shape_t* shape, uint64_t& offset) {
  std::string shape_type = fan::read_data<std::string>(in, offset);
  switch (fan::get_hash(shape_type.c_str())) {
  case fan::get_hash("rectangle"): {
    loco_t::rectangle_t::properties_t p;
    p.position = fan::read_data<decltype(p.position)>(in, offset);
    p.size = fan::read_data<decltype(p.size)>(in, offset);
    p.rotation_point = fan::read_data<decltype(p.rotation_point)>(in, offset);
    p.color = fan::read_data<decltype(p.color)>(in, offset);
    p.angle = fan::read_data<decltype(p.angle)>(in, offset);
    *shape = p;
    return false;
  }
  case fan::get_hash("light"): {
    loco_t::light_t::properties_t p;
    p.position = fan::read_data<decltype(p.position)>(in, offset);
    p.parallax_factor = fan::read_data<decltype(p.parallax_factor)>(in, offset);
    p.size = fan::read_data<decltype(p.size)>(in, offset);
    p.rotation_point = fan::read_data<decltype(p.rotation_point)>(in, offset);
    p.color = fan::read_data<decltype(p.color)>(in, offset);
    p.rotation_vector = fan::read_data<decltype(p.rotation_vector)>(in, offset);
    p.flags = fan::read_data<decltype(p.flags)>(in, offset);
    p.angle = fan::read_data<decltype(p.angle)>(in, offset);
    *shape = p;
    break;
  }
  case fan::get_hash("line"): {
    loco_t::line_t::properties_t p;
    p.color = fan::read_data<decltype(p.color)>(in, offset);
    p.src = fan::read_data<decltype(p.src)>(in, offset);
    p.dst = fan::read_data<decltype(p.dst)>(in, offset);
    *shape = p;
    break;
  }
  case fan::get_hash("sprite"): {
    loco_t::sprite_t::properties_t p;
    p.position = fan::read_data<decltype(p.position)>(in, offset);
    p.parallax_factor = fan::read_data<decltype(p.parallax_factor)>(in, offset);
    p.size = fan::read_data<decltype(p.size)>(in, offset);
    p.rotation_point = fan::read_data<decltype(p.rotation_point)>(in, offset);
    p.color = fan::read_data<decltype(p.color)>(in, offset);
    p.angle = fan::read_data<decltype(p.angle)>(in, offset);
    p.flags = fan::read_data<decltype(p.flags)>(in, offset);
    p.tc_position = fan::read_data<decltype(p.tc_position)>(in, offset);
    p.tc_size = fan::read_data<decltype(p.tc_size)>(in, offset);
    *shape = p;
    break;
  }
  case fan::get_hash("unlit_sprite"): {
    loco_t::unlit_sprite_t::properties_t p;
    p.position = fan::read_data<decltype(p.position)>(in, offset);
    p.parallax_factor = fan::read_data<decltype(p.parallax_factor)>(in, offset);
    p.size = fan::read_data<decltype(p.size)>(in, offset);
    p.rotation_point = fan::read_data<decltype(p.rotation_point)>(in, offset);
    p.color = fan::read_data<decltype(p.color)>(in, offset);
    p.angle = fan::read_data<decltype(p.angle)>(in, offset);
    p.flags = fan::read_data<decltype(p.flags)>(in, offset);
    p.tc_position = fan::read_data<decltype(p.tc_position)>(in, offset);
    p.tc_size = fan::read_data<decltype(p.tc_size)>(in, offset);
    *shape = p;
    break;
  }
  case loco_t::shape_type_t::circle: {
    loco_t::circle_t::properties_t p;
    p.position = fan::read_data<decltype(p.position)>(in, offset);
    p.radius = fan::read_data<decltype(p.radius)>(in, offset);
    p.rotation_point = fan::read_data<decltype(p.rotation_point)>(in, offset);
    p.color = fan::read_data<decltype(p.color)>(in, offset);
    p.rotation_vector = fan::read_data<decltype(p.rotation_vector)>(in, offset);
    p.angle = fan::read_data<decltype(p.angle)>(in, offset);
    *shape = p;
    break;
  }
  case loco_t::shape_type_t::grid: {
    loco_t::grid_t::properties_t p;
    p.position = fan::read_data<decltype(p.position)>(in, offset);
    p.size = fan::read_data<decltype(p.size)>(in, offset);
    p.grid_size = fan::read_data<decltype(p.grid_size)>(in, offset);
    p.rotation_point = fan::read_data<decltype(p.rotation_point)>(in, offset);
    p.color = fan::read_data<decltype(p.color)>(in, offset);
    p.angle = fan::read_data<decltype(p.angle)>(in, offset);
    *shape = p;
    break;
  }
  case loco_t::shape_type_t::particles: {
    loco_t::particles_t::properties_t p;
    p.position = fan::read_data<decltype(p.position)>(in, offset);
    p.size = fan::read_data<decltype(p.size)>(in, offset);
    p.color = fan::read_data<decltype(p.color)>(in, offset);
    p.begin_time = fan::read_data<decltype(p.begin_time)>(in, offset);
    p.alive_time = fan::read_data<decltype(p.alive_time)>(in, offset);
    p.respawn_time = fan::read_data<decltype(p.respawn_time)>(in, offset);
    p.count = fan::read_data<decltype(p.count)>(in, offset);
    p.position_velocity = fan::read_data<decltype(p.position_velocity)>(in, offset);
    p.angle_velocity = fan::read_data<decltype(p.angle_velocity)>(in, offset);
    p.begin_angle = fan::read_data<decltype(p.begin_angle)>(in, offset);
    p.end_angle = fan::read_data<decltype(p.end_angle)>(in, offset);
    p.angle = fan::read_data<decltype(p.angle)>(in, offset);
    p.gap_size = fan::read_data<decltype(p.gap_size)>(in, offset);
    p.max_spread_size = fan::read_data<decltype(p.max_spread_size)>(in, offset);
    p.size_velocity = fan::read_data<decltype(p.size_velocity)>(in, offset);
    p.shape = fan::read_data<decltype(p.shape)>(in, offset);
    p.blending = fan::read_data<decltype(p.blending)>(in, offset);
    *shape = p;
    break;
  }
  default: {
    fan::throw_error("unimplemented");
  }
  }
  return false;
}
bool fan::graphics::shape_serialize(loco_t::shape_t& shape, std::string* out) {
  return shape_to_bin(shape, out);
}
#endif

bool fan::graphics::texture_packe0::push_texture(fan::graphics::context_image_nr_t image, const texture_properties_t& texture_properties) {

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

  auto img = gloco->image_get(image);

  auto data = gloco->image_get_pixel_data(image, GL_RGBA, texture_properties.uv_pos, texture_properties.uv_size);
  fan::vec2ui image_size(
    (uint32_t)(img.size.x * texture_properties.uv_size.x),
    (uint32_t)(img.size.y * texture_properties.uv_size.y)
  );


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

#if defined(loco_json)
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
  #if defined(loco_imgui)
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
  #if defined(loco_imgui)
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

#if defined(loco_imgui)

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

#if defined(loco_imgui)
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

      if (ImGui::ImageTextButton(gloco->default_texture, button.text.c_str(), fan::colors::white, button.size == 0 ? button_size : button.size)) {
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

#endif

bool fan::physics::is_on_sensor(fan::physics::body_id_t test_id, fan::physics::body_id_t sensor_id){
  return gloco->physics_context.is_on_sensor(test_id, sensor_id);
}

fan::physics::ray_result_t fan::physics::raycast(const fan::vec2& src, const fan::vec2& dst) {
  return gloco->physics_context.raycast(src, dst);
}

bool fan::graphics::is_mouse_clicked(int button) {
  return gloco->is_mouse_clicked(button);
}

bool fan::graphics::is_mouse_down(int button) {
  return gloco->is_mouse_down(button);
}

bool fan::graphics::is_mouse_released(int button) {
  return gloco->is_mouse_released(button);
}

fan::vec2 fan::graphics::get_mouse_drag(int button) {
  return gloco->get_mouse_drag(button);
}

void fan::graphics::set_window_size(const fan::vec2& size) {
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