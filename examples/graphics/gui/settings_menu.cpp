#include <fan/pch.h>

void modify_reflect_depth(std::vector<fan::graphics::rectangle_t>& ray_hit_point, std::vector<fan::graphics::line_t>& rays, int& prev_reflect_depth);


int reflect_depth = 4;

static const int fps_values[] = { 0, 30, 60, 144, 165, 240 };

void set_settings_theme();

struct settings_menu_t {
  fan_enum_string(
    settings_options, 
    GRAPHICS, 
    AUDIO
  );

  settings_menu_t() {
    options_toggle[0] = 1;
  }

  void change_target_fps(int direction) {
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
  void render_display_mode() {
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
  void render_target_fps() {
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
  void render_options_display() {
    ImGui::TableNextRow();
    render_display_mode();
    ImGui::TableNextRow();
    render_target_fps();
  }
  void render_settings_left_column() {
    ImGui::SetColumnWidth(0, ImGui::GetWindowWidth() * 0.5f);

    switch (current_option) {
    case GRAPHICS: {
      ImGui::TextColored(fan::color::hex(0x948c80ff) * 1.5, "DISPLAY");

      ImVec2 separator_start = ImGui::GetCursorScreenPos();
      separator_start.x = ImGui::GetCursorScreenPos().x;
      separator_start.y = ImGui::GetCursorScreenPos().y + ImGui::GetFontSize() + ImGui::GetStyle().ItemSpacing.y;
      ImVec2 separator_end = ImVec2(separator_start.x + ImGui::GetWindowWidth() * 0.5f, separator_start.y);
      separator_start.y -= 25.f;
      separator_end.y -= 25.f;
      ImGui::GetWindowDrawList()->AddLine(separator_start, separator_end, ImGui::GetColorU32(ImGuiCol_Separator), 1.0f);

      ImGui::Dummy(ImVec2(0, 4.0f));

      ImGui::BeginTable("settings_left_table", 2,
        ImGuiTableFlags_BordersInnerH
      );
      render_options_display();
      ImGui::EndTable();
      break;
    }
    }
   
  }
  void render_settings_right_column(f32_t min_x) {
    ImGui::NextColumn();
    ImGui::PushFont(gloco->fonts_bold[std::size(gloco->fonts_bold) - 3]);
    ImGui::Indent(min_x);
    ImGui::TextColored(fan::color::hex(0x948c80ff) * 1.5, "Setting Info");
    ImGui::Unindent(min_x);
    ImGui::PopFont();

    ImVec2 cursor_pos = ImGui::GetCursorPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 line_start = ImGui::GetCursorScreenPos();
    line_start.x -= cursor_pos.x;
    line_start.y -= cursor_pos.y;

    ImVec2 line_end = line_start;
    line_end.y += ImGui::GetContentRegionMax().y;

    draw_list->AddLine(line_start, line_end, IM_COL32(255, 255, 255, 255));
  }
  void render_settings_top(f32_t min_x) {
    ImGui::PushFont(gloco->fonts_bold[std::size(gloco->fonts_bold) - 2]);
    ImGui::Indent(min_x);
    ImGui::Text("Settings");
    ImGui::PopFont();

    ImGui::Separator();
    f32_t options_x = 356.f;
    ImGui::Indent(options_x);
    ImGui::PushFont(gloco->fonts_bold[2]);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(64, 5.f));
    ImGui::BeginTable("##settings_top_table", 2);
    ImGui::TableNextRow();
    for (std::size_t i = 0; i < std::size(settings_options_strings); ++i) {
      ImGui::TableNextColumn();
      bool is_toggled = options_toggle[i];
      if (is_toggled) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonHovered));
      }
      else {
        ImGui::PushStyleColor(ImGuiCol_Button, fan::colors::transparent);
      }
      if (ImGui::Button(settings_options_strings[i])) {
        options_toggle[i] = !options_toggle[i];
        if (options_toggle[i]) {
          options_toggle.fill(0);
          options_toggle[i] = 1;
          current_option = i;
        }
      }
      ImGui::PopStyleColor();
    }
    
    ImGui::EndTable();
    ImGui::PopStyleVar();

    ImGui::PopFont();
    ImGui::Unindent(options_x);
    ImGui::Separator();
  }
  void render() {
    f32_t min_x = 40.f;

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.09411764889955521f, 0.09411764889955521f, 0.09411764889955521f, 0.9f));
    ImGui::PushStyleColor(ImGuiCol_Separator, ImVec4(0.8, 0.8, 0.8, 1.0f));
    fan::graphics::gui::render_blank_window("Fan Settings Menu");

    render_settings_top(min_x);

    ImGui::NewLine();
    ImGui::Columns(2);

    // 50% left
    render_settings_left_column();

    // %50 right
    render_settings_right_column(min_x);
    ImGui::Columns(1);

    ImGui::Unindent(min_x);
    ImGui::End();
    ImGui::PopStyleColor(2);
  }
  int current_option = settings_options::GRAPHICS;
  std::array<bool, std::size(settings_options_strings)> options_toggle{0};
};

fan_language{
  set_settings_theme();
  std::vector<vertex_t> triangle_vertices{
    {vec2(400, 400), colors::orange},
    {vec2(400, 600), colors::orange},
    {vec2(700, 600), colors::orange},
  };
  physics_shapes::polygon_t triangle{{
    .vertices = triangle_vertices
  }};
  std::vector<physics_shapes::circle_t> circles;
  for (std::size_t i = 0; i < 5; ++i) {
    circles.push_back({ {
      .position = fan::random::vec2(0, gloco->window.get_size()),
      .radius = fan::random::f32(12, 84),
      .color = colors::orange,
    }});
  }

  auto box = physics_shapes::create_stroked_rectangle(gloco->window.get_size() / 2, gloco->window.get_size() / 2, 3);

  line_t ray{ { .src={0, 500, 0xfff}, .color = fan::colors::white}};

  std::vector<rectangle_t> ray_hit_point(reflect_depth, {{.size=4, .color=fan::colors::red}});
  std::vector<line_t> rays(reflect_depth, { { .src={0, 0, 0xfff}, .color = fan::colors::green}});

  int prev_reflect_depth = reflect_depth;
  settings_menu_t settings_menu;
  fan_window_loop{
    settings_menu.render();

    modify_reflect_depth(ray_hit_point, rays, prev_reflect_depth);

    fan::vec2 src = ray.get_src();
    fan::vec2 dst = ray.get_dst();
    ray.set_line(src, dst);
    if (is_mouse_down(fan::mouse_right)) {
      ray.set_line(get_mouse_position(), dst);
    }
    if (is_mouse_down()) {
      ray.set_line(src, get_mouse_position());
    }
    for (auto [i, d] : fan::enumerate(ray_hit_point)) {
      d.set_position(-1000);
      rays[i].set_line(0, 0);
    }
    
    int depth = 0;
    fan::vec2 current_src = src;
    fan::vec2 current_dst = dst;

    while (depth < reflect_depth) {
      if (auto result = physics::raycast(current_src, current_dst)) {
        ray_hit_point[depth].set_position(result.point);

        fan::vec2 direction = (current_dst - current_src).normalize();
        fan::vec2 reflection = direction - result.normal * 2 * direction.dot(result.normal);
        rays[depth].set_line(current_src, result.point);
        rays[depth].set_color(color::hsv(360.f * (depth / (f32_t)reflect_depth), 100, 100));

        current_src = result.point + reflection * 0.5f;
        current_dst = result.point + reflection * 10000.f;

        depth++;
      }
      else {
        break;
      }
    }

    text("reflect depth: " + std::to_string(reflect_depth), {5, 25});
  };
};

void modify_reflect_depth(std::vector<fan::graphics::rectangle_t>& ray_hit_point, std::vector<fan::graphics::line_t>& rays, int& prev_reflect_depth) {
  if (gloco->window.key_state(fan::mouse_scroll_up) != -1) {
    reflect_depth += 1;
  }
  else if (gloco->window.key_state(fan::mouse_scroll_down) != -1) {
    reflect_depth = std::max(reflect_depth - 1, 0);
  }
  if (reflect_depth != prev_reflect_depth) {
    ray_hit_point.resize(reflect_depth, {{.size=4, .color=fan::colors::red}});
    rays.resize(reflect_depth, { { .src={0, 0, 0xfff}, .color = fan::colors::green}});
    prev_reflect_depth = reflect_depth;
  }
}

void set_settings_theme() {
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
  style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
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
