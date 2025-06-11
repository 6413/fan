
struct settings_menu_t;

typedef void(*page_function_t)(settings_menu_t*, const fan::vec2& next_window_position, const fan::vec2& next_window_size);

// functions are defined in graphics.cpp

struct settings_menu_t {
  // pages are divided into two vertically

  static constexpr int wnd_flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
    ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar;

  static void menu_graphics_left(settings_menu_t* menu, const fan::vec2& next_window_position, const fan::vec2& next_window_size) {

    ImGui::SetNextWindowPos(next_window_position);
    ImGui::SetNextWindowSize(next_window_size);
    ImGui::SetNextWindowBgAlpha(0.99);
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
        ImGui::Text("Enable VSync");
        ImGui::TableNextColumn();
        if (ImGui::Checkbox("##enable_vsync", (bool*)&gloco->vsync)) {
          gloco->set_vsync(gloco->vsync);
        }
      }
      {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Show fps");
        ImGui::TableNextColumn();
        ImGui::Checkbox("##show_fps", (bool*)&gloco->show_fps);
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
  static void menu_graphics_right(settings_menu_t* menu, const fan::vec2& next_window_position, const fan::vec2& next_window_size) {

    ImGui::SetNextWindowPos(next_window_position);
    ImGui::SetNextWindowSize(next_window_size);
    ImGui::SetNextWindowBgAlpha(0.99);
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
  static void menu_audio_left(settings_menu_t* menu, const fan::vec2& next_window_position, const fan::vec2& next_window_size) {

  }
  static void menu_audio_right(settings_menu_t* menu, const fan::vec2& next_window_position, const fan::vec2& next_window_size) {
    loco_t::settings_menu_t::menu_graphics_right(menu, next_window_position, next_window_size);
  }
  void open() {
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
  void render_resolution_dropdown() {
    fan::vec2i current_size = gloco->window.get_size();

    int current_resolution = -1;
    for (int i = 0; i < std::size(fan::window_t::resolutions); ++i) {
      if (fan::window_t::resolutions[i] == fan::vec2i(current_size)) {
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

  void render_separator_with_margin(f32_t width, f32_t margin = 0.f) {
    ImVec2 separator_start = ImGui::GetCursorScreenPos();
    ImVec2 separator_end = ImVec2(separator_start.x + width - margin * 2, separator_start.y);
    separator_start.x += margin;

    ImGui::GetWindowDrawList()->AddLine(separator_start, separator_end, ImGui::GetColorU32(ImGuiCol_Separator), 1.0f);
  }
  void render_settings_left(const fan::vec2& next_window_position, const fan::vec2& next_window_size) {
    pages[current_page].page_left_render(this, next_window_position, next_window_size);
  }
  void render_settings_right(const fan::vec2& next_window_position, const fan::vec2& next_window_size, f32_t min_x) {
    pages[current_page].page_right_render(this, next_window_position, next_window_size);
  }
  fan::vec2 render_settings_top(f32_t min_x) {
    fan::vec2 main_window_size = gloco->window.get_size();
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(fan::vec2(main_window_size.x, main_window_size.y / 5));
    ImGui::SetNextWindowBgAlpha(0.99);
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
  void render() {
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
  void reset_page_selection() {
    for (auto& page : pages) {
      page.toggle = 0;
    }
  }

  static void set_settings_theme() {
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

  static constexpr const int fps_values[] = { 0, 30, 60, 144, 165, 240 };
  struct page_t {
    bool toggle = false;
    std::string name;
    page_function_t page_left_render;
    page_function_t page_right_render;
  };

  // start from page 0
  int current_page = 0;
  int current_resolution = 0;
  f32_t bloom_strength = 0;

  f32_t min_x = 40.f; // page
  std::deque<page_t> pages;
};