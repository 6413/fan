struct settings_menu_t;

typedef void(*page_function_t)(settings_menu_t*, const fan::vec2& next_window_position, const fan::vec2& next_window_size);

// functions are defined in graphics.cpp

#define gui fan::graphics::gui

struct settings_menu_t {
  // pages are divided into two vertically

  inline static bool hide_bg = false;

  static void begin_menu_left(
    const char* name, 
    const fan::vec2& next_window_position,
    const fan::vec2& next_window_size) 
  {
    gui::push_font(gui::get_font(24));
    gui::set_next_window_pos(next_window_position);
    gui::set_next_window_size(next_window_size);
    gui::set_next_window_bg_alpha(hide_bg ? 0 : 0.99);
    gui::begin(name, nullptr, wnd_flags);
  }
  static void end_menu_left() {
    gui::end();
    gui::pop_font();
  }

  static constexpr int wnd_flags = gui::window_flags_no_move | gui::window_flags_no_collapse |
    gui::window_flags_no_resize | gui::window_flags_no_title_bar;

  static constexpr fan::color title_color = fan::color::from_rgba(0x948c80ff) * 1.5f;

  static void menu_graphics_left(
    settings_menu_t* menu, 
    const fan::vec2& next_window_position,
    const fan::vec2& next_window_size) 
  {
    begin_menu_left("##Menu Graphics Left", next_window_position, next_window_size);
    {
      gui::text(title_color, "DISPLAY");
      gui::begin_table("settings_left_table_display", 2,
        gui::table_flags_borders_inner_h |
        gui::table_flags_borders_outer_h
      );
      {
        gui::table_next_row();
        menu->render_display_mode();
        gui::table_next_row();
        menu->render_target_fps();
        gui::table_next_row();
        menu->render_resolution_dropdown();

        {
          static const char* renderers[] = {
            "OpenGL",
          #if defined(fan_vulkan)
            "Vulkan",
          #endif
          };
          gui::table_next_column();
          gui::text("Renderer");
          gui::table_next_column();
          if (gui::begin_combo("##Renderer", renderers[gloco()->window.renderer])) {
            for (int i = 0; i < std::size(renderers); ++i) {
              bool is_selected = (gloco()->window.renderer == i);
              if (gui::selectable(renderers[i], is_selected)) {
                switch (i) {
                case 0:
                {
                  if (gloco()->window.renderer != fan::window_t::renderer_t::opengl) {
                    gloco()->reload_renderer_to = fan::window_t::renderer_t::opengl;
                  }
                  break;
                }
                case 1:
                {
                  if (gloco()->window.renderer != fan::window_t::renderer_t::vulkan) {
                    gloco()->reload_renderer_to = fan::window_t::renderer_t::vulkan;
                  }
                  break;
                }
                }
              }
              if (is_selected) {
                gui::set_item_default_focus();
              }
            }
            gui::end_combo();
          }
        }
      }

      gui::end_table();
    }
    gui::new_line();
    gui::new_line();
    #if defined(loco_framebuffer)
    {
      gui::text(title_color, "POST PROCESSING");
      gui::begin_table("settings_left_table_post_processing", 2,
        gui::table_flags_borders_inner_h |
        gui::table_flags_borders_outer_h
      );

      {
        gui::table_next_row();
        gui::table_next_column();
        gui::text("Bloom Strength");
        gui::table_next_column();
        if (gui::slider("##BloomStrengthSlider", &menu->bloom_strength, 0, 1)) {
          if (gloco()->window.renderer == fan::window_t::renderer_t::opengl) {
            gloco()->shader_set_value(gloco()->gl.m_fbo_final_shader, "bloom_strength", menu->bloom_strength);
          }
        }
      }

      gui::end_table();
    }
    #endif
    gui::new_line();
    gui::new_line();
    {
      gui::text(title_color, "PERFORMANCE STATS");
      gui::begin_table("settings_left_table_post_processing", 2,
        gui::table_flags_borders_inner_h |
        gui::table_flags_borders_outer_h
      );
      {
        gui::table_next_row();
        gui::table_next_column();
        gui::text("Enable VSync");
        gui::table_next_column();
        if (gui::checkbox("##enable_vsync", (bool*)&gloco()->vsync)) {
          gloco()->set_vsync(gloco()->vsync);
        }
      }
      {
        gui::table_next_row();
        gui::table_next_column();
        gui::text("Show fps");
        gui::table_next_column();
        gui::checkbox("##show_fps", (bool*)&gloco()->show_fps);
      }
    #if defined(fan_std23)
      {
        gui::table_next_row();
        gui::table_next_column();
        gui::text("Track Heap memory");
        gui::table_next_column();
        if (gui::checkbox("##track_heap", (bool*)&fan::heap_profiler_t::instance().enabled)) {
          gloco()->console.commands.call("debug_memory " + std::to_string((int)fan::heap_profiler_t::instance().enabled));
        }
      }
    #endif
      {
        gui::table_next_row();
        gui::table_next_column();
        gui::text("Track OpenGL calls");
        gui::table_next_column();
        gui::checkbox("##track_opengl_calls", (bool*)&fan_track_opengl_calls());
      }

      gui::end_table();
    }

    gui::new_line();
    gui::new_line();
    {
      gui::text(title_color, "DEBUG");
      gui::begin_table("settings_left_table_debug", 2,
        gui::table_flags_borders_inner_h |
        gui::table_flags_borders_outer_h
      );

      static bool hide_gui_settings = false;

      bool did_hide_bg = hide_gui_settings;
      {
        gui::table_next_row();

        gui::table_next_column();
        gui::text("Frustum culling");

        gui::table_next_column();
        gui::text("Enable frustum culling");
        if (gui::checkbox("##enable_culling", &gloco()->shapes.visibility.enabled)) {
          gloco()->set_culling_enabled(gloco()->shapes.visibility.enabled);
        }

        gui::text("Visualize culling");
        gui::checkbox("##visualize_culling", &gloco()->is_visualizing_culling);

        if (gloco()->is_visualizing_culling) {

          gui::text("Frustum culling extents padding (default render view)");
          gui::indent(10.f);
          //gui::table_next_column();
          if (gui::drag("##culling_bounds", &gloco()->shapes.visibility.padding, 1)) {
            for (auto& [cam_id, cam_state] : gloco()->shapes.visibility.camera_states) {
              cam_state.view_dirty = true;
            }
          }

          if (!hide_gui_settings) {
            hide_bg = gui::is_item_active();
            if (hide_bg != did_hide_bg) {
              did_hide_bg = true;
            }
          }

          gui::unindent();
        }
      }

      {
        gui::table_next_row();

        gui::table_next_column();
        gui::text("Hide settings background");

        gui::table_next_column();
        gui::checkbox("##hide_settings_bg", &hide_gui_settings);
        if (!did_hide_bg) {
          hide_bg = hide_gui_settings;
        }
      }

      {
        static const char* fill_modes[] = {
          "Fill",
          "Line"
        };
        gui::table_next_row();

        gui::table_next_column();
        gui::text("Fill mode");
        gui::table_next_column();
        static int fill_mode = 0;

        if (gui::begin_combo("##Fill_mode", fill_modes[fill_mode])) {
          for (int i = 0; i < std::size(fill_modes); ++i) {
            bool is_selected = (fill_mode == i);
            if (gui::selectable(fill_modes[i], is_selected)) {
              fill_mode = i;
              switch (i) {
              case 0:
              {
                gloco()->force_line_draw = false;
                break;
              }
              case 1:
              {
                gloco()->force_line_draw = true;
                break;
              }
              }
            }
            if (is_selected) {
              gui::set_item_default_focus();
            }
          }
          gui::end_combo();
        }
      }

      gui::end_table();
    }
    end_menu_left();
  }

  static void menu_graphics_right(settings_menu_t* menu, const fan::vec2& next_window_position, const fan::vec2& next_window_size) {

    gui::set_next_window_pos(next_window_position);
    gui::set_next_window_size(next_window_size);
    gui::set_next_window_bg_alpha(hide_bg ? 0 : 0.99);
    gui::begin("##Menu Graphics Right", nullptr, wnd_flags);

    gui::push_font(gui::get_font(32, true));
    gui::indent(menu->min_x);
    gui::text(title_color, "Setting Info");
    gui::unindent(menu->min_x);
    gui::pop_font();

    fan::vec2 cursor_pos = gui::get_cursor_pos();
    gui::draw_list_t* draw_list = gui::get_window_draw_list();
    fan::vec2 line_start = gui::get_cursor_screen_pos();
    line_start.x -= cursor_pos.x;
    line_start.y -= cursor_pos.y;

    fan::vec2 line_end = line_start;
    line_end.y += gui::get_content_region_max().y;

    draw_list->AddLine(line_start, line_end, fan::color(255, 255, 255, 255).get_gui_color());

    gui::end();
  }

  static void menu_audio_left(settings_menu_t* menu, const fan::vec2& next_window_position, const fan::vec2& next_window_size) {
  #if defined(fan_audio)
    begin_menu_left("##Menu Audio Left", next_window_position, next_window_size);
    {
      gui::begin_table("settings_left_table_display", 2,
        gui::table_flags_borders_inner_h |
        gui::table_flags_borders_outer_h
      );
      {
        gui::table_next_row();
        gui::table_next_column();
        gui::text("Volume");
        gui::table_next_column();
        f32_t volume = fan::audio::get_volume();
        if (gui::slider("##slider_volume", &volume, 0.f, 1.f, gui::slider_flags_always_clamp)) {
          fan::audio::set_volume(volume);
        }
      }
      gui::end_table();
    }
    end_menu_left();
  #endif
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
  #if defined(loco_framebuffer)
    if (gloco()->window.renderer == fan::window_t::renderer_t::opengl) {
      gloco()->shader_set_value(gloco()->gl.m_fbo_final_shader, "bloom_strength", bloom_strength);
    }
  #endif
  }

  void change_target_fps(int direction) {
    int index = 0;
    for (int i = 0; i < std::size(fps_values); ++i) {
      if (fps_values[i] == gloco()->target_fps) {
        index = i;
        break;
      }
    }
    index = (index + direction + std::size(fps_values)) % std::size(fps_values);
    gloco()->set_target_fps(fps_values[index]);
  }
  void render_display_mode() {
    static const char* display_mode_names[] = {
      "Windowed",
      "Borderless",
      "Windowed Fullscreen",
      "Fullscreen",
    };
    gui::table_next_column();
    gui::text("Display Mode");
    gui::table_next_column();
    if (gui::begin_combo("##Display Mode", display_mode_names[gloco()->window.display_mode - 1])) {
      for (int i = 0; i < std::size(display_mode_names); ++i) {
        bool is_selected = (gloco()->window.display_mode - 1 == i);
        if (gui::selectable(display_mode_names[i], is_selected)) {
          gloco()->window.set_display_mode(i + 1);
        }
        if (is_selected) {
          gui::set_item_default_focus();
        }
      }
      gui::end_combo();
    }
  }

  void render_target_fps() {
    gui::table_next_column();
    gui::text("Target Framerate");
    gui::table_next_column();
    gui::same_line();
    if (gui::arrow_button("##left_arrow", gui::dir_left)) {
      change_target_fps(-1);
    }
    gui::same_line();
    gui::text(gloco()->target_fps);
    gui::same_line();
    if (gui::arrow_button("##right_arrow", gui::dir_right)) {
      change_target_fps(1);
    }
  }

  void render_resolution_dropdown() {
    fan::vec2i current_size = gloco()->window.get_size();

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

    gui::table_next_column();
    gui::text("Resolution");
    gui::table_next_column();

    fan::vec2i window_size = gloco()->window.get_size();
    std::string custom_res = std::to_string(window_size.x) + "x" + std::to_string(window_size.y);
    const char* current_label = (current_resolution == std::size(fan::window_t::resolutions)) ?
      custom_res.c_str() : fan::window_t::resolution_labels[current_resolution];

    if (gui::begin_combo("##ResolutionCombo", current_label)) {
      for (int i = 0; i < std::size(fan::window_t::resolution_labels); ++i) {
        bool is_selected = (current_resolution == i);
        if (gui::selectable(fan::window_t::resolution_labels[i], is_selected)) {
          current_resolution = i;
          gloco()->window.set_size(fan::window_t::resolutions[i]);
        }
        if (is_selected) {
          gui::set_item_default_focus();
        }
      }
      if (current_resolution == -1 && gui::selectable(custom_res.c_str(), current_resolution == std::size(fan::window_t::resolutions))) {
        current_resolution = std::size(fan::window_t::resolutions);
      }
      gui::end_combo();
    }
  }

  void render_separator_with_margin(f32_t width, f32_t margin = 0.f) {
    fan::vec2 separator_start = gui::get_cursor_screen_pos();
    fan::vec2 separator_end = fan::vec2(separator_start.x + width - margin * 2, separator_start.y);
    separator_start.x += margin;

    gui::get_window_draw_list()->AddLine(separator_start, separator_end, gui::get_color_u32(gui::col_separator), 1.0f);
  }

  void render_settings_left(const fan::vec2& next_window_position, const fan::vec2& next_window_size) {
    pages[current_page].page_left_render(this, next_window_position, next_window_size);
  }

  void render_settings_right(const fan::vec2& next_window_position, const fan::vec2& next_window_size, f32_t min_x) {
    pages[current_page].page_right_render(this, next_window_position, next_window_size);
  }

  fan::vec2 render_settings_top(f32_t min_x) {
    fan::vec2 main_window_size = gloco()->window.get_size();
    gui::set_next_window_pos(fan::vec2(0, 0));
    gui::set_next_window_size(fan::vec2(main_window_size.x, main_window_size.y / 5));
    gui::set_next_window_bg_alpha(0.99);
    gui::begin("##Fan Settings Nav", nullptr,
      gui::window_flags_no_move | gui::window_flags_no_collapse |
      gui::window_flags_no_resize | gui::window_flags_no_title_bar
    );
    gui::push_font(gui::get_font(48, true));
    gui::indent(min_x);
    gui::text("Settings");
    gui::pop_font();

    render_separator_with_margin(gui::get_content_region_avail().x - min_x);
    f32_t options_x = 256.f;
    gui::indent(options_x);
    gui::push_font(gui::get_font(32, true));
    gui::push_style_var(gui::style_var_frame_padding, fan::vec2(64, 5.f));
    gui::begin_table("##settings_top_table", pages.size());
    gui::table_next_row();
    for (std::size_t i = 0; i < std::size(pages); ++i) {
      gui::table_next_column();
      bool& is_toggled = pages[i].toggle;
      if (is_toggled) {
        gui::push_style_color(gui::col_button, gui::get_color(gui::col_button_hovered));
      }
      else {
        gui::push_style_color(gui::col_button, fan::colors::transparent);
      }
      if (gui::button(pages[i].name.c_str())) {
        pages[i].toggle = !pages[i].toggle;
        if (pages[i].toggle) {
          reset_page_selection();
          pages[i].toggle = 1;
          current_page = i;
        }
      }
      gui::pop_style_color();
    }

    gui::end_table();
    gui::pop_style_var();

    gui::pop_font();
    gui::unindent(options_x);
    render_separator_with_margin(gui::get_content_region_avail().x - min_x);
    fan::vec2 window_size = gui::get_window_size();
    gui::unindent();
    gui::end();
    return window_size;
  }

  void render() {
    if (gloco()->reload_renderer_to != (decltype(gloco()->reload_renderer_to))-1) {
      set_settings_theme();
    }

    set_settings_theme();

    //gui::push_style_color(gui::col_window_bg, fan::color(0.09411764889955521f, 0.09411764889955521f, 0.09411764889955521f, 0.99f));
    gui::push_style_color(gui::col_window_bg, fan::color(0.01f, 0.01f, 0.01f, 0.99f));
    gui::push_style_color(gui::col_separator, fan::color(0.8, 0.8, 0.8, 1.0f));

    fan::vec2 main_window_size = gloco()->window.get_size();
    fan::vec2 window_size = render_settings_top(min_x);

    fan::vec2 next_window_position = fan::vec2(0, window_size.y);
    fan::vec2 next_window_size = fan::vec2(main_window_size.x / 2.f, main_window_size.y - next_window_position.y);
    render_settings_left(next_window_position, next_window_size);

    next_window_position = next_window_position + fan::vec2(main_window_size.x / 2.f, 0);
    next_window_size = fan::vec2(main_window_size.x / 2.f, main_window_size.y - next_window_position.y);
    render_settings_right(next_window_position, next_window_size, min_x);

    gui::pop_style_color(2);
  }

  void reset_page_selection() {
    for (auto& page : pages) {
      page.toggle = 0;
    }
  }

  static void set_settings_theme() {
    auto& style = gui::get_style();

    style.Alpha = 1.0f;
    style.DisabledAlpha = 0.5f;
    style.WindowPadding = fan::vec2(13.0f, 10.0f);
    style.WindowRounding = 0.0f;
    style.WindowBorderSize = 1.0f;
    style.WindowMinSize = fan::vec2(32.0f, 32.0f);
    style.WindowTitleAlign = fan::vec2(0.5f, 0.5f);
    style.WindowMenuButtonPosition = gui::dir_right;
    style.ChildRounding = 3.0f;
    style.ChildBorderSize = 1.0f;
    style.PopupRounding = 5.0f;
    style.PopupBorderSize = 1.0f;
    style.FramePadding = fan::vec2(20.0f, 8.100000381469727f);
    style.FrameRounding = 2.0f;
    style.FrameBorderSize = 0.0f;
    style.ItemSpacing = fan::vec2(3.0f, 3.0f);
    style.ItemInnerSpacing = fan::vec2(3.0f, 8.0f);
    style.CellPadding = fan::vec2(6.0f, 14.10000038146973f);
    style.IndentSpacing = 0.0f;
    style.ColumnsMinSpacing = 10.0f;
    style.ScrollbarSize = 10.0f;
    style.ScrollbarRounding = 2.0f;
    style.GrabMinSize = 12.10000038146973f;
    style.GrabRounding = 1.0f;
    style.TabRounding = 2.0f;
    style.TabBorderSize = 0.0f;
    style.TabMinWidthForCloseButton = 5.0f;
    style.ColorButtonPosition = gui::dir_right;
    style.ButtonTextAlign = fan::vec2(0.5f, 0.5f);
    style.SelectableTextAlign = fan::vec2(0.0f, 0.0f);

    style.Colors[gui::col_text] = fan::color(0.9803921580314636f, 0.9803921580314636f, 0.9803921580314636f, 1.0f);
    style.Colors[gui::col_text_disabled] = fan::color(0.4980392158031464f, 0.4980392158031464f, 0.4980392158031464f, 1.0f);
    style.Colors[gui::col_window_bg] = fan::color(0.09411764889955521f, 0.09411764889955521f, 0.09411764889955521f, 0.99);
    style.Colors[gui::col_child_bg] = fan::color(0.1568627506494522f, 0.1568627506494522f, 0.1568627506494522f, 1.0f);
    style.Colors[gui::col_popup_bg] = fan::color(0.09411764889955521f, 0.09411764889955521f, 0.09411764889955521f, 1.0f);
    style.Colors[gui::col_border] = fan::color(1.0f, 1.0f, 1.0f, 0.09803921729326248f);
    style.Colors[gui::col_border_shadow] = fan::color(0.0f, 0.0f, 0.0f, 0.0f);
    style.Colors[gui::col_frame_bg] = fan::color(1.0f, 1.0f, 1.0f, 0.09803921729326248f);
    style.Colors[gui::col_frame_bg_hovered] = fan::color(1.0f, 1.0f, 1.0f, 0.1568627506494522f);
    style.Colors[gui::col_frame_bg_active] = fan::color(0.0f, 0.0f, 0.0f, 0.0470588244497776f);
    style.Colors[gui::col_title_bg] = fan::color(0.1176470592617989f, 0.1176470592617989f, 0.1176470592617989f, 1.0f);
    style.Colors[gui::col_title_bg_active] = fan::color(0.1568627506494522f, 0.1568627506494522f, 0.1568627506494522f, 1.0f);
    style.Colors[gui::col_title_bg_collapsed] = fan::color(0.1176470592617989f, 0.1176470592617989f, 0.1176470592617989f, 1.0f);
    style.Colors[gui::col_menu_bar_bg] = fan::color(0.09411764889955521f, 0.09411764889955521f, 0.09411764889955521f, 1.f);
    style.Colors[gui::col_scrollbar_bg] = fan::color(0.0f, 0.0f, 0.0f, 0.1098039224743843f);
    style.Colors[gui::col_scrollbar_grab] = fan::color(1.0f, 1.0f, 1.0f, 0.3921568691730499f);
    style.Colors[gui::col_scrollbar_grab_hovered] = fan::color(1.0f, 1.0f, 1.0f, 0.4705882370471954f);
    style.Colors[gui::col_scrollbar_grab_active] = fan::color(0.0f, 0.0f, 0.0f, 0.09803921729326248f);
    style.Colors[gui::col_check_mark] = fan::color(1.0f, 1.0f, 1.0f, 1.0f);
    style.Colors[gui::col_slider_grab] = fan::color(1.0f, 1.0f, 1.0f, 0.3921568691730499f);
    style.Colors[gui::col_slider_grab_active] = fan::color(1.0f, 1.0f, 1.0f, 0.3137255012989044f);
    style.Colors[gui::col_button] = fan::color(1.0f, 1.0f, 1.0f, 0.09803921729326248f);
    style.Colors[gui::col_button_hovered] = fan::color(1.0f, 1.0f, 1.0f, 0.1568627506494522f);
    style.Colors[gui::col_button_active] = fan::color(0.0f, 0.0f, 0.0f, 0.0470588244497776f);
    style.Colors[gui::col_header] = fan::color(1.0f, 1.0f, 1.0f, 0.09803921729326248f);
    style.Colors[gui::col_header_hovered] = fan::color(1.0f, 1.0f, 1.0f, 0.1568627506494522f);
    style.Colors[gui::col_header_active] = fan::color(0.0f, 0.0f, 0.0f, 0.0470588244497776f);
    style.Colors[gui::col_separator] = fan::color(1.0f, 1.0f, 1.0f, 0.1568627506494522f);
    style.Colors[gui::col_separator_hovered] = fan::color(1.0f, 1.0f, 1.0f, 0.2352941185235977f);
    style.Colors[gui::col_separator_active] = fan::color(1.0f, 1.0f, 1.0f, 0.2352941185235977f);
    style.Colors[gui::col_resize_grip] = fan::color(1.0f, 1.0f, 1.0f, 0.1568627506494522f);
    style.Colors[gui::col_resize_grip_hovered] = fan::color(1.0f, 1.0f, 1.0f, 0.2352941185235977f);
    style.Colors[gui::col_resize_grip_active] = fan::color(1.0f, 1.0f, 1.0f, 0.2352941185235977f);
    style.Colors[gui::col_tab] = fan::color(1.0f, 1.0f, 1.0f, 0.09803921729326248f);
    style.Colors[gui::col_tab_hovered] = fan::color(1.0f, 1.0f, 1.0f, 0.1568627506494522f);
    style.Colors[gui::col_tab_selected] = fan::color(1.0f, 1.0f, 1.0f, 0.3137255012989044f);
    style.Colors[gui::col_tab_dimmed] = fan::color(0.0f, 0.0f, 0.0f, 0.1568627506494522f);
    style.Colors[gui::col_tab_dimmed_selected] = fan::color(1.0f, 1.0f, 1.0f, 0.2352941185235977f);
    style.Colors[gui::col_plot_lines] = fan::color(1.0f, 1.0f, 1.0f, 0.3529411852359772f);
    style.Colors[gui::col_plot_lines_hovered] = fan::color(1.0f, 1.0f, 1.0f, 1.0f);
    style.Colors[gui::col_plot_histogram] = fan::color(1.0f, 1.0f, 1.0f, 0.3529411852359772f);
    style.Colors[gui::col_plot_histogram_hovered] = fan::color(1.0f, 1.0f, 1.0f, 1.0f);
    style.Colors[gui::col_table_header_bg] = fan::color(0.1568627506494522f, 0.1568627506494522f, 0.1568627506494522f, 1.0f);
    style.Colors[gui::col_table_border_strong] = fan::color(1.0f, 1.0f, 1.0f, 0.3137255012989044f);
    style.Colors[gui::col_table_border_light] = fan::color(1.0f, 1.0f, 1.0f, 0.196078434586525f);
    style.Colors[gui::col_table_row_bg] = fan::color(0.0f, 0.0f, 0.0f, 0.0f);
    style.Colors[gui::col_table_row_bg_alt] = fan::color(1.0f, 1.0f, 1.0f, 0.01960784383118153f);
    style.Colors[gui::col_text_selected_bg] = fan::color(0.0f, 0.0f, 0.0f, 1.0f);
    style.Colors[gui::col_drag_drop_target] = fan::color(0.168627455830574f, 0.2313725501298904f, 0.5372549295425415f, 1.0f);
    style.Colors[gui::col_nav_cursor] = fan::color(1.0f, 1.0f, 1.0f, 1.0f);
    style.Colors[gui::col_nav_windowing_highlight] = fan::color(1.0f, 1.0f, 1.0f, 0.699999988079071f);
    style.Colors[gui::col_nav_windowing_dim_bg] = fan::color(0.800000011920929f, 0.800000011920929f, 0.800000011920929f, 0.2000000029802322f);
    style.Colors[gui::col_modal_window_dim_bg] = fan::color(0.0f, 0.0f, 0.0f, 0.5647059082984924f);
  }

  static constexpr const int fps_values[] = {0, 30, 60, 144, 165, 240};
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

#undef gui