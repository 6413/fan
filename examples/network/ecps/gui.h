inline static fan::graphics::image_t icon_settings;
inline static fan::graphics::image_t icon_fullscreen;

inline static fan::vec2 popup_size{ 300, 100 };

struct ecps_gui_t {

#define This this

  std::vector<std::function<fan::event::task_t()>> task_queue;

  std::mutex task_queue_mutex;
  void backend_queue(const std::function<fan::event::task_t()>& func) {
    std::lock_guard<std::mutex> lock(task_queue_mutex);
    task_queue.emplace_back(func);
  }

  struct stream_button_renderer_t {
    static bool render_stream_toggle_button(ecps_backend_t::Protocol_ChannelID_t channel_id,
      fan::vec2 button_size = fan::vec2(200.0f, 0)) {
      bool is_streaming = ecps_backend.is_channel_streaming(channel_id);
      const char* text = is_streaming ? "Stop Stream" : "Start Stream";
      fan::vec4 color = is_streaming ? fan::vec4(0.8f, 0.2f, 0.2f, 0.9f) : fan::vec4(0.2f, 0.8f, 0.2f, 0.9f);

      gui::push_style_color(gui::col_button, color);
      bool clicked = gui::button(text, button_size);
      gui::pop_style_color();

      if (clicked) {
        if (is_streaming) {
          ecps_backend.set_channel_streaming(channel_id, false);
        }
        else {
          ecps_backend.share.m_NetworkFlow.Bucket = 0;
          ecps_backend.share.m_NetworkFlow.TimerLastCallAt = fan::event::now();
          ecps_backend.share.m_NetworkFlow.TimerCallCount = 0;
          ecps_backend.share.CalculateNetworkFlowBucket();
          ecps_backend.set_channel_streaming(channel_id, true);
        }
      }
      return clicked;
    }

    static bool render_centered_stream_button(ecps_backend_t::Protocol_ChannelID_t channel_id,
      fan::vec2 available_area,
      fan::vec2 window_content_size,
      f32_t bottom_margin = 50.0f) {
      const char* button_text = ecps_backend.is_channel_streaming(channel_id) ? "Stop Stream" : "Start Stream";

      fan::vec2 text_size = gui::calc_text_size(button_text);
      fan::vec2 button_size = fan::vec2(
        text_size.x + gui::get_style().FramePadding.x * 2,
        text_size.y + gui::get_style().FramePadding.y * 2
      );

      fan::vec2 button_pos = fan::vec2(
        (available_area.x - button_size.x) / 2,
        window_content_size.y - button_size.y - bottom_margin
      );

      gui::set_cursor_pos(button_pos);
      return render_stream_toggle_button(channel_id, button_size);
    }
  };

  struct icon_button_helper_t {
    static bool render_transparent_icon_button(const char* id, fan::graphics::image_t icon,
      fan::vec2 icon_size, fan::vec2 padding = fan::vec2(-1, -1)) {
      if (padding.x < 0) padding = fan::vec2(gui::get_style().FramePadding.y, gui::get_style().FramePadding.y);

      gui::push_style_color(gui::col_button, fan::vec4(0.3f, 0.3f, 0.3f, 0.0f));
      gui::push_style_color(gui::col_button_hovered, fan::vec4(0.5f, 0.5f, 0.5f, 0.9f));
      gui::push_style_color(gui::col_button_active, fan::vec4(0.6f, 0.6f, 0.6f, 1.0f));
      gui::push_style_var(gui::style_var_frame_padding, padding);

      bool clicked = gui::image_button(id, icon, icon_size);

      gui::pop_style_var(1);
      gui::pop_style_color(3);
      return clicked;
    }
  };

  struct button_helper_t {
    static f32_t calculate_text_button_width(const char* text, f32_t extra_padding = 20.0f) {
      fan::vec2 text_size = gui::calc_text_size(text);
      return text_size.x + gui::get_style().FramePadding.x * 2 + extra_padding;
    }

    static bool render_colored_button(const char* text, fan::vec4 color, fan::vec2 size = fan::vec2(0, 0)) {
      gui::push_style_color(gui::col_button, color);
      bool clicked = gui::button(text, size);
      gui::pop_style_color();
      return clicked;
    }
  };

  struct splitter_renderer_t {
    static void render_vertical_splitter(const char* id, f32_t content_height,
      f32_t& unclamped_width, f32_t& clamped_width,
      f32_t min_width, f32_t max_width, bool& is_resizing) {
      gui::push_style_color(gui::col_button, fan::vec4(0.5f, 0.5f, 0.5f, 0.3f));
      gui::push_style_color(gui::col_button_hovered, fan::vec4(0.7f, 0.7f, 0.7f, 0.5f));
      gui::push_style_color(gui::col_button_active, fan::vec4(0.8f, 0.8f, 0.8f, 0.7f));

      f32_t hitbox_width = 8.0f;
      f32_t splitter_width = 2.0f;
      f32_t visual_offset = (hitbox_width - splitter_width) * 0.5f;

      if (gui::invisible_button(id, fan::vec2(hitbox_width, content_height))) {}

      if (gui::is_item_active() && gui::is_mouse_dragging(0)) {
        unclamped_width -= gui::get_io().MouseDelta.x;
        clamped_width = std::clamp(unclamped_width, min_width, max_width);
        is_resizing = true;
        gui::set_mouse_cursor(gui::mouse_cursor_resize_ew);
      }
      else if (is_resizing && !gui::is_mouse_dragging(0)) {
        is_resizing = false;
      }

      if (gui::is_item_hovered()) {
        gui::set_mouse_cursor(gui::mouse_cursor_resize_ew);
      }

      fan::vec2 splitter_pos = gui::get_item_rect_min();
      splitter_pos.x += visual_offset;
      fan::vec2 splitter_max = fan::vec2(splitter_pos.x + splitter_width, splitter_pos.y + content_height);

      gui::get_window_draw_list()->AddRectFilled(
        splitter_pos, splitter_max,
        gui::is_item_hovered() ? gui::get_color_u32(gui::col_button_hovered) : gui::get_color_u32(gui::col_button)
      );

      gui::pop_style_color(3);
    }
  };

  struct stream_area_calculator_t {
    static fan::vec2 calculate_fitted_size(fan::vec2 available_area, fan::vec2 decoded_size) {
      f32_t aspect = 16.0f / 9.0f;
      if (decoded_size.x > 0 && decoded_size.y > 0) {
        aspect = decoded_size.x / decoded_size.y;
      }

      fan::vec2 full_size = (available_area.x / available_area.y > aspect)
        ? fan::vec2(available_area.y * aspect, available_area.y)
        : fan::vec2(available_area.x, available_area.x / aspect);

      return full_size / 2;
    }
  };

  struct network_frame_helper_t {
    #undef This
    #define This OFFSETLESS(this, ecps_gui_t, network_frame_helper) 
    void setup_network_frame(fan::vec2 available_area) {
      auto rt = get_render_thread();
      if (!rt) return;

      fan::vec2 center = available_area / 2;
      rt->network_frame.set_position(center);

      fan::vec2 fitted_size = stream_area_calculator_t::calculate_fitted_size(
        available_area, rt->screen_decoder.decoded_size);

      if (rt->network_frame.get_image() != engine.default_texture) {
        rt->network_frame.set_size(fitted_size);
      }
      else {
        rt->network_frame.set_size(fan::vec2(0, 0));
      }
    }
  }network_frame_helper;
  #undef This
  #define This this

  ecps_gui_t() {

    engine.clear_color = gui::get_color(gui::col_window_bg);

    icon_settings = engine.image_load("icons/settings.png", {
     .min_filter = fan::graphics::image_filter::linear,
     .mag_filter = fan::graphics::image_filter::linear,
      });

    icon_fullscreen = engine.image_load("icons/fullscreen.png", {
     .min_filter = fan::graphics::image_filter::nearest,
     .mag_filter = fan::graphics::image_filter::nearest,
      });

    if (fan::io::file::exists(config_path)) {
      std::string data;
      fan::io::file::read(std::string(config_path), &data);
      config = fan::json::parse(data);
    }

    std::string ip, port;
    if (config.contains("server")) {
      if (config["server"].contains("ip")) {
        ip = config["server"]["ip"];
      }
      if (config["server"].contains("port")) {
        port = config["server"]["port"];
      }
    }
    if (ip.size() && port.size()) {
      backend_queue([=]() -> fan::event::task_t {
        try {
          co_await ecps_backend.connect(ip, string_to_number(port));
          co_await ecps_backend.login();

          auto channel_id = co_await ecps_backend.channel_create();
          co_await ecps_backend.channel_join(channel_id, true);
          co_await ecps_backend.request_channel_list();

          This->selected_channel_id = channel_id;
        }
        catch (...) {

        }
        });
    }
  }
  static uint32_t string_to_number(const std::string& str) {
    uint32_t v = 0;
    try {
      v = std::stoul(str);
    }
    catch (const std::invalid_argument& e) {
      fan::print("The input couldn't be converted to a number");
      return -1;
    }
    catch (const std::out_of_range& e) {
      fan::print("The input was too large for unsigned long");
      return -1;
    }
    return v;
  }

  struct drop_down_server_t {
#undef This
#define This OFFSETLESS(this, ecps_gui_t, drop_down_server)

    void render() {

      bool contains_server = This->config.contains("server");
      static std::string ip = contains_server && This->config["server"].contains("ip") ? This->config["server"]["ip"] : "";
      static std::string port = contains_server && This->config["server"].contains("port") ? This->config["server"]["port"] : "";

      fan::json server_json = This->config.contains("server") ? This->config["server"] : fan::json();

      gui::push_style_var(gui::style_var_window_padding, fan::vec2(13.0000, 20.0000));
      gui::push_style_var(gui::style_var_item_spacing, fan::vec2(14, 16));
      if (toggle_render_server_create) {
        gui::set_next_window_pos(engine.window.get_size() / 2 - popup_size / 2);
        gui::open_popup("server_create");
      }

      if (gui::begin_popup("server_create")) {
        gui::push_item_width(300);
        popup_size = gui::get_window_size();
        gui::text("Channel Create");

        static std::string name;
        gui::input_text("name", &name);

        if (gui::button("Create")) {
          This->backend_queue([=]() -> fan::event::task_t {
            try {
              auto channel_id = co_await ecps_backend.channel_create();
              co_await ecps_backend.channel_join(channel_id, true);
            }
            catch (...) {}
            });

          gui::close_current_popup();
          toggle_render_server_create = false;
        }
        gui::same_line();
        if (gui::button("Cancel")) {
          gui::close_current_popup();
          toggle_render_server_create = false;
        }

        gui::pop_item_width();
        gui::end_popup();
      }

      if (toggle_render_server_connect) {
        gui::set_next_window_pos(engine.window.get_size() / 2 - popup_size / 2);
        gui::open_popup("server_connect");
      }

      if (gui::begin_popup("server_connect")) {

        gui::push_item_width(300);

        popup_size = gui::get_window_size();

        gui::input_text("ip", &ip);
        gui::input_text("port", &port);

        if (gui::button("Connect")) {
          {
            if (ip.size()) {
              server_json["ip"] = ip;
            }
            if (port.size()) {
              server_json["port"] = port;
            }
            if (ip.size() && port.size()) {
              ecps_backend.ip = ip;
              ecps_backend.port = std::stoul(port);
              This->backend_queue([=]() -> fan::event::task_t {
                try {
                  bool did_connect = co_await ecps_backend.connect(ip, string_to_number(port));
                  if (did_connect) {
                    co_await ecps_backend.login();
                    co_await ecps_backend.request_channel_list();
                  }
                }
                catch (...) {
                }
                });
            }
          }

          This->write_to_config("server", server_json);
          gui::close_current_popup();
          toggle_render_server_connect = false;
        }
        gui::same_line();
        if (gui::button("Cancel")) {
          gui::close_current_popup();
          toggle_render_server_connect = false;
        }
        gui::pop_item_width();
        gui::end_popup();
      }

      if (toggle_render_server_join) {
        gui::set_next_window_pos(engine.window.get_size() / 2 - popup_size / 2);
        gui::open_popup("join channel");
      }

      if (gui::begin_popup("join channel")) {
        gui::push_item_width(300);

        popup_size = gui::get_window_size();

        if (gui::button("Connect")) {
          This->write_to_config("server", server_json);
          gui::close_current_popup();
          toggle_render_server_join = false;
        }
        gui::same_line();
        if (gui::button("Cancel")) {
          gui::close_current_popup();
          toggle_render_server_join = false;
        }
        gui::pop_item_width();
        gui::end_popup();
      }

      gui::pop_style_var(2);
    }

    bool toggle_render_server_create = false;
    bool toggle_render_server_connect = false;
    bool toggle_render_server_join = false;
  }drop_down_server;

  void render_stream() {
    fan::vec2 viewport_size = gui::get_content_region_avail();
    fan::vec2 popup_size = fan::vec2(viewport_size.x * 0.8f, 80);
    fan::vec2 stream_pos = gui::get_cursor_screen_pos() + fan::vec2(viewport_size.x / 2 - popup_size.x / 2, 0);
    fan::vec2 start_pos = fan::vec2(stream_pos.x, stream_pos.y + viewport_size.y + 50);
    fan::vec2 target_pos = fan::vec2(stream_pos.x, stream_pos.y + viewport_size.y - 90);

    gui::push_style_var(gui::style_var_frame_rounding, 12.f);
    gui::push_style_var(gui::style_var_window_rounding, 12.f);

    bool trigger_popup = false;
    {
      static fan::vec2 last_mouse = gui::get_io().MousePos;
      fan::vec2 current_mouse_pos = gui::get_io().MousePos;

      fan::vec2 gui_window_size = gui::get_window_size();
      fan::vec2 parent_min = gui::get_window_pos();
      fan::vec2 parent_max = fan::vec2(parent_min.x + gui_window_size.x,
        parent_min.y + gui_window_size.y);

      fan::vec2 hitbox_size = parent_max - parent_min;

      f32_t cursor_y = gui::get_cursor_pos_y();
      gui::set_cursor_pos_y(cursor_y + hitbox_size.y / 2 + (hitbox_size.y * 0.75) / 2);
      hitbox_size.y *= 0.15;
      gui::dummy(hitbox_size);

      bool inside_window = gui::is_item_hovered(gui::hovered_flags_allow_when_overlapped_by_window);

      bool mouse_moved = 0;

      last_mouse = current_mouse_pos;
      trigger_popup = inside_window;
    }

    gui::pop_style_var(2);
  }

  struct stream_settings_t {
    int selected_scaling_quality = 1;
    int selected_resolution = 0;
    int framerate = 30;
    int selected_encoder = 0;
    int selected_decoder = 0;
    int input_control = 0;
    bool show_in_stream_view = false;

    f32_t bitrate_mbps = 10;
    bool use_adaptive_bitrate = true;
    int bitrate_mode = 0;

    f32_t settings_panel_width = 350.0f;
    bool is_resizing = false;

    bool p_open = false;
    bool show_network_debug = false;
  }stream_settings;

  struct channel_list_window_t {
#undef This
#define This OFFSETLESS(this, ecps_gui_t, window_handler)

    void render() {
      if (!p_open) return;

      auto* viewport = gui::get_main_viewport();
      fan::vec2 window_pos = fan::vec2(viewport->WorkPos.x, viewport->WorkPos.y);
      fan::vec2 window_size = fan::vec2(viewport->WorkSize.x, viewport->WorkSize.y);
      gui::set_next_window_pos(window_pos);
      gui::set_next_window_size(window_size);
      gui::set_next_window_bg_alpha(0.99);
      if (gui::begin("Channel Browser", &p_open,
        gui::window_flags_no_docking |
        gui::window_flags_no_saved_settings |
        gui::window_flags_no_move |
        gui::window_flags_no_collapse |
        gui::window_flags_no_background |
        gui::window_flags_no_resize |
        gui::window_flags_no_title_bar)
        ) {

        std::string current_user = "User: " + ecps_backend.get_current_username();
        fan::vec2 text_size = gui::calc_text_size(current_user.c_str());
        gui::same_line(gui::get_window_size().x - text_size.x - 20);
        gui::text(current_user.c_str(), fan::color(0.7f, 0.7f, 1.0f, 1.0f));

        static bool first_open = true;
        if (first_open && p_open) {
          first_open = false;
          This->backend_queue([=]() -> fan::event::task_t {
            try {
              co_await ecps_backend.request_channel_list();
              if (This->selected_channel_id.i != (uint16_t)-1) {
                co_await ecps_backend.request_channel_session_list(This->selected_channel_id);
              }
            }
            catch (...) {
              fan::print("Failed to auto-refresh channel list");
            }
            });
        }

        if (!p_open) {
          first_open = true;
        }

        if (gui::begin_tab_bar("MainTabs")) {

          if (gui::begin_tab_item("Channels", nullptr, main_tab == 0 ? gui::tab_item_flags_set_selected : 0)) {
            if (main_tab == 0) main_tab = -1;

            current_tab = 0;

            render_channels_content();
            gui::end_tab_item();
          }

          if (gui::begin_tab_item("Stream View", nullptr, main_tab == 1 ? gui::tab_item_flags_set_selected : 0)) {
            if (main_tab == 1) main_tab = -1;

            current_tab = 1;

            render_stream_view_content();
            gui::end_tab_item();
          }

          gui::end_tab_bar();
        }
      }
      gui::end();
    }

    void render_channels_content() {
      if (auto rt = get_render_thread(); rt) {
        rt->network_frame.set_size(0);
      }
      engine.viewport_set(engine.orthographic_render_view.viewport, 0, engine.window.get_size());

      fan::vec2 avail_size = gui::get_content_region_avail();
      if (avail_size.x > 0 && avail_size.y > 0) {

        f32_t left_window_padding = 8.0f;
        f32_t right_window_padding = 12.0f;
        f32_t splitter_spacing = 12.0f;
        f32_t splitter_width = 2.0f;

        f32_t total_spacing = left_window_padding + splitter_spacing + splitter_width + splitter_spacing + right_window_padding;

        static f32_t channel_details_width = 0.0f;
        static f32_t unclamped_details_width = 0.0f;
        static bool initialized = false;
        if (!initialized) {
          f32_t available_content_width = avail_size.x - total_spacing;
          channel_details_width = available_content_width * 0.5f;
          unclamped_details_width = channel_details_width;
          initialized = true;
        }

        f32_t min_details_width = 300.0f;
        f32_t max_details_width = avail_size.x * 0.7f;
        channel_details_width = std::clamp(
          unclamped_details_width,
          min_details_width,
          max_details_width
        );

        f32_t channel_list_width = avail_size.x - channel_details_width - total_spacing;

        gui::dummy(fan::vec2(left_window_padding, 0));
        gui::same_line(0, 0);

        gui::set_next_window_bg_alpha(0);
        gui::begin_child("##channel_list_panel", fan::vec2(channel_list_width, avail_size.y), true);

        gui::spacing();
        gui::spacing();

        gui::spacing();
        gui::spacing();

        gui::push_item_width(150);
        gui::input_text("Search", &search_filter);
        gui::pop_item_width();

        gui::spacing();

        gui::spacing();
        gui::spacing();

        gui::set_next_window_bg_alpha(0.99);
        gui::table_flags_t table_flags = gui::table_flags_row_bg
          | gui::table_flags_borders_inner
          | gui::table_flags_borders_outer
          | gui::table_flags_resizable
          | gui::table_flags_sizing_fixed_same;
        if (gui::begin_child("ChannelList", fan::vec2(0, -60), true)) {
          if (gui::begin_table("ChannelTable", 4, table_flags)) {

            gui::table_setup_column("ID", gui::table_column_flags_width_fixed, 50.f, 0);
            gui::table_setup_column("Stream Name", gui::table_column_flags_width_stretch, 0.f, 0);
            gui::table_setup_column("Viewers");
            gui::table_setup_column("Live Time");

            gui::table_headers_row();

            int row_index = 0;
            for (const auto& channel : ecps_backend.available_channels) {
              gui::push_id(row_index);
              if (!search_filter.empty()) {
                std::string name_lower = channel.name;
                std::string filter_lower = search_filter;
                std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
                std::transform(filter_lower.begin(), filter_lower.end(), filter_lower.begin(), ::tolower);
                if (name_lower.find(filter_lower) == std::string::npos) {
                  continue;
                }
              }

              bool is_selected = (This->selected_channel_id.i == channel.channel_id.i);

              bool is_host_of_channel = ecps_backend.is_current_user_host_of_channel(channel.channel_id);

              std::string display_name = channel.name;
              if (is_host_of_channel) {
                display_name = "⭐ " + channel.name;
              }
              if (channel.is_password_protected) {
              }

              bool already_joined = false;
              for (const auto& joined_channel : ecps_backend.channel_info) {
                if (joined_channel.channel_id.i == channel.channel_id.i) {
                  already_joined = true;
                  break;
                }
              }
              gui::table_next_row();
              {
                gui::table_set_column_index(0);
                gui::text(std::to_string(channel.channel_id.i));
              }

              {
                gui::table_set_column_index(1);
                gui::push_style_color(gui::col_header_active, gui::get_style().Colors[gui::col_header_hovered]);

                if (gui::selectable(display_name.c_str(), is_selected, gui::selectable_flags_span_all_columns)) {
                  This->selected_channel_id = channel.channel_id;

                  This->backend_queue([channel_id = channel.channel_id]() -> fan::event::task_t {
                    try {
                      co_await ecps_backend.request_channel_session_list(channel_id);
                    }
                    catch (...) {
                      fan::print("Failed to request session list");
                    }
                    });
                }
                gui::pop_style_color();
              }
              {
                gui::table_set_column_index(2);
                gui::text(std::to_string(channel.user_count - 1));
              }
              {
                gui::table_set_column_index(3);
                uint64_t stream_time_ns = ecps_backend.get_channel_stream_time(channel.channel_id);
                double stream_time_s = stream_time_ns / 1e+9;
                uint64_t hours = stream_time_s / 3600;
                uint64_t minutes = fmod(stream_time_s, 3600.0) / 60.0;
                uint64_t seconds = fmod(stream_time_s, 60.0);
                if (!stream_time_ns) {
                  hours = minutes = seconds = 0;
                }
                gui::text(fan::format("{:02}:{:02}:{:02}", hours, minutes, seconds));
              }
              row_index++;
              gui::pop_id();
            }
            gui::end_table();
          }

          if (ecps_backend.available_channels.empty()) {
            if (ecps_backend.channel_list_received) {
              gui::text("No channels available, create a new channel.");
            }
            else {
              gui::text("Click 'Refresh Channel List' to load channels");
            }
          }
        }
        gui::end_child();

        gui::spacing();

        bool has_selection = (This->selected_channel_id.i != (uint16_t)-1);
        bool already_in_selected = false;
        bool is_host_of_selected = false;

        if (has_selection) {
          for (const auto& joined_channel : ecps_backend.channel_info) {
            if (joined_channel.channel_id.i == This->selected_channel_id.i) {
              already_in_selected = true;

              auto session_it = ecps_backend.channel_sessions.find(This->selected_channel_id.i);
              if (session_it != ecps_backend.channel_sessions.end()) {
                std::string current_username = ecps_backend.get_current_username();
                for (const auto& session : session_it->second) {
                  if (session.username == current_username && session.is_host) {
                    is_host_of_selected = true;
                    break;
                  }
                }
              }
              break;
            }
          }
        }

        if (has_selection && !is_host_of_selected) {
          if (already_in_selected) {
            f32_t button_width = button_helper_t::calculate_text_button_width("Leave");
            if (button_helper_t::render_colored_button("Leave", fan::vec4(0.8f, 0.3f, 0.3f, 1.0f) / 1.1, fan::vec2(button_width, 0))) {
              This->backend_queue([channel_id = This->selected_channel_id]() -> fan::event::task_t {
                try {
                  fan::print("TODO channel_leave");
                  co_return;
                }
                catch (...) {
                  fan::print("Failed to leave channel");
                }
                });
            }
            gui::same_line();
          }
          else {
            f32_t button_width = button_helper_t::calculate_text_button_width("Join");
            if (button_helper_t::render_colored_button("Join", fan::vec4(0.3f, 0.8f, 0.3f, 1.0f) / 1.1, fan::vec2(button_width, 0))) {
              This->backend_queue([channel_id = This->selected_channel_id]() -> fan::event::task_t {
                try {
                  co_await ecps_backend.channel_join(channel_id);
                  co_await ecps_backend.request_channel_list();
                  co_await ecps_backend.request_channel_session_list(channel_id);
                }
                catch (...) {
                  fan::print("Failed to join channel");
                }
                });
            }
            gui::same_line();
          }
        }

        f32_t add_button_width = button_helper_t::calculate_text_button_width("Add new");
        if (button_helper_t::render_colored_button("Add new", fan::vec4(0.3f, 0.3f, 0.3f, 1.0f), fan::vec2(add_button_width, 0))) {
          This->backend_queue([this]() -> fan::event::task_t {
            try {
              auto channel_id = co_await ecps_backend.channel_create();
              co_await ecps_backend.channel_join(channel_id);
              co_await ecps_backend.request_channel_list();
              co_await ecps_backend.request_channel_session_list(channel_id);
              This->selected_channel_id = channel_id;
            }
            catch (...) {
              fan::print("Failed to create channel");
            }
            });
        }
        gui::same_line();
        
        f32_t connect_button_width = button_helper_t::calculate_text_button_width("Connect");
        if (button_helper_t::render_colored_button("Connect", fan::vec4(0.3f, 0.3f, 0.3f, 1.0f), fan::vec2(connect_button_width, 0))) {
          This->drop_down_server.toggle_render_server_connect = true;
        }

        gui::end_child();

        gui::same_line(0, splitter_spacing);

        static bool is_resizing_channels = false;
        splitter_renderer_t::render_vertical_splitter("##channel_splitter_hitbox", avail_size.y,
          unclamped_details_width, channel_details_width, min_details_width, max_details_width, is_resizing_channels);

        gui::same_line(0, splitter_spacing);

        gui::begin_child("##channel_details_panel", fan::vec2(channel_details_width, avail_size.y), true);

        gui::dummy(fan::vec2(12, 0));
        gui::same_line(0, 0);
        gui::begin_group();

        if (This->selected_channel_id.i != (uint16_t)-1) {
          std::string channel_name = "Channel " + std::to_string(This->selected_channel_id.i);
          for (const auto& channel : ecps_backend.available_channels) {
            if (channel.channel_id.i == This->selected_channel_id.i) {
              channel_name = channel.name;
              break;
            }
          }
          gui::text(("Channel: " + channel_name).c_str());

          gui::separator();

          gui::spacing();

          if (gui::begin_tab_bar("ChannelDetailTabs")) {
            gui::spacing();
            gui::spacing();
            if (gui::begin_tab_item("Users", nullptr, detail_tab == 0 ? gui::tab_item_flags_set_selected : 0)) {
              if (detail_tab == 0) detail_tab = -1;

              gui::spacing();
              gui::spacing();
              if (gui::button("Refresh Users")) {
                This->backend_queue([channel_id = This->selected_channel_id]() -> fan::event::task_t {
                  try {
                    co_await ecps_backend.request_channel_session_list(channel_id);
                  }
                  catch (...) {
                    fan::print("Failed to refresh session list");
                  }
                  });
              }

              gui::spacing();
              gui::spacing();
              gui::spacing();

              gui::set_next_window_bg_alpha(0.99);
              if (gui::begin_child("SessionList", fan::vec2(0, -120), true)) {
                auto it = ecps_backend.channel_sessions.find(This->selected_channel_id.i);
                if (it != ecps_backend.channel_sessions.end()) {
                  for (const auto& session : it->second) {
                    std::string user_display = session.username;
                    fan::color user_color = fan::colors::white;
                    if (session.is_host) {
                      user_display += " (Host)";
                      user_color = fan::vec4(1.0f, 0.8f, 0.2f, 1.0f);
                    }

                    gui::text(user_display.c_str(), user_color);
                    gui::same_line();
                    gui::text(("ID: " + std::to_string(session.session_id.i)).c_str());
                  }
                }
                else {
                  gui::text("No user data loaded.");
                  gui::text("Select a channel to view users.");
                }
              }
              gui::end_child();

              gui::spacing();
              if (is_host_of_selected) {
                stream_button_renderer_t::render_stream_toggle_button(This->selected_channel_id);
              }

              gui::end_tab_item();
            }

            if (is_host_of_selected) {
              if (gui::begin_tab_item("Stream Settings", nullptr, detail_tab == 1 ? gui::tab_item_flags_set_selected : 0)) {
                if (detail_tab == 1) detail_tab = -1;

                render_stream_settings_content_compact();
                gui::end_tab_item();
              }
            }

            gui::end_tab_bar();
          }
        }
        else {
          gui::spacing();
          gui::text("Select a channel.");
        }

        gui::end_group();
        gui::end_child();
      }
    }

    void render_stream_settings_content_compact() {
      gui::push_style_var(gui::style_var_item_spacing, fan::vec2(0, 15.f));
      uint32_t channel_id = ecps_backend.channel_info.size() > 0 ? 0 : -1;

      f32_t available_width = gui::get_content_region_avail().x;
      bool is_narrow = available_width < 300.0f;

      gui::spacing();

      if (ecps_backend.is_channel_streaming(This->selected_channel_id)) {
        This->resolution_controls.render_resolution_controls();

        gui::text("Framerate");
        gui::push_item_width(-1);
        do {
          if (is_narrow) {
            gui::slider_int("##framerate_compact", &This->stream_settings.framerate, 15.0f, 120.0f, "%.0f");
          }
          else {
            gui::input_int("##framerate_compact", &This->stream_settings.framerate, 5, 30);
          }
          if (gui::is_item_deactivated_after_edit()) {
            auto* rt = render_thread_ptr.load(std::memory_order_acquire);
            if (rt) {
              rt->screen_encoder.config_.frame_rate = This->stream_settings.framerate;
              {
                std::lock_guard<std::mutex> lock(rt->screen_encoder.mutex);
                rt->screen_encoder.update_flags |= codec_update_e::frame_rate;
              }
            }
          }
        } while (0);
        gui::pop_item_width();

        gui::separator();

        {
          gui::text("Bitrate Mode");
          const char* bitrate_modes[] = { "Automatic", "Manual" };
          if (gui::combo("##bitrate_mode", &This->stream_settings.bitrate_mode, bitrate_modes, 2)) {
            if (This->stream_settings.bitrate_mode == 0) {
              This->stream_settings.use_adaptive_bitrate = true;
            }
            else {
              This->stream_settings.use_adaptive_bitrate = false;
            }
          }

          if (This->stream_settings.bitrate_mode == 1) {
            gui::text("Bitrate (Mbps)");
            gui::push_item_width(-1);
            if (gui::slider_float("##bitrate_compact", &This->stream_settings.bitrate_mbps, 0.5f, 50.0f, "%.1f", gui::slider_flags_always_clamp)) {
              auto* rt = get_render_thread();
              if (rt) {
                rt->screen_encoder.config_.bitrate = This->stream_settings.bitrate_mbps * 1000000;
                {
                  std::lock_guard<std::mutex> lock(rt->screen_encoder.mutex);
                  rt->screen_encoder.update_flags |= codec_update_e::rate_control;
                }
              }
            }
            gui::pop_item_width();
          }
        }

        gui::separator();
        do {
          static auto encoder_names = [] {
            auto* rt = render_thread_ptr.load(std::memory_order_acquire);
            if (!rt) {
              return std::vector<std::string>{"libx264"};
            }

            auto encoders = rt->screen_encoder.get_encoders();
            return encoders;
            }();

          static auto encoder_options = [] {
            std::vector<const char*> names;
            names.reserve(encoder_names.size());

            for (const auto& name : encoder_names) {
              names.push_back(name.c_str());
            }
            return names;
            }();

          gui::text("Encoder");
          gui::push_item_width(-1);

          if (gui::combo("##encoder_compact", &This->stream_settings.selected_encoder,
            encoder_options.data(), encoder_options.size())) {
            auto* rt = render_thread_ptr.load(std::memory_order_acquire);
            if (rt) {
              std::string encoder_name = encoder_names[This->stream_settings.selected_encoder];

              codec_config_t::codec_type_e codec_type;

              if (encoder_name.find("264") != std::string::npos) {
                codec_type = codec_config_t::H264;
              }
              else if (encoder_name.find("265") != std::string::npos ||
                encoder_name.find("hevc") != std::string::npos) {
                codec_type = codec_config_t::H265;
              }
              else if (encoder_name.find("av1") != std::string::npos) {
                codec_type = codec_config_t::AV1;
              }
              else {
                codec_type = codec_config_t::H264;
              }

              rt->screen_encoder.config_.codec = codec_type;
              rt->screen_encoder.new_codec = This->stream_settings.selected_encoder;
              rt->screen_encoder.update_flags |= codec_update_e::codec;
              rt->screen_encoder.encode_write_flags |= codec_update_e::force_keyframe;
            }
          }
          gui::pop_item_width();
        } while (0);

      }
      if (ecps_backend.is_viewing_any_channel()) {
        do {

          static auto decoder_names = [] {
            auto* rt = render_thread_ptr.load(std::memory_order_acquire);
            if (!rt) {
              return std::vector<std::string>{
                "h264_cuvid", "auto-detect", "libx264", "libx265", "libaom-av1"
              };
            }

            auto decoders = rt->screen_decoder.get_decoders();
            return decoders;
            }();
          static bool decoder_inited = false;
          if (decoder_inited == false) {
            decoder_inited = true;
            auto* rt = render_thread_ptr.load(std::memory_order_acquire);
            if (rt) {
              for (int i = 0; i < decoder_names.size(); i++) {
                if (decoder_names[i] == rt->screen_decoder.name) {
                  This->stream_settings.selected_decoder = i;
                  break;
                }
              }
              if (This->stream_settings.selected_decoder >= decoder_names.size()) {
                for (int i = 0; i < decoder_names.size(); i++) {
                  if (decoder_names[i] == "auto-detect") {
                    This->stream_settings.selected_decoder = i;
                    break;
                  }
                }
              }
            }
          }

          static auto decoder_options = [] {
            std::vector<const char*> names;
            names.reserve(decoder_names.size());
            for (const auto& name : decoder_names) {
              names.push_back(name.c_str());
            }
            return names;
            }();

          gui::text("Decoder");
          gui::push_item_width(-1);
          if (gui::combo("##decoder_compact", &This->stream_settings.selected_decoder,
            decoder_options.data(), decoder_options.size())) {
            auto* rt = render_thread_ptr.load(std::memory_order_acquire);
            if (rt) {
              std::string decoder_name = decoder_names[This->stream_settings.selected_decoder];
              std::string codec_type;

              if (decoder_name.find("h264") != std::string::npos || decoder_name == "h264") {
                codec_type = "h264";
              }
              else if (decoder_name.find("hevc") != std::string::npos || decoder_name == "hevc") {
                codec_type = "hevc";
              }
              else if (decoder_name.find("av1") != std::string::npos || decoder_name == "av1") {
                codec_type = "av1";
              }
              else if (decoder_name == "auto-detect") {
                codec_type = "h264";
              }
              else {
                codec_type = "h264";
              }

              rt->screen_decoder.new_codec = This->stream_settings.selected_decoder;
              rt->screen_decoder.update_flags |= codec_update_e::codec;

              rt->screen_decoder.name = codec_type;

              This->request_idr_reset();
            }
          }
          gui::pop_item_width();
        } while (0);
      }

      {
        auto* rt = render_thread_ptr.load(std::memory_order_acquire);
        if (rt && ecps_backend.is_streaming_to_any_channel()) {
          gui::separator();

          gui::text("Scaling Quality");

          static auto quality_options = []() {
            std::vector<const char*> options;
            for (int i = 0; i < 5; ++i) {
              auto quality = static_cast<codec_config_t::scaling_quality_e>(i);
              options.push_back(fan::graphics::get_scaling_quality_name(quality));
            }
            return options;
            }();

          gui::push_item_width(-1);
          if (gui::combo("##scaling_quality", &This->stream_settings.selected_scaling_quality,
            quality_options.data(), quality_options.size())) {

            auto new_quality = static_cast<codec_config_t::scaling_quality_e>(
              This->stream_settings.selected_scaling_quality);

            rt->screen_encoder.update_scaling_quality(new_quality);

#if ecps_debug_prints >= 1
            fan::print("GUI: Changed scaling quality to: " +
              std::string(fan::graphics::get_scaling_quality_name(new_quality)));
#endif
          }
          gui::pop_item_width();
        }
      }

      gui::separator();

      gui::text("Input Control");

      const char* input_control_options[] = {
          "None", "Keyboard Only", "Keyboard + Mouse"
      };
      gui::push_item_width(-1);
      gui::combo("##input_control_compact", &This->stream_settings.input_control, input_control_options,
        sizeof(input_control_options) / sizeof(input_control_options[0]));
      gui::pop_item_width();

      gui::separator();

      if (gui::button(This->stream_settings.show_network_debug ? "Hide Network Debug" : "Show Network Debug", fan::vec2(-1, 0))) {
        This->stream_settings.show_network_debug = !This->stream_settings.show_network_debug;
      }
      if (This->stream_settings.show_network_debug) {
        gui::separator();
        gui::text("Network Debug Info");

        gui::push_style_color(gui::col_child_bg, fan::vec4(0.1f, 0.1f, 0.1f, 0.8f));
        gui::begin_child("##network_debug", fan::vec2(0, 0), 1 |
          fan::graphics::gui::child_flags_auto_resize_x |
          fan::graphics::gui::child_flags_auto_resize_y);

        auto& stats = ecps_backend.view.m_stats;

        gui::text("Ping (ms): " + fan::to_string(ecps_backend.ping_ms, 4));

        gui::text("=== FRAME STATS ===");
        gui::text(("Total Frames: " + std::to_string(stats.Frame_Total)).c_str());
        gui::text(("Dropped Frames: " + std::to_string(stats.Frame_Drop)).c_str());

        if (stats.Frame_Total > 0) {
          double frame_drop_rate = (double)stats.Frame_Drop * 100.0 / stats.Frame_Total;
          gui::text(("Frame Drop Rate: " + std::to_string(frame_drop_rate) + "%").c_str(),
            frame_drop_rate > 5.0 ? fan::color(1, 0.5f, 0.5f, 1) : fan::color(0.5f, 1, 0.5f, 1));
        }

        gui::spacing();

        gui::text("=== PACKET STATS ===");
        gui::text(("Total Packets: " + std::to_string(stats.Packet_Total)).c_str());
        gui::text(("Head Drops: " + std::to_string(stats.Packet_HeadDrop)).c_str());
        gui::text(("Body Drops: " + std::to_string(stats.Packet_BodyDrop)).c_str());

        uint64_t total_packet_drops = stats.Packet_HeadDrop + stats.Packet_BodyDrop;
        gui::text(("Total Packet Drops: " + std::to_string(total_packet_drops)).c_str());

        if (stats.Packet_Total > 0) {
          double packet_drop_rate = (double)total_packet_drops * 100.0 / stats.Packet_Total;
          gui::text(("Packet Drop Rate: " + std::to_string(packet_drop_rate) + "%").c_str(),
            packet_drop_rate > 2.0 ? fan::color(1, 0.5f, 0.5f, 1) : fan::color(0.5f, 1, 0.5f, 1));
        }

        gui::spacing();

        if (ecps_backend.is_streaming_to_any_channel()) {
          gui::text("=== NETWORK FLOW ===");
          auto& flow = ecps_backend.share.m_NetworkFlow;
          gui::text(("Bucket: " + std::to_string(flow.Bucket) + " bits").c_str());
          gui::text(("Bucket Size: " + std::to_string(flow.BucketSize) + " bits").c_str());

          double bucket_percentage = (double)flow.Bucket * 100.0 / flow.BucketSize;
          gui::text(("Bucket Fill: " + std::to_string(bucket_percentage) + "%").c_str(),
            bucket_percentage < 20.0 ? fan::color(1, 0.5f, 0.5f, 1) : fan::color(0.5f, 1, 0.5f, 1));
        }

        gui::end_child();
        gui::pop_style_color();

        gui::spacing();

        if (gui::button("Reset Statistics", fan::vec2(-1, 0))) {
          stats.Frame_Drop = 0;
          stats.Frame_Total = 0;
          stats.Packet_Total = 0;
          stats.Packet_HeadDrop = 0;
          stats.Packet_BodyDrop = 0;
        }
      }

      gui::separator();

      if (gui::button("Reset Settings", fan::vec2(-1, 0))) {
        This->stream_settings.selected_resolution = 0;
        This->stream_settings.framerate = 30;
        This->stream_settings.bitrate_mbps = 5;
        This->stream_settings.input_control = 0;
      }
      gui::pop_style_var();
    }

    void process_stream_settings() {

      fan::vec2 avail_size = gui::get_content_region_avail();
      f32_t button_height = gui::get_frame_height() + gui::get_style().ItemSpacing.y;
      f32_t content_height = avail_size.y - button_height - gui::get_style().ItemSpacing.y;
      f32_t left_window_padding = 8.0f;
      f32_t right_window_padding = 12.0f;
      f32_t splitter_spacing = 12.0f;
      f32_t splitter_width = 2.0f;

      f32_t min_settings_width = 250.0f;
      f32_t max_settings_width = avail_size.x * 0.6f;

      static f32_t unclamped_settings_width = This->stream_settings.settings_panel_width;

      This->stream_settings.settings_panel_width = std::clamp(
        unclamped_settings_width,
        min_settings_width,
        max_settings_width
      );

      f32_t total_spacing = left_window_padding + splitter_spacing + splitter_width + splitter_spacing + right_window_padding;
      f32_t stream_width = avail_size.x - This->stream_settings.settings_panel_width - total_spacing;

      gui::dummy(fan::vec2(left_window_padding, 0));
      gui::same_line(0, 0);

      gui::set_next_window_bg_alpha(0);
      gui::begin_child("##stream_display_split", fan::vec2(stream_width, content_height), false);

      gui::set_viewport(engine.orthographic_render_view);

      if (auto rt = get_render_thread(); rt) {
        fan::vec2 stream_area = gui::get_content_region_avail();
        This->network_frame_helper.setup_network_frame(stream_area);
        This->render_fps_counter(stream_width);

#if ecps_debug_prints >= 3
        static int debug_counter = 0;
        if (++debug_counter % 300 == 1) {
          fan::vec2 fitted_size = stream_area_calculator_t::calculate_fitted_size(stream_area, rt->screen_decoder.decoded_size);
          f32_t frame_aspect = 16.0f / 9.0f;
          if (rt->screen_decoder.decoded_size.x > 0 && rt->screen_decoder.decoded_size.y > 0) {
            frame_aspect = static_cast<f32_t>(rt->screen_decoder.decoded_size.x) / static_cast<f32_t>(rt->screen_decoder.decoded_size.y);
          }
          fan::printf("VIEWER: stream_area={}x{}, decoded={}x{}, aspect={:.3f}, display_size={}x{}",
            stream_area.x, stream_area.y,
            rt->screen_decoder.decoded_size.x, rt->screen_decoder.decoded_size.y,
            frame_aspect, fitted_size.x * 2, fitted_size.y * 2);
        }
#endif
      }

      gui::end_child();

      gui::same_line(0, splitter_spacing);

      splitter_renderer_t::render_vertical_splitter("##stream_splitter_hitbox", content_height,
        unclamped_settings_width, This->stream_settings.settings_panel_width, min_settings_width, max_settings_width, This->stream_settings.is_resizing);

      gui::same_line(0, splitter_spacing);

      gui::begin_child("##stream_settings_panel", fan::vec2(This->stream_settings.settings_panel_width, content_height), true);

      gui::dummy(fan::vec2(12, 0));
      gui::same_line(0, 0);
      gui::begin_group();

      render_stream_settings_content_compact();

      gui::end_group();
      gui::end_child();
    }

    void process_no_stream_settings() {
      fan::vec2 avail_size = gui::get_content_region_avail();
      f32_t button_height = gui::get_frame_height() + gui::get_style().ItemSpacing.y;
      f32_t content_height = avail_size.y - button_height - gui::get_style().ItemSpacing.y;
      f32_t side_padding = 12.0f;

      gui::dummy(fan::vec2(side_padding, 0));
      gui::same_line(0, 0);

      gui::set_next_window_bg_alpha(0);
      gui::begin_child("##stream_display", fan::vec2(avail_size.x - (side_padding * 2), content_height), false);

      gui::set_viewport(engine.orthographic_render_view);

      if (ecps_backend.channel_info.empty()) {
        if (auto rt = get_render_thread(); rt) {
          rt->network_frame.set_image(engine.default_texture);
        }
      }

      if (auto rt = get_render_thread(); rt) {
        fan::vec2 stream_area = gui::get_content_region_avail();
        This->network_frame_helper.setup_network_frame(stream_area);
        This->render_fps_counter(stream_area.x);

        if (rt->network_frame.get_image() == engine.default_texture) {
          gui::set_cursor_pos(stream_area / 2 - fan::vec2(100, 20));
        }
      }

      gui::end_child();
    }

    void render_stream_view_content() {
      gui::spacing();
      gui::spacing();

      fan::vec2 avail_size = gui::get_content_region_avail();

      f32_t button_height = gui::get_frame_height() + gui::get_style().ItemSpacing.y;
      f32_t content_height = avail_size.y - button_height - gui::get_style().ItemSpacing.y;

      if (avail_size.x > 0 && content_height > 0) {
        if (This->stream_settings.show_in_stream_view) {
          process_stream_settings();
        }
        else {
          process_no_stream_settings();
        }
      }

      gui::spacing();

      f32_t main_content_width = This->stream_settings.show_in_stream_view ?
        (avail_size.x - This->stream_settings.settings_panel_width - 40.0f) :
        avail_size.x;

      // Use helper for stream button with proper positioning
      f32_t button_width = 200;
      f32_t center_pos = (main_content_width - button_width) / 2;
      gui::set_cursor_pos_x(center_pos);
      stream_button_renderer_t::render_stream_toggle_button(This->selected_channel_id, fan::vec2(button_width, 0));

      gui::same_line();

      f32_t frame_height = gui::get_frame_height();
      gui::set_cursor_pos_x(main_content_width - (frame_height * 2) - 10);

      // Use helper for icon buttons
      f32_t settings_icon_size = gui::get_text_line_height();
      f32_t settings_equal_padding = gui::get_style().FramePadding.y;
      if (icon_button_helper_t::render_transparent_icon_button("#btn_settings", icon_settings, 
          fan::vec2(settings_icon_size, settings_icon_size), fan::vec2(settings_equal_padding, settings_equal_padding))) {
        This->stream_settings.show_in_stream_view = !This->stream_settings.show_in_stream_view;
      }

      gui::same_line(main_content_width - frame_height);

      f32_t equal_padding = gui::get_style().FramePadding.y;
      if (icon_button_helper_t::render_transparent_icon_button("#btn_fullscreen", icon_fullscreen, 
          fan::vec2(settings_icon_size, settings_icon_size), fan::vec2(equal_padding, equal_padding))) {
        This->is_fullscreen_stream = true;
        engine.window.set_borderless();
      }
    }

    int main_tab = 0;
    int detail_tab = 0;
    int current_tab = 0;
    bool p_open = true;
    bool auto_refresh = true;
    int refresh_interval = 1;
    std::string search_filter;
  }window_handler;

  struct resolution_gui_controls_t {
    int selected_aspect_ratio = 4;
    int selected_resolution = 0;
    std::string custom_resolution = "";

    uint32_t last_applied_width = 0;
    uint32_t last_applied_height = 0;
    bool dropdown_needs_sync = false;

    void render_resolution_controls() {
      auto* rt = render_thread_ptr.load(std::memory_order_acquire);
      uint32_t current_width = 0, current_height = 0;
      if (rt) {
        current_width = rt->screen_encoder.config_.width;
        current_height = rt->screen_encoder.config_.height;
      }

      if (current_width != last_applied_width || current_height != last_applied_height) {
        sync_dropdown_to_current_resolution(current_width, current_height);
        last_applied_width = current_width;
        last_applied_height = current_height;
      }

      gui::text("Aspect Ratio");
      auto aspect_options = rt->screen_encoder.resolution_manager.get_aspect_ratio_options();

      std::vector<std::string> aspect_strings;
      std::vector<const char*> aspect_cstrings;
      for (const auto& [name, ratio] : aspect_options) {
        aspect_strings.push_back(name);
      }
      for (const auto& str : aspect_strings) {
        aspect_cstrings.push_back(str.c_str());
      }

      gui::push_item_width(-1);
      if (gui::combo("##aspect_ratio", &selected_aspect_ratio,
        aspect_cstrings.data(), aspect_cstrings.size())) {
        update_available_resolutions();
        sync_dropdown_to_current_resolution(current_width, current_height);
      }
      gui::pop_item_width();

      gui::separator();

      gui::text("Resolution");

      std::vector<resolution_system_t::resolution_t> available_resolutions = get_filtered_resolutions();

      std::vector<std::string> res_strings;
      std::vector<const char*> res_cstrings;

      bool found_current = false;
      for (int i = 0; i < available_resolutions.size(); i++) {
        const auto& res = available_resolutions[i];
        res_strings.push_back(res.name + " (" + res.category + ")");

        if (res.width == current_width && res.height == current_height) {
          selected_resolution = i;
          found_current = true;
        }
      }

      if (!found_current && current_width > 0 && current_height > 0) {
        resolution_system_t::resolution_t current_res;
        current_res.width = current_width;
        current_res.height = current_height;
        current_res.name = std::to_string(current_width) + "x" + std::to_string(current_height);
        current_res.category = "Current";

        available_resolutions.insert(available_resolutions.begin(), current_res);
        res_strings.insert(res_strings.begin(), current_res.name + " (" + current_res.category + ")");
        selected_resolution = 0;
      }

      for (const auto& str : res_strings) {
        res_cstrings.push_back(str.c_str());
      }

      gui::push_item_width(-1);
      if (gui::combo("##resolution", &selected_resolution,
        res_cstrings.data(), res_cstrings.size())) {
        if (selected_resolution >= 0 && selected_resolution < available_resolutions.size()) {
          apply_resolution(available_resolutions[selected_resolution]);
        }
      }
      gui::pop_item_width();

      gui::separator();
    }

  private:
    std::vector<resolution_system_t::resolution_t> get_filtered_resolutions() {
      auto* rt = render_thread_ptr.load(std::memory_order_acquire);
      if (!rt) return {};
      if (selected_aspect_ratio == 4) {
        return rt->screen_encoder.resolution_manager.detected_info.matching_aspect_resolutions;
      }
      else {
        auto aspect_options = rt->screen_encoder.resolution_manager.get_aspect_ratio_options();
        f32_t selected_aspect = aspect_options[selected_aspect_ratio].second;
        return resolution_system_t::get_resolutions_by_aspect(selected_aspect, 0.01f);
      }
    }

    void sync_dropdown_to_current_resolution(uint32_t current_width, uint32_t current_height) {
      if (current_width == 0 || current_height == 0) return;

      auto* rt = render_thread_ptr.load(std::memory_order_acquire);
      if (!rt) return;

      f32_t current_aspect = static_cast<f32_t>(current_width) / current_height;
      auto aspect_options = rt->screen_encoder.resolution_manager.get_aspect_ratio_options();

      for (int i = 0; i < aspect_options.size(); i++) {
        if (std::abs(aspect_options[i].second - current_aspect) < 0.02f) {
          selected_aspect_ratio = i;
          break;
        }
      }

      auto available_resolutions = get_filtered_resolutions();
      selected_resolution = 0;

      for (int i = 0; i < available_resolutions.size(); i++) {
        if (available_resolutions[i].width == current_width &&
          available_resolutions[i].height == current_height) {
          selected_resolution = i;
          break;
        }
      }
    }

    void update_available_resolutions() {
      selected_resolution = 0;
    }

    void apply_resolution(const resolution_system_t::resolution_t& resolution) {
      auto* rt = render_thread_ptr.load(std::memory_order_acquire);
      if (rt) {
        rt->screen_encoder.set_user_resolution(resolution.width, resolution.height);
        rt->screen_encoder.encode_write_flags |= codec_update_e::force_keyframe;

        if (rt->ecps_gui.stream_settings.bitrate_mode == 0) {
          rt->screen_encoder.config_.bitrate = dynamic_config_t::get_adaptive_bitrate();
          {
            std::lock_guard<std::mutex> lock(rt->screen_encoder.mutex);
            rt->screen_encoder.update_flags |= codec_update_e::rate_control;
          }
        }

        last_applied_width = resolution.width;
        last_applied_height = resolution.height;
      }
    }

    void parse_and_apply_custom_resolution() {
      size_t x_pos = custom_resolution.find('x');
      if (x_pos == std::string::npos) x_pos = custom_resolution.find('X');

      if (x_pos != std::string::npos && x_pos > 0 && x_pos < custom_resolution.length() - 1) {
        try {
          uint32_t width = std::stoul(custom_resolution.substr(0, x_pos));
          uint32_t height = std::stoul(custom_resolution.substr(x_pos + 1));

          if (width >= 640 && width <= 7680 && height >= 480 && height <= 4320) {
            resolution_system_t::resolution_t custom_res;
            custom_res.width = width;
            custom_res.height = height;
            custom_res.name = std::to_string(width) + "x" + std::to_string(height);
            custom_res.category = "Custom";
            apply_resolution(custom_res);
          }
          else {
            fan::print("Resolution out of bounds: " + custom_resolution);
          }
        }
        catch (const std::exception& e) {
          fan::print("Invalid resolution format: " + custom_resolution);
        }
      }
    }
  }resolution_controls;

#undef This
#define This this

  void request_idr_reset() {
    backend_queue([]() -> fan::event::task_t {
      co_await ecps_backend.request_idr_reset();
      });
  }

  void render_fullscreen_stream() {
    auto* viewport = gui::get_main_viewport();
    fan::vec2 window_pos = fan::vec2(viewport->WorkPos.x, viewport->WorkPos.y);
    fan::vec2 window_size = fan::vec2(viewport->WorkSize.x, viewport->WorkSize.y);
    gui::set_next_window_pos(window_pos);
    gui::set_next_window_size(window_size);
    gui::set_next_window_bg_alpha(1.0f);

    gui::push_style_var(gui::style_var_window_padding, fan::vec2(0, 0));
    gui::push_style_var(gui::style_var_window_border_size, 0.0f);
    gui::push_style_var(gui::style_var_window_rounding, 0.0f);
    gui::push_style_var(gui::style_var_item_spacing, fan::vec2(0, 0));
    gui::push_style_var(gui::style_var_frame_padding, fan::vec2(0, 0));

    if (gui::begin("##FullscreenStream", nullptr,
      gui::window_flags_no_docking |
      gui::window_flags_no_saved_settings |
      gui::window_flags_no_move |
      gui::window_flags_no_collapse |
      gui::window_flags_no_resize |
      gui::window_flags_no_title_bar |
      gui::window_flags_no_scrollbar |
      gui::window_flags_no_scroll_with_mouse |
      gui::window_flags_no_background)) {

      fan::vec2 window_content_size = gui::get_content_region_avail();

      gui::set_viewport(engine.orthographic_render_view);

      fan::vec2 stream_area = window_content_size;
      if (This->stream_settings.show_in_stream_view) {
        stream_area.x -= This->stream_settings.settings_panel_width;
      }

      auto rt = get_render_thread();
      if (!rt) {
        gui::pop_style_var(5);
        gui::end();
        return;
      }

      This->network_frame_helper.setup_network_frame(stream_area);
      This->render_fps_counter(stream_area.x);

      gui::pop_style_var(5);

      if (This->stream_settings.show_in_stream_view) {
        gui::set_cursor_pos(fan::vec2(stream_area.x, 0));
        gui::begin_child("##fullscreen_settings_panel",
          fan::vec2(This->stream_settings.settings_panel_width, window_content_size.y),
          true);

        gui::dummy(fan::vec2(12, 0));
        gui::same_line(0, 0);
        gui::begin_group();

        window_handler.render_stream_settings_content_compact();

        gui::end_group();
        gui::end_child();
      }

      gui::push_style_var(gui::style_var_frame_padding, fan::vec2(12, 8));

      window_content_size = gui::get_content_region_avail();
      stream_button_renderer_t::render_centered_stream_button(This->selected_channel_id, stream_area, window_content_size);

      gui::pop_style_var(1);

      f32_t icon_size = 24.0f;
      f32_t button_spacing = 10.0f;
      f32_t margin = 20.0f;

      f32_t current_y = gui::get_cursor_pos().y;
      
      f32_t settings_x = stream_area.x - (icon_size * 2) - button_spacing - margin;
      gui::set_cursor_pos(fan::vec2(settings_x, current_y));

      if (icon_button_helper_t::render_transparent_icon_button("#btn_fullscreen_settings", icon_settings, fan::vec2(icon_size, icon_size))) {
        This->stream_settings.show_in_stream_view = !This->stream_settings.show_in_stream_view;
      }

      gui::same_line();

      if (icon_button_helper_t::render_transparent_icon_button("#btn_exit_fullscreen", icon_fullscreen, fan::vec2(icon_size, icon_size))) {
        is_fullscreen_stream = false;
        engine.window.set_windowed();
      }
    }
    else {
      gui::pop_style_var(5);
    }

    gui::end();
  }

  void render_fps_counter(f32_t stream_width, f32_t font_size = 15.f) {
    auto rt = get_render_thread();
    if (!rt || rt->network_frame.get_image() == engine.default_texture) {
      return;
    }

    f32_t decoder_fps = rt->displayed_fps;
    std::string fps_text = fan::to_string(decoder_fps, 1);

    gui::push_font(gui::get_font(font_size));

    fan::vec2 text_size = gui::calc_text_size(fps_text);
    fan::vec2 bg_padding = fan::vec2(gui::get_font_size() * 0.4f, gui::get_font_size() * 0.2f);
    fan::vec2 fps_pos = fan::vec2(stream_width - text_size.x - 20, 10);

    gui::set_cursor_pos(fps_pos);

    fan::vec2 bg_min = gui::get_cursor_screen_pos() - bg_padding;
    fan::vec2 bg_max = bg_min + text_size + bg_padding * 2;

    gui::get_window_draw_list()->AddRectFilled(
      bg_min, bg_max,
      fan::color(0, 0, 0, 0.7f).to_u32(),
      gui::get_font_size() * 0.2f
    );

    gui::text(fps_text, fan::color(0.5f, 1, 0.5f, 1));
    gui::pop_font();
  }

  void render() {
    if (is_fullscreen_stream) {
      render_fullscreen_stream();
      return;
    }

    drop_down_server.render();

    gui::set_next_window_pos(0);
    gui::set_next_window_size(gui::get_io().DisplaySize);
    gui::begin("A", 0, gui::window_flags_no_docking |
      gui::window_flags_no_saved_settings |
      gui::window_flags_no_focus_on_appearing |
      gui::window_flags_no_move |
      gui::window_flags_no_collapse |
      gui::window_flags_no_background |
      gui::window_flags_no_resize |
      gui::window_flags_no_title_bar |
      gui::window_flags_no_bring_to_front_on_focus |
      gui::window_flags_no_inputs);

    if (window_handler.p_open) {
      window_handler.render();
    }

    render_stream();

    gui::end();
  }

  inline static const char* config_path = "config.json";
  void write_to_config(const std::string& key, auto value) {
    config[key] = value;
    fan::io::file::write(config_path, config.dump(2), std::ios_base::binary);
  }

  bool is_fullscreen_stream = false;

  fan::json config;

  bool show_own_stream = false;
  ecps_backend_t::Protocol_ChannelID_t selected_channel_id{ (uint16_t)-1 };

#undef This
#undef engine
};