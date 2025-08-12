inline static fan::graphics::image_t icon_settings;

inline static fan::vec2 popup_size{ 300, 100 };

inline static std::string directory_path = "./";

struct ecps_gui_t {

#define This this

  std::vector<std::function<fan::event::task_t()>> task_queue;

  //#define render_thread (*OFFSETLESS(This, render_thread_t, ecps_gui))
  void backend_queue(const std::function<fan::event::task_t()>& func) {
    render_mutex.lock();
    task_queue.emplace_back(func);
    render_mutex.unlock();
  }
  //#undef render_thread

  ecps_gui_t() {

    engine.clear_color = gui::get_color(gui::col_window_bg);

    icon_settings = engine.image_load(directory_path + "icons/settings.png", {
     .min_filter = fan::graphics::image_filter::linear,
     .mag_filter = fan::graphics::image_filter::linear,
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
      static std::string channel = (contains_server && This->config["server"].contains("channel"))
        ? (This->config["server"]["channel"].is_string()
          ? This->config["server"]["channel"].get<std::string>()
          : std::to_string(This->config["server"]["channel"].get<int>()))
        : "";

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

        // create channel
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
        gui::input_text("channel id", &channel);

        if (gui::button("Connect")) {
          {
            if (ip.size()) {
              server_json["ip"] = ip;
            }
            if (port.size()) {
              server_json["port"] = port;
            }
            if (ip.size() && port.size()) {
              This->backend_queue([=]() -> fan::event::task_t {
                try {
                  co_await ecps_backend.connect(ip, string_to_number(port));
                  co_await ecps_backend.login();
                  co_await ecps_backend.request_channel_list();
                }
                catch (...) {}
                });
            }
          }

          This->write_to_config("server", server_json);
          // server creation logic
          gui::close_current_popup();
          toggle_render_server_connect = false;
        }//
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

        gui::input_text("channel id", &channel);

        if (gui::button("Connect")) {
          if (channel.size()) {
            server_json["channel"] = channel;
          }

          if (channel.size()) {
            This->backend_queue([=]() -> fan::event::task_t {
              try {
                if (channel.size()) {
                  co_await ecps_backend.channel_join(string_to_number(channel));
                }
              }
              catch (...) {}
              });
          }

          This->write_to_config("server", server_json);
          // server creation logic
          gui::close_current_popup();
          toggle_render_server_join = false;
        }//
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

      // 15% of screen from center to bottom
      f32_t cursor_y = gui::get_cursor_pos_y();
      gui::set_cursor_pos_y(cursor_y + hitbox_size.y / 2 + (hitbox_size.y * 0.75) / 2);
      hitbox_size.y *= 0.15;
      gui::dummy(hitbox_size);

      bool inside_window = gui::is_item_hovered(gui::hovered_flags_allow_when_overlapped_by_window);

      bool mouse_moved = /* && (current_mouse_pos.x != last_mouse.x || current_mouse_pos.y != last_mouse.y)*/0;

      last_mouse = current_mouse_pos;
      trigger_popup = inside_window/* && mouse_moved*/;
    }

    gui::pop_style_var(2);
  }

  struct stream_settings_t {
    int selected_resolution = 0;
    f32_t framerate = 30;
    f32_t bitrate_mbps = 5;
    int selected_encoder = 0;
    int selected_decoder = 0;
    int input_control = 0;
    bool show_in_stream_view = true;

    f32_t settings_panel_width = 350.0f;
    bool is_resizing = false;

    bool p_open = false;
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
              if (selected_channel_id.i != (uint16_t)-1) {
                co_await ecps_backend.request_channel_session_list(selected_channel_id);
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

        // MAIN TAB BAR
        if (gui::begin_tab_bar("MainTabs")) {

          // CHANNELS TAB
          if (gui::begin_tab_item("Channels", nullptr, main_tab == 0 ? gui::tab_item_flags_set_selected : 0)) {
            if (main_tab == 0) main_tab = -1;

            current_tab = 0;

            render_channels_content();
            gui::end_tab_item();
          }

          // STREAM VIEW TAB
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

        if (gui::button("Refresh Channel List")) {
          This->backend_queue([=]() -> fan::event::task_t {
            try {
              co_await ecps_backend.request_channel_list();
            }
            catch (...) {
              fan::print("Failed to request channel list");
            }
            });
        }

        gui::spacing();
        gui::spacing();

        gui::push_item_width(150);
        gui::input_text("Search", &search_filter);
        gui::pop_item_width();

        gui::spacing();

        gui::spacing();
        gui::spacing();

        //if (ecps_backend.available_channels.size() && !ecps_backend.is_channel_available(selected_channel_id)) {
        //  selected_channel_id.invalidate();
        //}

        gui::set_next_window_bg_alpha(0.99);
        if (gui::begin_child("ChannelList", fan::vec2(0, -60), true)) {
          if (gui::begin_table("ChannelTable", 1, gui::table_flags_row_bg)) {
            int row_index = 0;
            for (const auto& channel : ecps_backend.available_channels) {
              if (!search_filter.empty()) {
                std::string name_lower = channel.name;
                std::string filter_lower = search_filter;
                std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
                std::transform(filter_lower.begin(), filter_lower.end(), filter_lower.begin(), ::tolower);
                if (name_lower.find(filter_lower) == std::string::npos) {
                  continue;
                }
              }

              bool is_selected = (selected_channel_id.i == channel.channel_id.i);
              std::string display_name = " " + channel.name;
              if (channel.is_password_protected) {
                //display_name += "";
              }
              display_name += " (" + std::to_string(channel.user_count) + " users)";

              bool already_joined = false;
              for (const auto& joined_channel : ecps_backend.channel_info) {
                if (joined_channel.channel_id.i == channel.channel_id.i) {
                  already_joined = true;
                  break;
                }
              }

              gui::table_next_row();
              gui::table_set_column_index(0);

              gui::push_style_color(gui::col_header_active, gui::get_style().Colors[gui::col_header_hovered]);

              if (gui::selectable(display_name.c_str(), is_selected, gui::selectable_flags_span_all_columns)) {
                selected_channel_id = channel.channel_id;

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
              row_index++;
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

        // Bottom buttons
        bool has_selection = (selected_channel_id.i != (uint16_t)-1);
        bool already_in_selected = false;
        bool is_host_of_selected = false;

        if (has_selection) {
          for (const auto& joined_channel : ecps_backend.channel_info) {
            if (joined_channel.channel_id.i == selected_channel_id.i) {
              already_in_selected = true;

              auto session_it = ecps_backend.channel_sessions.find(selected_channel_id.i);
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
            gui::push_style_color(gui::col_button, fan::vec4(0.8f, 0.3f, 0.3f, 1.0f) / 1.1);
            std::string leave_text = "Leave";
            fan::vec2 text_size = gui::calc_text_size(leave_text.c_str());
            f32_t button_width = text_size.x + gui::get_style().FramePadding.x * 2 + 20;

            if (gui::button(leave_text.c_str(), fan::vec2(button_width, 0))) {
              This->backend_queue([channel_id = selected_channel_id]() -> fan::event::task_t {
                try {
                  fan::print("TODO channel_leave");
                  co_return;
                }
                catch (...) {
                  fan::print("Failed to leave channel");
                }
                });
            }
            gui::pop_style_color();
            gui::same_line();
          }
          else {
            gui::push_style_color(gui::col_button, fan::vec4(0.3f, 0.8f, 0.3f, 1.0f) / 1.1);
            std::string text = "Join";
            fan::vec2 text_size = gui::calc_text_size(text);
            f32_t button_width = text_size.x + gui::get_style().FramePadding.x * 2 + 20;

            if (gui::button(text, fan::vec2(button_width, 0))) {
              This->backend_queue([channel_id = selected_channel_id]() -> fan::event::task_t {
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
            gui::pop_style_color();
            gui::same_line();
          }
        }

        gui::push_style_color(gui::col_button, fan::vec4(0.3f, 0.3f, 0.3f, 1.0f));
        {
          std::string text = "Add new";
          fan::vec2 text_size = gui::calc_text_size(text.c_str());
          f32_t button_width = text_size.x + gui::get_style().FramePadding.x * 2 + 20;
          if (gui::button(text, fan::vec2(button_width, 0))) {
            This->backend_queue([this]() -> fan::event::task_t {
              try {
                auto channel_id = co_await ecps_backend.channel_create();
                co_await ecps_backend.channel_join(channel_id);
                co_await ecps_backend.request_channel_list();
                co_await ecps_backend.request_channel_session_list(channel_id);
                selected_channel_id = channel_id;
              }
              catch (...) {
                fan::print("Failed to create channel");
              }
              });
          }
        }
        gui::same_line();
        {
          std::string text = "Connect";
          fan::vec2 text_size = gui::calc_text_size(text.c_str());
          f32_t button_width = text_size.x + gui::get_style().FramePadding.x * 2 + 20;
          if (gui::button(text, fan::vec2(button_width, 0))) {
            This->drop_down_server.toggle_render_server_connect = true;
          }
        }
        gui::pop_style_color();

        gui::end_child();

        gui::same_line(0, splitter_spacing);

        gui::push_style_color(gui::col_button, fan::vec4(0.5f, 0.5f, 0.5f, 0.3f));
        gui::push_style_color(gui::col_button_hovered, fan::vec4(0.7f, 0.7f, 0.7f, 0.5f));
        gui::push_style_color(gui::col_button_active, fan::vec4(0.8f, 0.8f, 0.8f, 0.7f));

        static bool is_resizing_channels = false;

        f32_t hitbox_width = 8.0f;
        f32_t visual_offset = (hitbox_width - splitter_width) * 0.5f;

        if (gui::invisible_button("##channel_splitter_hitbox", fan::vec2(hitbox_width, avail_size.y))) {
        }

        if (gui::is_item_active() && gui::is_mouse_dragging(0)) {
          fan::vec2 mouse_delta = gui::get_io().MouseDelta;
          unclamped_details_width -= mouse_delta.x;
          channel_details_width = std::clamp(unclamped_details_width, min_details_width, max_details_width);
          is_resizing_channels = true;
          gui::set_mouse_cursor(gui::mouse_cursor_resize_ew);
        }
        else if (is_resizing_channels && !gui::is_mouse_dragging(0)) {
          is_resizing_channels = false;
        }

        if (gui::is_item_hovered()) {
          gui::set_mouse_cursor(gui::mouse_cursor_resize_ew);
        }

        fan::vec2 splitter_pos = gui::get_item_rect_min();
        splitter_pos.x += visual_offset;
        fan::vec2 splitter_max = fan::vec2(splitter_pos.x + splitter_width, splitter_pos.y + avail_size.y);

        gui::get_window_draw_list()->AddRectFilled(
          splitter_pos,
          splitter_max,
          gui::is_item_hovered() ? gui::get_color_u32(gui::col_button_hovered) : gui::get_color_u32(gui::col_button)
        );

        gui::pop_style_color(3);

        gui::same_line(0, splitter_spacing);

        gui::begin_child("##channel_details_panel", fan::vec2(channel_details_width, avail_size.y), true);

        gui::dummy(fan::vec2(12, 0));
        gui::same_line(0, 0);
        gui::begin_group();

        if (selected_channel_id.i != (uint16_t)-1) {
          std::string channel_name = "Channel " + std::to_string(selected_channel_id.i);
          for (const auto& channel : ecps_backend.available_channels) {
            if (channel.channel_id.i == selected_channel_id.i) {
              channel_name = channel.name;
              break;
            }
          }
          gui::text(("Channel: " + channel_name).c_str());

          if (has_selection && is_host_of_selected) {
            gui::text("You are the host of this channel", fan::color(1.0f, 0.8f, 0.2f, 1.0f));
          }

          gui::separator();

          gui::spacing();

          if (gui::begin_tab_bar("ChannelDetailTabs")) {
            gui::spacing();
            gui::spacing();
            // USERS TAB
            if (gui::begin_tab_item("Users", nullptr, detail_tab == 0 ? gui::tab_item_flags_set_selected : 0)) {
              if (detail_tab == 0) detail_tab = -1;

              gui::spacing();
              gui::spacing();
              if (gui::button("Refresh Users")) {
                This->backend_queue([channel_id = selected_channel_id]() -> fan::event::task_t {
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
                auto it = ecps_backend.channel_sessions.find(selected_channel_id.i);
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
              {
                fan::vec2 icon_size = 48.f;
                fan::vec2 button_padding = gui::get_style().FramePadding;
                fan::vec2 button_size = icon_size + button_padding * 2;
                f32_t button_width = 200;

                bool is_streaming_current = ecps_backend.is_channel_streaming(selected_channel_id);

                if (is_streaming_current) {
                  gui::push_style_color(gui::col_button, fan::vec4(0.8f, 0.2f, 0.2f, 1.0f));
                  if (gui::button("Stop Stream", fan::vec2(button_width, button_size.y))) {
                    ecps_backend.set_channel_streaming(selected_channel_id, false);
                  }
                  gui::pop_style_color();
                }
                else {
                  gui::push_style_color(gui::col_button, fan::vec4(0.2f, 0.8f, 0.2f, 1.0f));
                  if (gui::button("Start Stream", fan::vec2(button_width, button_size.y))) {
                    ecps_backend.share.m_NetworkFlow.Bucket = 0;
                    ecps_backend.share.m_NetworkFlow.TimerLastCallAt = fan::event::now();
                    ecps_backend.share.m_NetworkFlow.TimerCallCount = 0;
                    ecps_backend.share.CalculateNetworkFlowBucket();
                    ecps_backend.set_channel_streaming(selected_channel_id, true);
                  }
                  gui::pop_style_color();
                }

                gui::same_line();
                fan::vec2 avail = gui::get_content_region_avail();
                gui::dummy(fan::vec2(avail.x - button_size.x, 0));
                gui::same_line();

                if (gui::image_button("#btn_stream_settings", icon_settings, icon_size)) {
                  detail_tab = 1;
                }

                gui::same_line();
                if (gui::button("View Stream")) {
                  main_tab = 1;
                }
              }

              gui::end_tab_item();
            }

            if (gui::begin_tab_item("Stream Settings", nullptr, detail_tab == 1 ? gui::tab_item_flags_set_selected : 0)) {
              if (detail_tab == 1) detail_tab = -1;

              render_stream_settings_content_compact();
              gui::end_tab_item();
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

      if (ecps_backend.is_channel_streaming(This->window_handler.selected_channel_id)) {
        This->resolution_controls.render_resolution_controls();
      }

      gui::text("Framerate");
      gui::push_item_width(-1);
      do {
        if (is_narrow) {
          gui::slider_float("##framerate_compact", &This->stream_settings.framerate, 15.0f, 120.0f, "%.0f");
        }
        else {
          gui::input_float("##framerate_compact", &This->stream_settings.framerate, 5, 30);
        }
        if (gui::is_item_deactivated_after_edit()) {
          auto* rt = render_thread_ptr.load(std::memory_order_acquire);
          if (rt) {
            rt->screen_encoder.config_.frame_rate = This->stream_settings.framerate;
            rt->screen_encoder.encoder_.update_config(rt->screen_encoder.config_, fan::graphics::codec_update_e::frame_rate);
          }
        }
      } while (0);
      gui::pop_item_width();

      gui::separator();

      gui::text("Bitrate (Mbps)");
      gui::push_item_width(-1);
      do {
        gui::slider_float("##bitrate_compact", &This->stream_settings.bitrate_mbps, 0.5f, 50.0f, "%.1f");
        if (gui::is_item_deactivated_after_edit()) {
          if (channel_id == (uint32_t)-1) break;
          auto* rt = get_render_thread();
          if (rt) {
            rt->screen_encoder.config_.bitrate = This->stream_settings.bitrate_mbps * 1000000;
            rt->screen_encoder.encoder_.update_config(rt->screen_encoder.config_, codec_update_e::rate_control);
          }
        }
      } while (0);
      gui::pop_item_width();
      gui::separator();
      if (ecps_backend.is_streaming_to_any_channel()) {
        do {
          static auto encoder_names = [] {
            auto* rt = render_thread_ptr.load(std::memory_order_acquire);
            if (!rt) {
              return std::vector<std::string>{"libx264"}; // Fallback
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
              rt->screen_encoder.new_codec = This->stream_settings.selected_encoder;
              rt->screen_encoder.update_flags |= codec_update_e::codec;
              rt->screen_encoder.encode_write_flags |= codec_update_e::force_keyframe;
            }
          }
          gui::pop_item_width();
        } while (0);

      }
      if (ecps_backend.is_viewing_any_channel()) {
        // Decoder
        do {
          static auto decoder_names = [] {
            auto* rt = render_thread_ptr.load(std::memory_order_acquire);
            if (!rt) {
              return std::vector<std::string>{
                "auto-detect", "libx264", "libx265", "libaom-av1"
              }; // Fallback
            }

            auto decoders = rt->screen_decoder.get_decoders();
            return decoders;
            }();
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
              rt->screen_decoder.new_codec = This->stream_settings.selected_decoder;
              rt->screen_decoder.update_flags |= fan::graphics::codec_update_e::codec;

              This->backend_queue([=]() -> fan::event::task_t {
                try {
                  for (const auto& channel : ecps_backend.channel_info) {
                    if (channel.is_viewing) {
                      ecps_backend_t::Protocol_C2S_t::Channel_ScreenShare_ViewToShare_t rest;
                      rest.ChannelID = channel.channel_id;
                      rest.Flag = ecps_backend_t::ProtocolChannel::ScreenShare::ChannelFlag::ResetIDR;
                      co_await ecps_backend.tcp_write(
                        ecps_backend_t::Protocol_C2S_t::Channel_ScreenShare_ViewToShare,
                        &rest,
                        sizeof(rest)
                      );
                    }
                  }
                }
                catch (...) {}
                });
            }
          }
          gui::pop_item_width();
        } while (0);
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

      if (gui::button("Reset Settings", fan::vec2(-1, 0))) {
        This->stream_settings.selected_resolution = 0;
        This->stream_settings.framerate = 30;
        This->stream_settings.bitrate_mbps = 5;
        This->stream_settings.input_control = 0;
      }
      gui::pop_style_var();
    }

    void render_stream_view_content() {
      gui::spacing();
      gui::spacing();

      bool is_streaming_current = ecps_backend.is_channel_streaming(selected_channel_id);

      if (is_streaming_current) {
        gui::push_style_color(gui::col_button, fan::vec4(0.8f, 0.2f, 0.2f, 1.0f));
        if (gui::button("Stop Stream")) {
          ecps_backend.set_channel_streaming(selected_channel_id, false);
        }
        gui::pop_style_color();
      }
      else {
        gui::push_style_color(gui::col_button, fan::vec4(0.2f, 0.8f, 0.2f, 1.0f));
        if (gui::button("Start Stream")) {
          ecps_backend.share.m_NetworkFlow.Bucket = 0;
          ecps_backend.share.m_NetworkFlow.TimerLastCallAt = fan::event::now();
          ecps_backend.share.m_NetworkFlow.TimerCallCount = 0;
          ecps_backend.share.CalculateNetworkFlowBucket();
          ecps_backend.set_channel_streaming(selected_channel_id, true);
        }
        gui::pop_style_color();
      }

      gui::same_line(0, 20);
      gui::checkbox("Show Own Stream", &This->show_own_stream);

      gui::same_line(0, 20);
      gui::push_style_color(gui::col_button,
        This->stream_settings.show_in_stream_view ?
        fan::vec4(0.8f, 0.6f, 0.2f, 1.0f) : fan::vec4(0.2f, 0.6f, 0.8f, 1.0f));

      if (gui::button(This->stream_settings.show_in_stream_view ? "Hide Settings" : "Show Settings")) {
        This->stream_settings.show_in_stream_view = !This->stream_settings.show_in_stream_view;
      }
      gui::pop_style_color();

      gui::same_line(0, 20);
      gui::push_style_color(gui::col_button, fan::vec4(0.2f, 0.6f, 0.8f, 1.0f));
      if (gui::button("Fullscreen")) {
        This->is_fullscreen_stream = true;
      }
      gui::pop_style_color();

      gui::spacing();
      gui::spacing();


      fan::vec2 avail_size = gui::get_content_region_avail();
      if (avail_size.x > 0 && avail_size.y > 0) {

        if (This->stream_settings.show_in_stream_view) {

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
          gui::begin_child("##stream_display_split", fan::vec2(stream_width, avail_size.y), false);

          gui::set_viewport(engine.orthographic_render_view);

          if (auto rt = get_render_thread(); rt) {
            fan::vec2 stream_area = gui::get_content_region_avail();
            fan::vec2 center = stream_area / 2;
            rt->network_frame.set_position(center);
            f32_t frame_aspect = 16.0f / 9.0f;

            if (rt->screen_decoder.decoded_size.x > 0 && rt->screen_decoder.decoded_size.y > 0) {
              frame_aspect = static_cast<f32_t>(rt->screen_decoder.decoded_size.x) /
                rt->screen_decoder.decoded_size.y;
            }

            fan::vec2 full_size = (stream_area.x / stream_area.y > frame_aspect)
              ? fan::vec2(stream_area.y * frame_aspect, stream_area.y)
              : fan::vec2(stream_area.x, stream_area.x / frame_aspect);

            full_size /= 2;

            if (rt->network_frame.get_image() != engine.default_texture) {
              rt->network_frame.set_size(full_size);
            }
          }

          gui::end_child();

          gui::same_line(0, splitter_spacing);

          gui::push_style_color(gui::col_button, fan::vec4(0.5f, 0.5f, 0.5f, 0.3f));
          gui::push_style_color(gui::col_button_hovered, fan::vec4(0.7f, 0.7f, 0.7f, 0.5f));
          gui::push_style_color(gui::col_button_active, fan::vec4(0.8f, 0.8f, 0.8f, 0.7f));

          f32_t hitbox_width = 8.0f;
          f32_t visual_offset = (hitbox_width - splitter_width) * 0.5f;

          if (gui::invisible_button("##stream_splitter_hitbox", fan::vec2(hitbox_width, avail_size.y))) {

          }

          if (gui::is_item_active() && gui::is_mouse_dragging(0)) {
            fan::vec2 mouse_delta = gui::get_io().MouseDelta;
            unclamped_settings_width -= mouse_delta.x;
            This->stream_settings.settings_panel_width = std::clamp(unclamped_settings_width, min_settings_width, max_settings_width);
            This->stream_settings.is_resizing = true;
            gui::set_mouse_cursor(gui::mouse_cursor_resize_ew);
          }
          else if (This->stream_settings.is_resizing && !gui::is_mouse_dragging(0)) {
            This->stream_settings.is_resizing = false;
          }

          if (gui::is_item_hovered()) {
            gui::set_mouse_cursor(gui::mouse_cursor_resize_ew);
          }

          fan::vec2 splitter_pos = gui::get_item_rect_min();
          splitter_pos.x += visual_offset;
          fan::vec2 splitter_max = fan::vec2(splitter_pos.x + splitter_width, splitter_pos.y + avail_size.y);

          gui::get_window_draw_list()->AddRectFilled(
            splitter_pos,
            splitter_max,
            gui::is_item_hovered() ? gui::get_color_u32(gui::col_button_hovered) : gui::get_color_u32(gui::col_button)
          );

          gui::pop_style_color(3);

          gui::same_line(0, splitter_spacing);

          gui::begin_child("##stream_settings_panel", fan::vec2(This->stream_settings.settings_panel_width, avail_size.y), true);

          gui::dummy(fan::vec2(12, 0));
          gui::same_line(0, 0);
          gui::begin_group();

          render_stream_settings_content_compact();

          gui::end_group();
          gui::end_child();

        }
        else {
          f32_t side_padding = 12.0f;

          gui::dummy(fan::vec2(side_padding, 0));
          gui::same_line(0, 0);

          gui::set_next_window_bg_alpha(0);
          gui::begin_child("##stream_display", fan::vec2(avail_size.x - (side_padding * 2), avail_size.y), false);

          gui::set_viewport(engine.orthographic_render_view);

          if (ecps_backend.channel_info.empty()) {
            if (auto rt = get_render_thread(); rt) {
              rt->network_frame.set_image(engine.default_texture);
            }
          }

          if (auto rt = get_render_thread(); rt) {
            fan::vec2 stream_area = gui::get_content_region_avail();
            fan::vec2 center = stream_area / 2;
            rt->network_frame.set_position(center);

            f32_t frame_aspect = 16.0f / 9.0f;

            if (rt->screen_decoder.decoded_size.x > 0 && rt->screen_decoder.decoded_size.y > 0) {
              frame_aspect = static_cast<f32_t>(rt->screen_decoder.decoded_size.x) /
                rt->screen_decoder.decoded_size.y;
            }

            fan::vec2 full_size = (stream_area.x / stream_area.y > frame_aspect)
              ? fan::vec2(stream_area.y * frame_aspect, stream_area.y)
              : fan::vec2(stream_area.x, stream_area.x / frame_aspect);

            full_size /= 2;

            if (rt->network_frame.get_image() != engine.default_texture) {
              rt->network_frame.set_size(full_size);
            }

            if (rt->network_frame.get_image() == engine.default_texture) {
              gui::set_cursor_pos(center - fan::vec2(100, 20));
            }
          }

          gui::end_child();
        }
      }
    }

    int main_tab = 0;
    int detail_tab = 0;
    int current_tab = 0;
    bool p_open = true;
    bool auto_refresh = true;
    int refresh_interval = 2; // seconds
    std::string search_filter;
    ecps_backend_t::Protocol_ChannelID_t selected_channel_id{ (uint16_t)-1 };

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
        selected_resolution = 0; // Select the current one
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

      fan::vec2 center = window_content_size / 2;

      auto rt = get_render_thread();
      if (!rt) {
        return;
      }
      rt->network_frame.set_position(center);

      f32_t aspect = 16.0f / 9.0f;

      if (rt->screen_decoder.decoded_size.x > 0 && rt->screen_decoder.decoded_size.y > 0) {
        aspect = static_cast<f32_t>(rt->screen_decoder.decoded_size.x) /
          rt->screen_decoder.decoded_size.y;
      }

      fan::vec2 full_size;

      if (window_content_size.x / window_content_size.y > aspect) {
        full_size = fan::vec2(window_content_size.y * aspect, window_content_size.y);
      }
      else {
        full_size = fan::vec2(window_content_size.x, window_content_size.x / aspect);
      }

      full_size /= 2;

      if (auto rt = get_render_thread(); rt) {
        if (rt->network_frame.get_image() != engine.default_texture) {
          rt->network_frame.set_size(full_size);
        }
      }

      gui::pop_style_var(5);

      gui::push_style_var(gui::style_var_frame_padding, fan::vec2(8, 4));

      std::string button_text = "e";
      fan::vec2 text_size = gui::calc_text_size(button_text.c_str());
      fan::vec2 button_size = fan::vec2(text_size.x + 16, 30);

      gui::set_cursor_pos(fan::vec2(window_content_size.x - button_size.x - 10, 10));

      gui::push_style_color(gui::col_button, fan::vec4(0.3f, 0.3f, 0.3f, 0.8f));
      gui::push_style_color(gui::col_button_hovered, fan::vec4(0.5f, 0.5f, 0.5f, 0.9f));
      gui::push_style_color(gui::col_button_active, fan::vec4(0.6f, 0.6f, 0.6f, 1.0f));

      if (gui::button(button_text.c_str(), button_size)) {
        is_fullscreen_stream = false;
      }

      gui::pop_style_color(3);
      gui::pop_style_var(1);
    }
    gui::end();
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

#undef This
#undef engine
};