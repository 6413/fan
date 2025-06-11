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

    icon_settings = engine.image_load(directory_path + "icons/settings.png", {
     .min_filter = fan::graphics::image_filter::linear,
     .mag_filter = fan::graphics::image_filter::linear,
    });

    if (fan::io::file::exists(config_path)) {
      std::string data;
      fan::io::file::read(config_path, &data);
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
              co_await ecps_backend.channel_join(channel_id);
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
                catch (...) { }
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
              catch (...) { }
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

    gui::animated_popup_window("##stream_settings_overlay", popup_size, start_pos, target_pos, trigger_popup, [&] {

      fan::vec2 icon_size = 48.f;

      fan::vec2 cursor_pos = gui::get_cursor_pos();

      fan::vec2 button_padding = gui::get_style().FramePadding;
      fan::vec2 button_size = icon_size + button_padding * 2;

      fan::vec2 left = fan::vec2(
        button_padding.x,
        (popup_size.y - button_size.y) / 2
      );
      fan::vec2 center = fan::vec2(
        (popup_size.x - button_size.x) / 2,
        (popup_size.y - button_size.y) / 2
      );
      fan::vec2 right = fan::vec2(
        popup_size.x - button_size.x - button_padding.x,
        (popup_size.y - button_size.y) / 2
      );
      float button_width = 200;
      gui::set_cursor_pos(center - fan::vec2(button_width / 2, 0));

      if (is_streaming) {
        gui::push_style_color(gui::col_button, fan::vec4(0.8f, 0.2f, 0.2f, 1.0f));
        if (gui::button("Stop Stream", fan::vec2(button_width, button_size.y))) {
          is_streaming = false;
        }
        gui::pop_style_color();
      }
      else {////
        gui::push_style_color(gui::col_button, fan::vec4(0.2f, 0.8f, 0.2f, 1.0f));
        if (gui::button("Start Stream", fan::vec2(button_width, button_size.y))) {
          ecps_backend.share.m_NetworkFlow.Bucket = 0;
          ecps_backend.share.m_NetworkFlow.TimerLastCallAt = fan::event::now();
          ecps_backend.share.m_NetworkFlow.TimerCallCount = 0;
          ecps_backend.share.CalculateNetworkFlowBucket();
          is_streaming = true;
        }
        gui::pop_style_color();
      }

      gui::set_cursor_pos(right);
      if (gui::image_button("#btn_stream_settings", icon_settings, icon_size)) {
        stream_settings.p_open = !stream_settings.p_open;
      }
      });
    gui::pop_style_var(2);
  }

  struct stream_settings_t {
#undef This
#define This OFFSETLESS(this, ecps_gui_t, stream_settings)

    void render() {
      fan::vec2 window_size = engine.window.get_size() / 2;
      fan::vec2 window_pos = engine.window.get_size() / 2 - window_size / 2;
      window_size.y += window_size.y / 2;
      window_pos.y -= window_size.y / 4.5;
      gui::set_next_window_pos(window_pos, gui::cond_once);
      gui::set_next_window_size(window_size);
      if (gui::begin("##stream_settings_root", &p_open, gui::window_flags_no_collapse)) {
        /*if (ecps_backend.channel_info.empty()) {
          gui::text("Joining channel opens more stream settings");
          gui::end();
          return;
        }*/

        gui::spacing();

        uint32_t channel_id = ecps_backend.channel_info.size() > 0 ? 0 : -1;

        gui::push_style_color(gui::col_child_bg, gui::get_color(gui::col_frame_bg));
        gui::begin_child("##video_settings", fan::vec2(0, 220), true);
        {
          gui::text("Video Settings");
          gui::separator();
          gui::spacing();

          gui::columns(2, "video_cols", false);

          gui::text("Resolution:");
          const char* resolution_options[] = {
            "1920x1080", "1680x1050", "1600x900", "1440x900",
            "1366x768", "1280x720", "1024x768", "800x600"
          };
          gui::push_item_width(-1);
          gui::combo("##resolution", &selected_resolution, resolution_options,
            sizeof(resolution_options) / sizeof(resolution_options[0]));
          gui::pop_item_width();

          gui::next_column();

          gui::push_item_width(-1);
          gui::text("Framerate: " + std::to_string(framerate));

          gui::push_item_width(-1);
          do {
            gui::input_float("##framerate", &framerate, 10, 10);
            if (gui::is_item_deactivated_after_edit()) {
              if (channel_id == (uint32_t)-1) {
                break;
              }
              /*Protocol_ChannelID_t ChannelID;
              ChannelID.g() = channel_id;
              auto ChannelCommon = ChannelMap_GetOutputPointerSafe(&g_pile->ChannelMap, &ChannelID);
              if (ChannelCommon->GetState() != ChannelState_t::ScreenShare_Share) {
                break;
              }*/
              screen_encode->mutex.lock();
              screen_encode->settings.InputFrameRate = framerate;
              screen_encode->update_flags |= fan::graphics::codec_update_e::frame_rate;
              screen_encode->mutex.unlock();
            }
          } while (0);
          gui::pop_item_width();


          gui::columns(1);
          gui::spacing();

          do {
            //initialize
            static int initialize = 1;
            do {
              if (channel_id != (uint32_t)-1 && initialize) {
               /* Protocol_ChannelID_t ChannelID;
                ChannelID.g() = channel_id;
                auto ChannelCommon = ChannelMap_GetOutputPointerSafe(&g_pile->ChannelMap, &ChannelID);
                if (ChannelCommon == nullptr || ChannelCommon->GetState() != ChannelState_t::ScreenShare_Share) {
                  break;
                }*/

                //auto sd = (Channel_ScreenShare_Share_t*)ChannelCommon->m_StateData;
                //sd->ThreadCommon->EncoderSetting.Mutex.Lock(); // do i need to lock for reading
                framerate = screen_encode->settings.InputFrameRate;
                bitrate_mbps = screen_encode->settings.RateControl.VBR.bps / 1000000.0;
                //sd->ThreadCommon->EncoderSetting.Mutex.Unlock();
                initialize = false;
              }
            } while (0);

            gui::text("Bitrate: mbps " + std::to_string(bitrate_mbps));
            gui::push_item_width(-1);
            gui::input_float("##bitrate", &bitrate_mbps, 1, 100);
            if (gui::is_item_deactivated_after_edit()) {
              if (channel_id == (uint32_t)-1) {
                break;
              }
             /* Protocol_ChannelID_t ChannelID;
              ChannelID.g() = channel_id;
              auto ChannelCommon = ChannelMap_GetOutputPointerSafe(&g_pile->ChannelMap, &ChannelID);
              if (ChannelCommon->GetState() != ChannelState_t::ScreenShare_Share) {
                break;
              }*/

              //auto sd = (Channel_ScreenShare_Share_t*)ChannelCommon->m_StateData;
              screen_encode->mutex.lock();
              screen_encode->settings.RateControl.VBR.bps = bitrate_mbps * 1000000.0;
              screen_encode->update_flags |= fan::graphics::codec_update_e::rate_control;
              screen_encode->mutex.unlock();
            }
            gui::pop_item_width();
          } while (0);
        }
        gui::end_child();
        gui::pop_style_color();

        gui::spacing();

        gui::push_style_color(gui::col_child_bg, gui::get_color(gui::col_frame_bg));
        gui::begin_child("##codec_settings", fan::vec2(0, 120), true);
        {
          gui::text("Codec Settings");
          gui::separator();
          gui::spacing();

          gui::columns(2, "codec_cols", false);

          do {//encoder
            static auto encoder_names = [] {
              auto encoders = screen_encode->get_encoders();
              std::array<std::string, encoders.size()> encoder_names;
              for (std::size_t i = 0; i < encoders.size(); ++i) {
                encoder_names[i] = encoders[i].Name;
              }
              return encoder_names;
            }();

            //pain
            static auto encoder_options = [] {
              std::array<const char*, encoder_names.size()> names;
              for (size_t i = 0; i < encoder_names.size(); ++i) {
                names[i] = encoder_names[i].c_str();
              }
              return names;
            }();

            if (!This->is_streaming) {
              gui::columns(1, "codec_cols", false);
              break;
            }

            gui::push_item_width(-1);
            gui::text("Encoder:");
            selected_encoder = screen_encode->EncoderID;
            if (gui::combo("##encoder", (int*)&selected_encoder, encoder_options.data(),
              encoder_options.size())) {
              screen_encode->mutex.lock();
              screen_encode->new_codec = selected_encoder;
              screen_encode->update_flags |= fan::graphics::codec_update_e::codec;
              screen_encode->encode_write_flags |= fan::graphics::codec_update_e::reset_IDR;
              screen_encode->mutex.unlock();
            }
            gui::pop_item_width();
          } while (0);//encoder

          gui::next_column();

          do {//decoder
            static auto decoder_names = [] {
              auto decoders = screen_decode->get_decoders();
              std::array<std::string, decoders.size()> decoder_names;
              for (std::size_t i = 0; i < decoder_names.size(); ++i) {
                decoder_names[i] = decoders[i].Name;
              }
              return decoder_names;
            }();

            //pain
            static auto decoder_options = [] {
              std::array<const char*, decoder_names.size()> names;
              for (size_t i = 0; i < decoder_names.size(); ++i) {
                names[i] = decoder_names[i].c_str();
              }
              return names;
            }();

            gui::push_item_width(-1);
            gui::text("Decoder:");
            selected_decoder = screen_decode->DecoderID;
            if (gui::combo("##decoder", (int*)&selected_decoder, decoder_options.data(),
              decoder_options.size())) {
              screen_decode->mutex.lock();
              screen_decode->new_codec = selected_decoder;
              screen_decode->update_flags |= fan::graphics::codec_update_e::codec;
              screen_decode->mutex.unlock();
              This->backend_queue([=]() -> fan::event::task_t {
                try {
                  ecps_backend_t::Protocol_C2S_t::Channel_ScreenShare_ViewToShare_t rest;
                  rest.ChannelID = ecps_backend.channel_info.front().channel_id;
                  rest.Flag = ecps_backend_t::ProtocolChannel::ScreenShare::ChannelFlag::ResetIDR;
                  int idr_request_count = 1;
                  co_await ecps_backend.tcp_write(
                    ecps_backend_t::Protocol_C2S_t::Channel_ScreenShare_ViewToShare,
                    &rest,
                    sizeof(rest)
                  );
                }
                catch (...) {}
                });
            }
            gui::pop_item_width();
          } while (0);//decoder

          gui::columns(1);
        }
        gui::end_child();
        gui::pop_style_color();

        gui::spacing();

        gui::push_style_color(gui::col_child_bg, gui::get_color(gui::col_frame_bg));
        gui::begin_child("##input_settings", fan::vec2(0, 120), true);
        {
          gui::text("Input Control");
          gui::separator();
          gui::spacing();

          const char* input_control_options[] = {
            "None", "Keyboard Only", "Keyboard + Mouse"
          };
          gui::push_item_width(-1);
          gui::combo("##input_control", &input_control, input_control_options,
            sizeof(input_control_options) / sizeof(input_control_options[0]));
          gui::pop_item_width();
        }
        gui::end_child();
        gui::pop_style_color();
      }
      gui::end();
    }

    int selected_resolution = 0;
    f32_t framerate = 30;
    f32_t bitrate_mbps = 5;
    int selected_encoder = 0;
    int selected_decoder = 0;
    int input_control = 0;

    bool p_open = false;
  }stream_settings;

  struct channel_list_window_t {
#undef This
#define This OFFSETLESS(this, ecps_gui_t, channel_list_window)

    void render() {
      if (!p_open) return;

      fan::vec2 window_size = engine.window.get_size() * 0.8f;
      fan::vec2 window_pos = engine.window.get_size() / 2 - window_size / 2;

      gui::set_next_window_pos(window_pos, gui::cond_first_use_ever);
      gui::set_next_window_size(window_size, gui::cond_first_use_ever);

      if (gui::begin("Channel Browser", &p_open)) {

        // Display current username in top right
        std::string current_user = "User: " + ecps_backend.get_current_username(); // Assuming this method exists
        fan::vec2 text_size = gui::calc_text_size(current_user.c_str());
        gui::same_line(gui::get_window_size().x - text_size.x - 20);
        gui::text(current_user.c_str(), fan::vec4(0.7f, 0.7f, 1.0f, 1.0f)); // Light blue color

        static bool first_open = true;
        if (first_open && p_open) {
          first_open = false;
          This->backend_queue([=]() -> fan::event::task_t {
            try {
              co_await ecps_backend.request_channel_list();

              // Also refresh user list if a channel is already selected
              if (selected_channel_id.i != (uint16_t)-1) {
                co_await ecps_backend.request_channel_session_list(selected_channel_id);
              }
            }
            catch (...) {
              fan::print("Failed to auto-refresh channel list");
            }
            });
        }

        // Reset first_open flag when window is closed
        if (!p_open) {
          first_open = true;
        }

        gui::columns(2, "channel_browser_cols", true);

        gui::text("Channels");
        gui::separator();

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

        if (auto_refresh) {
          static auto last_refresh = std::chrono::steady_clock::now();
          auto now = std::chrono::steady_clock::now();
          if (std::chrono::duration_cast<std::chrono::seconds>(now - last_refresh).count() >= refresh_interval) {
            This->backend_queue([=]() -> fan::event::task_t {
              try {
                co_await ecps_backend.request_channel_list();
              }
              catch (...) {}
              });
            last_refresh = now;
          }
        }

        gui::push_item_width(150);
        gui::input_text("Search", &search_filter);
        gui::pop_item_width();

        gui::separator();

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
              std::string display_name = channel.name;
              if (channel.is_password_protected) {
                display_name += " 🔒";
              }
              display_name += " (" + std::to_string(channel.user_count) + " users)";

              // Check if already joined
              bool already_joined = false;
              for (const auto& joined_channel : ecps_backend.channel_info) {
                if (joined_channel.channel_id.i == channel.channel_id.i) {
                  already_joined = true;
                  break;
                }
              }

              if (already_joined) {
                display_name += " ✓";
              }

              gui::table_next_row();
              gui::table_set_column_index(0);

              // Set active color to be the same as hover color
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

          // No channels
          if (ecps_backend.available_channels.empty()) {
            if (ecps_backend.channel_list_received) {
              gui::text("No channels available.");
              gui::text("Create a new channel to get started!");
            }
            else {
              gui::text("Click 'Refresh Channel List' to load channels");
            }
          }
        }
        gui::end_child();

        gui::spacing();

        bool has_selection = (selected_channel_id.i != (uint16_t)-1);
        bool already_in_selected = false;
        bool is_host_of_selected = false;

        if (has_selection) {
          // Check if already joined and if user is host
          for (const auto& joined_channel : ecps_backend.channel_info) {
            if (joined_channel.channel_id.i == selected_channel_id.i) {
              already_in_selected = true;

              // Check if current user is host of this channel
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

        if (!is_host_of_selected) {
          if (already_in_selected) {
            // Leave button
            gui::push_style_color(gui::col_button, fan::vec4(0.8f, 0.3f, 0.3f, 1.0f) / 1.1);
            std::string leave_text = "Leave";
            fan::vec2 text_size = gui::calc_text_size(leave_text.c_str());
            float button_width = text_size.x + gui::get_style().FramePadding.x * 2 + 20;

            if (gui::button(leave_text.c_str(), fan::vec2(button_width, 0))) {
              This->backend_queue([channel_id = selected_channel_id]() -> fan::event::task_t {
                try {
                  // co_await ecps_backend.channel_leave(channel_id);
                  fan::print("TODO channel_leave");
                  co_return;
                }
                catch (...) {
                  fan::print("Failed to leave channel");
                }
                });
            }
            gui::pop_style_color();
          }
          else {
            // Join button
            gui::push_style_color(gui::col_button, fan::vec4(0.3f, 0.8f, 0.3f, 1.0f) / 1.1);
            std::string text = "Join";
            fan::vec2 text_size = gui::calc_text_size(text);
            float button_width = text_size.x + gui::get_style().FramePadding.x * 2 + 20;

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
          }
          gui::same_line();
        }
        else if (has_selection) {
          gui::text("You are the host of this channel", fan::vec4(1.0f, 0.8f, 0.2f, 1.0f)); // Gold color
          gui::same_line();
        }

        {
          gui::push_style_color(gui::col_button, fan::vec4(0.3f, 0.3f, 0.3f, 1.0f));
          {
            std::string text = "Add new";
            fan::vec2 text_size = gui::calc_text_size(text.c_str());
            float button_width = text_size.x + gui::get_style().FramePadding.x * 2 + 20;
            if (gui::button(text, fan::vec2(button_width, 0))) {
              This->backend_queue([]() -> fan::event::task_t {
                try {
                  auto channel_id = co_await ecps_backend.channel_create();
                  co_await ecps_backend.request_channel_list();
                  co_await ecps_backend.request_channel_session_list(channel_id);
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
            float button_width = text_size.x + gui::get_style().FramePadding.x * 2 + 20;
            if (gui::button(text, fan::vec2(button_width, 0))) {
              This->drop_down_server.toggle_render_server_connect = true;
            }
          }
          gui::pop_style_color();
        }

        gui::next_column();

        gui::text("Users in Channel");
        gui::separator();

        if (selected_channel_id.i != (uint16_t)-1) {
          std::string channel_name = "Channel " + std::to_string(selected_channel_id.i);
          for (const auto& channel : ecps_backend.available_channels) {
            if (channel.channel_id.i == selected_channel_id.i) {
              channel_name = channel.name;
              break;
            }
          }

          gui::text(("Channel: " + channel_name).c_str());

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

          gui::separator();

          if (gui::begin_child("SessionList", fan::vec2(0, 0), true)) {
            auto it = ecps_backend.channel_sessions.find(selected_channel_id.i);
            if (it != ecps_backend.channel_sessions.end()) {
              for (const auto& session : it->second) {
                std::string user_display = session.username;
                fan::color user_color = fan::colors::white;
                if (session.is_host) {
                  user_display += " (Host)";
                  user_color = fan::vec4(1.0f, 0.8f, 0.2f, 1.0f); // Gold color for host
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
        }
        else {
          gui::text("Select a channel to view users");
          gui::separator();
          gui::text("Click on any channel in the left panel");
          gui::text("to see who's currently connected.");
        }

        gui::columns(1);
      }
      gui::end();
    }

    bool p_open = true;
    bool auto_refresh = true;
    int refresh_interval = 5; // seconds
    std::string search_filter;
    ecps_backend_t::Protocol_ChannelID_t selected_channel_id{ (uint16_t)-1 };

  }channel_list_window;


#undef This
#define This this

  void render_menu_bar() {
    if (gui::begin_main_menu_bar()) {
      if (gui::begin_menu("Channel")) {
        if (gui::menu_item("Browse all")) {
          channel_list_window.p_open = true;
        }
        gui::end_menu();
      }
      gui::end_main_menu_bar();
    }
  }

  void render() {
    render_menu_bar();
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

    fan::vec2 viewport_size = engine.viewport_get(engine.orthographic_camera.viewport).viewport_size;
    engine.camera_set_ortho(
      engine.orthographic_camera.camera,
      fan::vec2(0, viewport_size.x),
      fan::vec2(0, viewport_size.y)
    );

    { // Background
      gui_background.set_position(fan::vec2(engine.window.get_size() / 2));
      gui_background.set_size(engine.window.get_size() / 2 + fan::vec2(0, 1));
    }

    f32_t top_bar = gui::get_frame_height();
    fan::vec2 win = engine.window.get_size();
    fan::vec2 avail = win / 2 - fan::vec2(0, top_bar / 2);

    f32_t aspect = 16.0f / 9.0f;
    fan::vec2 full_size = (avail.x / avail.y > aspect)
      ? fan::vec2(avail.y * aspect, avail.y)
      : fan::vec2(avail.x, avail.x / aspect);

    render_thread->screen_frame.set_position(win / 2 + fan::vec2(0, top_bar / 2));
    if (render_thread->screen_frame.get_image() == gloco->default_texture) {      
      render_thread->screen_frame.set_size(0);
    }
    else {
      render_thread->screen_frame.set_size(full_size);
    }
    if (show_own_stream && render_thread->local_frame.get_image() != gloco->default_texture) {
      render_thread->local_frame.set_size(full_size);
    }
    else {
      render_thread->local_frame.set_size(0);
    }

    render_stream();

    if (stream_settings.p_open) {
      stream_settings.render();
    }

    if (channel_list_window.p_open) {
      channel_list_window.render();
    }

    gui::end();
  }

  inline static const char* config_path = "config.json";
  void write_to_config(const std::string& key, auto value) {
    config[key] = value;
    fan::io::file::write(config_path, config.dump(2), std::ios_base::binary);
  }

  fan::json config;
  bool is_streaming = false;

  bool show_own_stream = false; 

  fan::graphics::sprite_t gui_background{ {
    .position = fan::vec3(gloco->window.get_size() / 2, 0),
    .size = gloco->window.get_size() / 2,
    .image = engine.image_create(gui::get_color(gui::col_window_bg).set_alpha(1))
  } };

#undef This
#undef engine
};