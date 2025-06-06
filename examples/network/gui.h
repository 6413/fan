inline static fan::graphics::image_t icon_settings;

inline static fan::vec2 popup_size{ 300, 100 };

inline static std::string directory_path = "examples/graphics/screen_share_gui/";

struct ecps_gui_t {

#define This this

  std::vector<std::function<fan::event::task_t()>> task_queue;

#define render_thread (*OFFSETLESS(This, render_thread_t, ecps_gui))
  void backend_queue(const std::function<fan::event::task_t()>& func) {
    render_thread.frame_mutex.lock();
    task_queue.emplace_back(func);
    render_thread.frame_mutex.unlock();
  }
#undef render_thread

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
            if (channel.size()) {
              server_json["channel"] = channel;
            }
            if (ip.size() && port.size()) {
              This->backend_queue([=]() -> fan::event::task_t {
                try {
                  co_await ecps_backend.connect(ip, string_to_number(port));
                  co_await ecps_backend.login();
                  if (channel.size()) {
                    co_await ecps_backend.channel_join(string_to_number(channel));
                  }
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

    gui::animated_popup_window("##stream_settings_overlay", popup_size, start_pos, target_pos, [&] {

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
          //ecps_backend.share.m_NetworkFlow.TimerLastCallAt = EV_nowi(&g_pile->listener);
          ecps_backend.share.m_NetworkFlow.TimerCallCount = 0;
          is_streaming = true;
        }
        gui::pop_style_color();
      }

      gui::set_cursor_pos(right);
      if (gui::image_button("#btn_stream_settings", icon_settings, icon_size)) {
        stream_settings.p_open = true;
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
        if (ecps_backend.channel_info.empty()) {
          gui::text("Joining channel opens more stream settings");
          gui::end();
          return;
        }

        gui::spacing();

        uint32_t channel_id = ecps_backend.channel_info.front().channel_id;

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
              screen_encode.mutex.lock();
              screen_encode.settings.InputFrameRate = framerate;
              screen_encode.update_flags |= fan::graphics::codec_update_e::frame_rate;
              screen_encode.mutex.unlock();
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
                framerate = screen_encode.settings.InputFrameRate;
                bitrate_mbps = screen_encode.settings.RateControl.TBR.bps / 1000000.0;
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
              screen_encode.mutex.lock();
              screen_encode.settings.RateControl.TBR.bps = bitrate_mbps * 1000000.0;
              screen_encode.update_flags |= fan::graphics::codec_update_e::rate_control;
              screen_encode.mutex.unlock();
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
              auto encoders = screen_encode.get_encoders();
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

            gui::push_item_width(-1);
            gui::text("Encoder:");
            selected_encoder = screen_encode.EncoderID;
            if (gui::combo("##encoder", (int*)&selected_encoder, encoder_options.data(),
              encoder_options.size())) {
              screen_encode.mutex.lock();
              screen_encode.new_codec = selected_encoder;
              screen_encode.update_flags |= fan::graphics::codec_update_e::codec;
              screen_encode.mutex.unlock();
            }
            gui::pop_item_width();
          } while (0);//encoder

          gui::next_column();

          do {//decoder
            static auto decoder_names = [] {
              auto decoders = render_thread->screen_decode.get_decoders();
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
            selected_decoder = render_thread->screen_decode.DecoderID;
            if (gui::combo("##decoder", (int*)&selected_decoder, decoder_options.data(),
              decoder_options.size())) {
              screen_encode.mutex.lock();
              render_thread->screen_decode.new_codec = selected_decoder;
              render_thread->screen_decode.update_flags |= fan::graphics::codec_update_e::codec;
              screen_encode.mutex.unlock();
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


#undef This
#define This this

  void render_menu_bar() {
    if (gui::begin_main_menu_bar()) {
      if (gui::begin_menu("Server")) {
        if (gui::menu_item("Create Channel")) {
          drop_down_server.toggle_render_server_create = true;
        }
        if (gui::menu_item("Connect")) {
          drop_down_server.toggle_render_server_connect = true;
        }
        if (gui::menu_item("Join Channel")) {
          drop_down_server.toggle_render_server_join = true;
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

    render_stream();

    if (stream_settings.p_open) {
      stream_settings.render();
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

#undef This
#undef engine
};