#include <string>

#include <fan/imgui/imgui.h>
#include <fan/math/math.h>
import fan;

//TODO make shortcut to open stats menu

using namespace fan::graphics;

fan::graphics::image_t image_stream;
fan::graphics::image_t icon_settings;

fan::vec2 popup_size{ 300, 100 };

std::string directory_path = "examples/graphics/screen_share_gui/";

struct ecps_gui_t {
  engine_t engine;

  ecps_gui_t() {
    icon_settings = engine.image_load(directory_path + "icons/settings.png", {
      .min_filter=fan::graphics::image_filter::linear,
      .mag_filter = fan::graphics::image_filter::linear,
    });
  }

#define engine This->engine

  struct drop_down_server_t {
    #define This OFFSETLESS(this, ecps_gui_t, drop_down_server)

    void render() {
      gui::push_style_var(gui::style_var_window_padding, fan::vec2(13.0000, 20.0000));
      gui::push_style_var(gui::style_var_item_spacing, fan::vec2(14, 16));
      if (toggle_render_server_create) {
        gui::set_next_window_pos(engine.window.get_size() / 2 - popup_size / 2);
        gui::open_popup("server_create");
      }

      if (gui::begin_popup("server_create")) {
        gui::push_item_width(300);
        popup_size = gui::get_window_size();
        gui::text("Server Create");

        static std::string name;
        gui::input_text("name", &name);

        if (gui::button("Create")) {
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

        static std::string ip;
        gui::input_text("ip", &ip);

        static std::string port;
        gui::input_text("port", &port);

        if (gui::button("Create")) {
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
      gui::pop_style_var(2);
    }

    bool toggle_render_server_create = false;
    bool toggle_render_server_connect = false;
  }drop_down_server;

  void render_stream() {
    fan::vec2 viewport_size = gui::get_content_region_avail();
    if (image_stream == engine.default_texture) {
      static fan::graphics::image_t image = engine.image_create(gui::get_color(gui::col_window_bg));
      gui::image(engine.default_texture, viewport_size);
    }
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
        if (gui::button("Start Stream", fan::vec2(button_width - 100.f, button_size.y))) {
          is_streaming = true;
        }
        gui::pop_style_color();
      }

      gui::set_cursor_pos(right);
      if (gui::image_button("#btn_stream_settings", icon_settings, icon_size)) {
        stream_settings.p_open = true;
      }
      // stream_settings.render();
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
        gui::spacing();

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

          gui::text("Framerate:");
          const char* framerate_options[] = {
            "30 fps", "48 fps", "60 fps", "120 fps", "144 fps"
          };
          gui::push_item_width(-1);
          gui::combo("##framerate", &selected_framerate, framerate_options,
            sizeof(framerate_options) / sizeof(framerate_options[0]));
          gui::pop_item_width();

          gui::columns(1);
          gui::spacing();

          gui::text("Bitrate: kbps " + std::to_string(bitrate_kbps));
          gui::push_item_width(-1);
          gui::input_int("##bitrate", &bitrate_kbps, 1000, 10000);
          gui::pop_item_width();
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

          gui::text("Encoder:");
          const char* encoder_options[] = {
            "H.264 (x264)", "NVENC"
          };
          gui::push_item_width(-1);
          gui::combo("##encoder", &selected_encoder, encoder_options,
            sizeof(encoder_options) / sizeof(encoder_options[0]));
          gui::pop_item_width();

          gui::next_column();

          gui::text("Decoder:");
          const char* decoder_options[] = {
            "CUVID", "Software", "Auto"
          };
          gui::push_item_width(-1);
          gui::combo("##decoder", &selected_decoder, decoder_options,
            sizeof(decoder_options) / sizeof(decoder_options[0]));
          gui::pop_item_width();

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
    int selected_framerate = 2; // 60fps
    int bitrate_kbps = 5000;
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
        if (gui::menu_item("Create")) {
          drop_down_server.toggle_render_server_create = true;
        }
        if (gui::menu_item("Connect")) {
          drop_down_server.toggle_render_server_connect = true;
        }
        gui::end_menu();
      }
      gui::end_main_menu_bar();
    }
  }

  void render() {
    ecps_gui.render_menu_bar();
    drop_down_server.render();

    gui::begin("A", 0, gui::window_flags_no_title_bar);
    
    render_stream();

    if (stream_settings.p_open) {
      stream_settings.render();
    }

    gui::end();
  }

  bool is_streaming = false;

#undef This
#undef engine
}ecps_gui;

int main() {

  ecps_gui.engine.loop([&] {
    ecps_gui.render();
  });

  return 0;
}
