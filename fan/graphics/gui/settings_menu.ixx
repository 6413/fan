module;

#if defined(FAN_GUI)

#include <deque>
#include <string>

#endif

export module fan.graphics.gui.settings_menu;

#if defined(FAN_GUI)

import fan.types.vector;
import fan.types.color;
import fan.types.json;
import fan.window;
import fan.graphics.gui.keybinds_menu;

export namespace fan::graphics::gui {
  struct settings_config_t {
    struct display_settings_t {
      int display_mode = 1;
      int target_fps = 165;
      int resolution_index = -1;
      fan::vec2i window_position = -1;
      fan::vec2i custom_resolution = fan::vec2i(-1, -1);
      int renderer = 0;
    };

    struct performance_settings_t {
      bool vsync = false;
      bool show_fps = false;
      bool track_heap = false;
      bool track_opengl_calls = false;
    };

    struct debug_settings_t {
      bool frustum_culling_enabled = true;
      bool visualize_culling = false;
      fan::vec2 culling_padding = 0.0f;
      bool hide_settings_bg = false;
      int fill_mode = 0;
    };

    struct audio_settings_t {
      f32_t volume = 1.0f;
    };

    struct post_processing_t {
      f32_t bloom_strength = 0.0445f;
    };

    void load_from_json(const fan::json& j);
    fan::json to_json() const;

    bool load();
    void save();

    std::string config_save_path = "fan_settings.json";

    display_settings_t display;
    performance_settings_t performance;
    debug_settings_t debug;
    audio_settings_t audio;
    post_processing_t post_processing;
  };

  struct settings_menu_t {
    typedef void(*page_function_t)(settings_menu_t*, const fan::vec2& next_window_position, const fan::vec2& next_window_size);

    fan::graphics::gui::keybind_menu_t keybind_menu;

    inline static bool hide_bg = false;

    settings_menu_t();
    void init_runtime();

    static bool draw_toggle_row(const char* label, const char* id, bool* enabled);

    static void draw_sub_row(
      const char* sublabel,
      auto widget_fn,
      f32_t sublabel_indent = 50.f,
      f32_t subwidget_indent = 20.f
    );

    static void begin_menu_left(const char* name, const fan::vec2& next_window_position, const fan::vec2& next_window_size);
    static void end_menu_left();

    static void menu_graphics_left(settings_menu_t* menu, const fan::vec2& next_window_position, const fan::vec2& next_window_size);
    static void menu_graphics_right(settings_menu_t* menu, const fan::vec2& next_window_position, const fan::vec2& next_window_size);
    static void menu_audio_left(settings_menu_t* menu, const fan::vec2& next_window_position, const fan::vec2& next_window_size);
    static void menu_audio_right(settings_menu_t* menu, const fan::vec2& next_window_position, const fan::vec2& next_window_size);

    void query_current_resolution();
    void on_window_resize(const fan::vec2i& new_size);
    void apply_config(bool construct, bool rest);

    void mark_dirty();
    void update();

    void change_target_fps(int direction);
    void render_display_mode();
    void render_target_fps();
    void render_resolution_dropdown();
    void render_separator_full_width(f32_t y_offset = 0.f);

    void render_settings_left(const fan::vec2& next_window_position, const fan::vec2& next_window_size);
    void render_settings_right(const fan::vec2& next_window_position, const fan::vec2& next_window_size, f32_t min_x);
    fan::vec2 render_settings_top(f32_t min_x);
    void render();

    void reset_page_selection();
    static void set_settings_theme();

    static constexpr int wnd_flags =
      gui::window_flags_no_move |
      gui::window_flags_no_collapse |
      gui::window_flags_no_resize |
      gui::window_flags_no_title_bar;

    static constexpr fan::color title_color = fan::color::from_rgba(0x948c80ff) * 1.5f;
    static constexpr const int fps_values[] = {0, 30, 60, 144, 165, 240};

    struct page_t {
      bool toggle = false;
      std::string name;
      page_function_t render_page_left;
      page_function_t render_page_right;
      f32_t split_ratio = 0.5f;
    };

    settings_config_t config;
    int current_page = 0;
    int current_resolution = 0;
    f32_t min_x = 40.f;
    std::deque<page_t> pages;
    bool is_dirty = false;
    fan::time::timer save_timer;
    inline static constexpr int64_t save_delay_ms = 1000;
    fan::window_t::resize_handle_t resize_handle;
    fan::window_t::move_handle_t move_handle;
  };
}

#endif