
struct settings_menu_t;

typedef void(*page_function_t)(settings_menu_t*);

// functions are defined in graphics.cpp

struct settings_menu_t {
  // pages are divided into two vertically

  static void menu_graphics_left(settings_menu_t* menu);
  static void menu_graphics_right(settings_menu_t* menu);
  static void menu_audio_left(settings_menu_t* menu);
  static void menu_audio_right(settings_menu_t* menu);
  void open();

  void change_target_fps(int direction);
  void render_display_mode();
  void render_target_fps();
  void render_resolution_dropdown();

  void render_separator_with_margin(f32_t width, f32_t margin = 0.f);
  void render_settings_left_column();
  void render_settings_right_column(f32_t min_x);
  void render_settings_top(f32_t min_x);
  void render();
  void reset_page_selection();

  static void set_settings_theme();

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