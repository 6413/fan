module;

#include <fan/utility.h>
#include <fan/event/types.h>

#include <string>
#include <ios>
#include <functional>
#include <filesystem>
#include <queue>
#include <cmath>
#include <algorithm>

export module fan.graphics.gui;

#if defined(FAN_GUI)
  import fan.print;
  import fan.utility;
  import fan.math;
  import fan.types.vector;
  import fan.types.color;
  import fan.types.quaternion;
  import fan.types.fstring;
  import fan.types.json;
  import fan.utility;
  import fan.event;
  import fan.fmt;

  import fan.graphics.common_context;
  import fan.graphics;
  import fan.io.file;
  import fan.io.directory;
  import fan.audio;

  import fan.file_dialog;

  export import fan.graphics.gui.base;
#endif


#if defined(FAN_GUI)

#if defined(FAN_GUI)
export namespace fan::graphics::gui {

  /// <summary>
  /// RAII containers for gui windows.
  /// </summary>
  struct window_t {
    window_t(const std::string& window_name, bool* p_open = 0, window_flags_t window_flags = 0);
    ~window_t();
    explicit operator bool() const;

  private:
    bool is_open;
  };
  /// <summary>
  /// RAII containers for gui child windows.
  /// </summary>
  struct child_window_t {
    child_window_t(const std::string& window_name, const fan::vec2& size = fan::vec2(0, 0), child_window_flags_t window_flags = 0);
    ~child_window_t();
    explicit operator bool() const;

  private:
    bool is_open;
  };

  /// <summary>
  /// RAII containers for gui tables.
  /// </summary>
  struct table_t {
    table_t(const std::string& str_id, int columns, table_flags_t flags = 0, const fan::vec2& outer_size = fan::vec2(0.0f, 0.0f), f32_t inner_width = 0.0f);
    ~table_t();
    explicit operator bool() const;

  private:
    bool is_open;
  };

#if !defined(__INTELLISENSE__)
#define fan_imgui_dragfloat_named(name, variable, speed, m_min, m_max) \
    fan::graphics::gui::drag_float(name, &variable, speed, m_min, m_max)
#endif

#define fan_imgui_dragfloat(variable, speed, m_min, m_max) \
      fan_imgui_dragfloat_named(STRINGIFY(variable), variable, speed, m_min, m_max)


#define fan_imgui_dragfloat1(variable, speed) \
      fan_imgui_dragfloat_named(STRINGIFY(variable), variable, speed, 0, 0)

  using gui_draw_cb_nr_t = fan::graphics::gui_draw_cb_nr_t;

  struct imgui_element_nr_t : gui_draw_cb_nr_t {
    using base_t = gui_draw_cb_nr_t;

    imgui_element_nr_t() = default;

    imgui_element_nr_t(const imgui_element_nr_t& nr);

    imgui_element_nr_t(imgui_element_nr_t&& nr);

    ~imgui_element_nr_t();

    fan::graphics::gui::imgui_element_nr_t& operator=(const imgui_element_nr_t& id);

    fan::graphics::gui::imgui_element_nr_t& operator=(imgui_element_nr_t&& id);
    void init();

    bool is_invalid() const;

    void invalidate_soft();

    void invalidate();

    void set(const std::function<void()>& lambda);
  };

  struct imgui_element_t : imgui_element_nr_t {
    imgui_element_t() = default;
    imgui_element_t(const std::function<void()>& lambda);
  };

  const char* item_getter1(const std::vector<std::string>& items, int index);
  void set_viewport(fan::graphics::viewport_t viewport);
  void set_viewport(const fan::graphics::render_view_t& render_view = fan::graphics::get_orthographic_render_view());

  void image(fan::graphics::image_t img, const fan::vec2& size, const fan::vec2& uv0 = fan::vec2(0, 0), const fan::vec2& uv1 = fan::vec2(1, 1), const fan::color& tint_col = fan::color(1, 1, 1, 1), const fan::color& border_col = fan::color(0, 0, 0, 0));
  bool image_button(const std::string& str_id, fan::graphics::image_t img, const fan::vec2& size, const fan::vec2& uv0 = fan::vec2(0, 0), const fan::vec2& uv1 = fan::vec2(1, 1), int frame_padding = -1, const fan::color& bg_col = fan::color(0, 0, 0, 0), const fan::color& tint_col = fan::color(1, 1, 1, 1));
  bool image_text_button(
    fan::graphics::image_t img,
    const std::string& text,
    const fan::color& color,
    const fan::vec2& size,
    const fan::vec2& uv0 = fan::vec2(0, 0),
    const fan::vec2& uv1 = fan::vec2(1, 1),
    int frame_padding = -1,
    const fan::color& bg_col = fan::color(0, 0, 0, 0),
    const fan::color& tint_col = fan::color(1, 1, 1, 1)
  );

  
  bool toggle_image_button(const std::string& char_id, fan::graphics::image_t image, const fan::vec2& size, bool* toggle);

  template <std::size_t N>
  bool toggle_image_button(const std::array<fan::graphics::image_t, N>& images, const fan::vec2& size, int* selectedIndex)
  {
    f32_t y_pos = get_cursor_pos_y() + get_style().WindowPadding.y - get_style().FramePadding.y / 2;

    bool clicked = false;
    bool pushed = false;

    for (std::size_t i = 0; i < images.size(); ++i) {
      fan::color tintColor = fan::color(0.2, 0.2, 0.2, 1.0);
      if (*selectedIndex == i) {
        tintColor = fan::color(0.2, 0.2, 0.2, 1.0f);
        push_style_color(col_button, tintColor);
        pushed = true;
      }
      /*if (ImGui::IsItemHovered()) {
      tintColor = fan::color(1, 1, 1, 1.0f);
      }*/
      set_cursor_pos_y(y_pos);

      if (fan::graphics::gui::image_button("##toggle_image_button" + std::to_string(i) + std::to_string((uint64_t)&clicked), images[i], size)) {
        *selectedIndex = i;
        clicked = true;
      }
      if (pushed) {
        pop_style_color();
        pushed = false;
      }
      same_line();
    }

    return clicked;
  }

  // untested
  void image_rotated(
    fan::graphics::image_t image,
    const fan::vec2& size,
    int angle,
    const fan::vec2& uv0 = fan::vec2(0, 0),
    const fan::vec2& uv1 = fan::vec2(1, 1),
    const fan::color& tint_col = fan::color(1, 1, 1, 1),
    const fan::color& border_col = fan::color(0, 0, 0, 0)
  );
#endif // fan graphics gui
} // namespace fan::graphics::gui

export namespace fan::graphics::gui {
  struct imgui_fs_var_t {
    fan::graphics::gui::imgui_element_t ie;

    imgui_fs_var_t() = default;

    template <typename T>
    imgui_fs_var_t(
      fan::graphics::shader_t shader_nr,
      const std::string& var_name,
      T initial_ = 0,
      f32_t speed = 1,
      f32_t min = -100000,
      f32_t max = 100000
    );
  };

#if defined(FAN_2D)
  void shape_properties(const fan::graphics::shapes::shape_t& shape);
#endif
} // namespace fan::graphics::gui

export namespace fan::graphics::gui {
  struct content_browser_t {
    struct file_info_t {
      std::string filename;
      std::wstring item_path;
      bool is_directory;
      fan::graphics::image_t preview_image;
      bool is_selected = false;
      //std::string 
    };

    struct selection_state_t {
      bool is_selecting = false;
      fan::vec2 selection_start;
      fan::vec2 selection_end;
      std::vector<size_t> selected_indices;
      bool ctrl_held = false;
    } selection_state;


    std::vector<file_info_t> directory_cache;

    fan::graphics::image_t icon_arrow_left = fan::graphics::image_load("images/content_browser/arrow_left.webp");
    fan::graphics::image_t icon_arrow_right = fan::graphics::image_load("images/content_browser/arrow_right.webp");

    fan::graphics::image_t icon_file = fan::graphics::image_load("images/content_browser/file.webp");
    fan::graphics::image_t icon_object = fan::graphics::image_load("images/content_browser/object.webp");
    fan::graphics::image_t icon_directory = fan::graphics::image_load("images/content_browser/folder.webp");

    fan::graphics::image_t icon_files_list = fan::graphics::image_load("images/content_browser/files_list.webp");
    fan::graphics::image_t icon_files_big_thumbnail = fan::graphics::image_load("images/content_browser/files_big_thumbnail.webp");

    bool item_right_clicked = false;
    std::string item_right_clicked_name;

    std::wstring asset_path = L"./";

    inline static fan::io::async_directory_iterator_t directory_iterator;

    std::filesystem::path current_directory;
    enum viewmode_e {
      view_mode_list,
      view_mode_large_thumbnails,
    };
    viewmode_e current_view_mode = view_mode_list;
    f32_t thumbnail_size = 128.0f;
    f32_t padding = 16.0f;
    std::string search_buffer;

    // lambda [this] capture
    content_browser_t(const content_browser_t&) = delete;
    content_browser_t(content_browser_t&&) = delete;

    content_browser_t();
    content_browser_t(bool no_init);
    content_browser_t(const std::wstring& path);
    void init(const std::wstring& path);

    void clear_selection();

    bool is_point_in_rect(const fan::vec2& point, const fan::vec2& rect_min, const fan::vec2& rect_max);

    void handle_rectangular_selection();

    void update_directory_cache();
    struct search_state_t {
      std::string query;
      bool is_recursive = false;
      bool is_searching = false;
      std::vector<file_info_t> found_files;
      std::queue<std::string> pending_directories;
      std::vector<std::pair<file_info_t, size_t>> sorted_cache;
      std::vector<std::pair<file_info_t, size_t>> sorted_search_cache;
      bool cache_dirty = true;
      bool search_cache_dirty = true;
    };

    search_state_t search_state;
    fan::io::async_directory_iterator_t search_iterator;

    void invalidate_cache();

    int get_pressed_key();

    void handle_keyboard_navigation(const std::string& filename, int pressed_key);

    void handle_right_click(const std::string& filename);

    void process_next_directory();

    void start_search(const std::string& query, bool recursive = false);
    void update_sorted_cache();

    void update_search_sorted_cache();

    void render();

    void render_large_thumbnails_view();


    void render_list_view();
    void handle_item_interaction(const file_info_t& file_info, size_t original_index);

    // [](const std::filesystem::path& path) {}
    void receive_drag_drop_target(std::function<void(const std::filesystem::path& fs)> receive_func);
  };

#if defined(FAN_2D)

  struct sprite_animations_t {
    //fan::vec2i frame_coords; // starting from top left and increasing by one to get the next frame into that direction

    fan::graphics::animation_nr_t current_animation_shape_nr;
    fan::graphics::animation_nr_t current_animation_nr;
    std::string animation_list_name_to_edit; // animation rename
    std::string animation_list_name_edit_buffer;

    bool adding_sprite_sheet = false;
    int hframes = 1;
    int vframes = 1;
    std::string sprite_sheet_drag_drop_name;

    std::vector<int> previous_hold_selected;

    bool set_focus = false;
    bool play_animation = false;
    bool toggle_play_animation = false;
    static inline constexpr f32_t animation_names_padding = 10.f;

    bool render_list_box(fan::graphics::animation_nr_t& shape_animation_id);

    bool render_selectable_frames(fan::graphics::sprite_sheet_animation_t& current_animation);
    bool render(const std::string& drag_drop_id, fan::graphics::animation_nr_t& shape_animation_id);
  };

  void fragment_shader_editor(uint16_t shape_type, std::string* fragment, bool* shader_compiled);
#endif

#if defined(FAN_2D)
  struct particle_editor_t {

    particle_editor_t();

    fan::graphics::shapes::particles_t::ri_t& get_ri();

    void handle_file_operations();

    void render_menu();
    void render_settings();
    void render();

    void fout(const std::string& filename);

  private:
    fan::graphics::shapes::shape_t particle_shape = fan::graphics::shapes::particles_t::properties_t{
      .position = fan::vec3(32.108f, -1303.084f, 10.0f),
      .start_size = fan::vec2(28.638f),
      .end_size = fan::vec2(28.638f),
      .begin_color = fan::color::from_rgba(0x33333369),
      .end_color = fan::color::from_rgba(0x33333369),
      .alive_time = 1768368768,
      .count = 1191,
      .start_velocity = fan::vec2(0.0f, 9104.127f),
      .end_velocity = fan::vec2(0.0f, 9104.127f),
      .begin_angle = 0,
      .end_angle = -0.16f,
      .angle = fan::vec3(0.0f, 0.0f, -0.494f),
      .spawn_spacing = fan::vec2(400.899f, 1.0f),
      .start_spread = fan::vec2(2648.021f, 1.0f),
      .end_spread = fan::vec2(2648.021f, 1.0f),
      .jitter_start = fan::vec2(0.0f),
      .jitter_end = fan::vec2(0.0f),
      .jitter_speed = 0.0f,
      .shape = fan::graphics::shapes::particles_t::shapes_e::rectangle,
      .image = fan::graphics::image_load("images/waterdrop.webp")
    };

  public:

    void set_particle_shape(fan::graphics::shape_t&& shape);

    // just for gui visualization
    fan::graphics::sprite_t particle_image_sprite;

    /*fan::color bg_color = fan::color::from_rgba(0xB8C4BFFF);*/
    fan::color bg_color = fan::colors::black;
    fan::color base_color = fan::color::from_rgba(0x33333369);
    f32_t color_intensity = 1.0f;
    fan::graphics::file_save_dialog_t save_file_dialog{};
    fan::graphics::file_open_dialog_t open_file_dialog{};
    std::string filename{};
  };

#endif


  struct dialogue_box_t {

    struct render_type_t;

  #include <fan/fan_bll_preset.h>
  #define BLL_set_prefix drawables
  #define BLL_set_type_node uint16_t
  #define BLL_set_NodeDataType render_type_t*
  #define BLL_set_Link 1
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_SafeNext 1
  #include <BLL/BLL.h>
    using drawable_nr_t = drawables_NodeReference_t;

    drawables_t drawables;

    struct render_type_t {
      virtual void render(dialogue_box_t* This, drawable_nr_t nr, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing) = 0;
      virtual ~render_type_t();
    };

    struct text_delayed_t : render_type_t {
      ~text_delayed_t() override;

      void render(dialogue_box_t* This, drawable_nr_t nr, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing) override;
      bool dialogue_line_finished = false; // for skipping
      std::string text;
      uint64_t character_per_s = 20;
      std::size_t render_pos = 0;
      fan::time::timer blink_timer{ (uint64_t)0.5e9, true };
      uint8_t render_cursor = false;
      fan::event::task_t character_advance_task;
    };
    struct text_t : render_type_t {
      std::string text;
      void render(dialogue_box_t* This, drawable_nr_t nr, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing) override;
    };

    struct button_t : render_type_t {
      fan::vec2 position = -1;
      fan::vec2 size = 0;
      std::string text;
      void render(dialogue_box_t* This, drawable_nr_t nr, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing) override;
    };

    struct separator_t : render_type_t {
      void render(dialogue_box_t* This, drawable_nr_t nr, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing) override;
    };

    dialogue_box_t();

    // 0-1
    void set_cursor_position(const fan::vec2& pos);
    void set_indent(f32_t indent);

    fan::event::task_value_resume_t<drawable_nr_t> text_delayed(const std::string& character_name, const std::string& text);
    fan::event::task_value_resume_t<drawable_nr_t> text_delayed(const std::string& character_name, const std::string& text, int characters_per_second);
    fan::event::task_value_resume_t<drawable_nr_t> text(const std::string& text);

    fan::event::task_value_resume_t<drawable_nr_t> button(const std::string& text, const fan::vec2& position, const fan::vec2& size = { 0, 0 });

    // default width 80% of the window
    fan::event::task_value_resume_t<drawable_nr_t> separator(f32_t width = 0.8);


    int get_button_choice();

    fan::event::task_t wait_user_input();
    void render(const std::string& window_name, font_t* font, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing, const std::function<void()>& inside_window_cb);

    void render(const std::string& window_name, font_t* font, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing);

    void clear();

    fan::vec2 cursor_position = -1;
    f32_t indent = -1;

    bool wait_user = false;
    drawable_nr_t button_choice;

    static void default_render_content(const fan::vec2& cursor_pos, f32_t indent);

    std::function<void(const fan::vec2&, f32_t)> render_content_cb = &default_render_content;

    f32_t font_size = 24;
  };
  // called inside window begin end
  void animated_popup_window(
    const std::string& popup_id,
    const fan::vec2& popup_size,
    const fan::vec2& start_pos,
    const fan::vec2& target_pos,
    bool trigger_popup,
    std::function<void()> content_cb,
    const f32_t anim_duration = 0.25f,
    const f32_t hide_delay = 0.5f
  );

  void text_partial_render(const std::string& text, size_t render_pos, f32_t wrap_width, f32_t line_spacing = 0);

  void render_texture_property(
    fan::graphics::shapes::shape_t& shape,
    int index,
    const char* label,
    const std::wstring& asset_path = L"./",
    f32_t image_size = 64.f,
    const char* receive_drag_drop_target_name = "CONTENT_BROWSER_ITEMS"
  );
  void render_image_filter_property(fan::graphics::shape_t& shape, const char* label);

  struct window_scope {
    template <typename... Args>
    window_scope(Args&&... args)
      : window(std::forward<Args>(args)...),
      active(static_cast<bool>(window)) {}

    explicit operator bool() const { return active; }

    fan::graphics::gui::window_t window;
    bool active;
  };

  struct hud : window_scope {
    hud(const std::string& name, bool* p_open = nullptr)
      : window_scope(
        (gui::set_next_window_pos(0),
          gui::set_next_window_size(gui::get_window_size()),
          name),
        p_open,
        gui::window_flags_no_background |
        gui::window_flags_no_nav |
        gui::window_flags_no_title_bar |
        gui::window_flags_no_resize |
        gui::window_flags_no_move |
        gui::window_flags_override_input
      ) {}
  };
}
/*
template fan::graphics::gui::imgui_fs_var_t::imgui_fs_var_t(
  fan::graphics::shader_t shader_nr,
  const std::string& var_name,
  fan::vec2 initial_,
  f32_t speed,
  f32_t min,
  f32_t max
);
template fan::graphics::gui::imgui_fs_var_t::imgui_fs_var_t(
  fan::graphics::shader_t shader_nr,
  const std::string& var_name,
  double initial_,
  f32_t speed,
  f32_t min,
  f32_t max
);
*/
#endif