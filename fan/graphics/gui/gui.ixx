module;

#if defined (FAN_WINDOW)

#include <fan/utility.h>

#endif

export module fan.graphics.gui;

#if defined (FAN_WINDOW)

import std;

#if defined(FAN_GUI)
  import fan.types;
  import fan.time;
  import fan.types.vector;
  import fan.types.color;
  
  import fan.types.compile_time_string;
  import fan.event;

  import fan.graphics.common_context;
  import fan.graphics.shapes;

  import fan.graphics.gui.types;
  import fan.graphics.gui.base;
  import fan.graphics.loco;
  import fan.console;
  
  import fan.formatter; // todo REMOVE
  import fan.graphics.common_types;
  import fan.log_dispatcher;
  import fan.graphics.shapes.types;
  import fan.event.types;
  import fan.io.types;
  import fan.window;
#endif

#if defined(FAN_3D)
  import fan.graphics.voxel;
#endif


#if defined(FAN_GUI)

export namespace fan {

  inline fan::console_t& g_console() {
    //static fan::console_t gconsole;
    return *(fan::console_t*)fan::graphics::ctx().console;
  }

  void printclnn(auto&&... values) {
    g_console().print(fan::format_args(values...) + " ", 0);
  }

  void printcl(auto&&... values) {
    g_console().print(fan::format_args(values...) + "\n", 0);
  }

  void printclnnh(int highlight, auto&&... values) {
    g_console().print(fan::format_args(values...) + " ", highlight);
  }

  void printclh(int highlight, auto&&... values) {
    g_console().print(fan::format_args(values...) + "\n", highlight);
  }

  inline void printcl_err(auto&&... values) {
    printclh(fan::graphics::highlight_e::error, values...);
  }

  inline void printcl_warn(auto&&... values) {
    printclh(fan::graphics::highlight_e::warning, values...);
  }
}

export namespace fan::graphics::gui {
  fan::log_dispatcher_t default_logger() {
    return fan::log_dispatcher_t {}
      .on("ERROR:", [](std::string_view l) { fan::printcl_err(l); })
      .on("error:", [](std::string_view l) { fan::printcl_err(l); })
      .on("WARNING:", [](std::string_view l) { fan::printcl_warn(l); })
      .on("warning:", [](std::string_view l) { fan::printcl_warn(l); })
      .otherwise([](std::string_view l) { fan::printcl(l); });
  }
}


export namespace fan::graphics::gui {

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

  bool toggle_image_button(fan::str_view_t char_id, fan::graphics::image_t image, const fan::vec2& size, bool* toggle);
  bool toggle_image_button(image_t* images, std::uint32_t count, const fan::vec2& size, int* selectedIndex);

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
  void shape_properties(fan::graphics::shape_t& shape);
#endif
} // namespace fan::graphics::gui

export namespace fan::graphics::gui {
  struct content_browser_t {
    struct file_info_t {
      std::string filename;
      std::string item_path;
      bool is_directory;
      fan::graphics::image_t preview_image;
      bool is_selected = false;
    };

    struct selection_state_t {
      bool is_selecting = false;
      fan::vec2 selection_start;
      fan::vec2 selection_end;
      std::vector<std::size_t> selected_indices;
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
    std::string pending_directory_change;

    std::string asset_path = "./";

    fan::io::async_directory_iterator_t directory_iterator;

    std::string current_directory;
    enum viewmode_e {
      view_mode_list,
      view_mode_large_thumbnails,
    };
    viewmode_e current_view_mode = view_mode_list;
    f32_t thumbnail_size = 128.0f;
    f32_t padding = 16.0f;
    std::string search_buffer;

    content_browser_t(const content_browser_t&) = delete;
    content_browser_t(content_browser_t&&) = delete;

    content_browser_t();
    content_browser_t(bool no_init);
    content_browser_t(const std::string& path);
    void init(const std::string& path);

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
      std::vector<std::pair<file_info_t, std::size_t>> sorted_cache;
      std::vector<std::pair<file_info_t, std::size_t>> sorted_search_cache;
      bool cache_dirty = true;
      bool search_cache_dirty = true;
    };

    search_state_t search_state;
    fan::io::async_directory_iterator_t search_iterator;

    void invalidate_cache();
    int get_pressed_key();
    void handle_keyboard_navigation(std::string_view filename, int pressed_key);
    void handle_right_click(std::string_view filename);
    void process_next_directory();
    void start_search(const std::string& query, bool recursive = false);
    void update_sorted_cache();
    void update_search_sorted_cache();
    void render();
    void render_large_thumbnails_view();
    void render_list_view();
    void handle_item_interaction(const file_info_t& file_info, std::size_t original_index);
    void receive_drag_drop_target(std::function<void(const std::string&)> receive_func);
  };

#if defined(FAN_2D)

  struct sprite_animations_t {
    //fan::vec2i frame_coords; // starting from top left and increasing by one to get the next frame into that direction

    fan::graphics::sprite_sheet_id_t current_animation_shape_nr;
    fan::graphics::sprite_sheet_id_t current_animation_nr;
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

    bool render_list_box(fan::graphics::sprite_sheet_id_t& shape_sprite_sheet_id);

    bool render_selectable_frames(fan::graphics::sprite_sheet_t& current_sprite_sheet);
    bool render(std::string_view drag_drop_id, fan::graphics::sprite_sheet_id_t& shape_sprite_sheet_id);
  };

  void fragment_shader_editor(std::uint16_t shape_type, std::string* fragment, bool* shader_compiled);
#endif

#if defined(FAN_2D)
  struct particle_editor_t {

    particle_editor_t();

    fan::graphics::shapes::particles_t::ri_t& get_ri();

    void render_menu();
    void render_settings();
    void render();

    #if defined(FAN_JSON)
    void fout(std::string_view filename);
    #endif

  private:
    fan::graphics::shape_t particle_shape = fan::graphics::shapes::particles_t::properties_t{
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
    fan::graphics::shape_t particle_image_sprite;

    /*fan::color bg_color = fan::color::from_rgba(0xB8C4BFFF);*/
    fan::color bg_color = fan::colors::black;
    fan::color base_color = fan::color::from_rgba(0x33333369);
    f32_t color_intensity = 1.0f;
    std::string filename{};
  };

#endif


  struct dialogue_box_t {
    struct render_type_t {
      virtual void render(dialogue_box_t* This, std::uint16_t nr, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing) = 0;
      virtual ~render_type_t() = default;
    };

    using drawable_nr_t = std::uint16_t;
    struct drawable_node_t {
      drawable_nr_t id;
      std::unique_ptr<render_type_t> ptr;
    };
    std::vector<drawable_node_t> drawables;
    drawable_nr_t next_id = 1;

    struct text_delayed_t : render_type_t {
      ~text_delayed_t() override;

      void render(dialogue_box_t* This, drawable_nr_t nr, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing) override;
      bool dialogue_line_finished = false; // for skipping
      std::string text;
      std::uint64_t character_per_s = 20;
      std::size_t render_pos = 0;
      fan::time::timer blink_timer{ (std::uint64_t)0.5e9, true };
      std::uint8_t render_cursor = false;
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

    fan::event::runv_t<drawable_nr_t> text_delayed(
      std::string_view character_name, 
      std::string_view text, 
      int characters_per_second = 20
    );
    fan::event::runv_t<drawable_nr_t> text(const std::string& text);

    fan::event::runv_t<drawable_nr_t> button(const std::string& text, const fan::vec2& position, const fan::vec2& size = { 0, 0 });

    // default width 80% of the window
    fan::event::runv_t<drawable_nr_t> separator(f32_t width = 0.8);


    int get_button_choice();

    fan::event::task_t wait_user_input();
    void render(fan::str_view_t window_name, font_t* font, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing, const std::function<void()>& inside_window_cb = []{});

    fan::event::runv_t<int> choice(
      std::string_view character_name,
      std::string_view question_text,
      std::span<const std::string_view> options,
      const fan::vec2& start = {0.5f, 0.3f},
      f32_t y_step = 0.15f
    );

    void clear();

    fan::vec2 cursor_position = -1;
    f32_t indent = -1;

    bool wait_user = false;
    int button_choice = -1;

    static void default_render_content(const fan::vec2& cursor_pos, f32_t indent);

    std::function<void(const fan::vec2&, f32_t)> render_content_cb = &default_render_content;

    f32_t font_size = 24;
  };
  // called inside window begin end
  void animated_popup_window(
    std::string_view popup_id,
    const fan::vec2& popup_size,
    const fan::vec2& start_pos,
    const fan::vec2& target_pos,
    bool trigger_popup,
    std::function<void()> content_cb,
    const f32_t anim_duration = 0.25f,
    const f32_t hide_delay = 0.5f
  );

  void text_partial_render(const std::string& text, std::size_t render_pos, f32_t wrap_width, f32_t line_spacing = 0);

  void render_texture_property(
    fan::graphics::shape_t& shape,
    int index,
    fan::str_view_t label,
    const std::string& asset_path = "./",
    f32_t image_size = 64.f,
    const char* receive_drag_drop_target_name = "CONTENT_BROWSER_ITEMS"
  );
  void render_image_filter_property(
    fan::graphics::shape_t& shape,
    fan::str_view_t label
  );

  struct shader_contols_t {
    bool vec3_as_color = false;
    bool vec4_as_color = true;
  };

  void shader_controls(fan::graphics::shader_t shader_id, const shader_contols_t& controls = {});

  struct hex_editor_t {
    struct config_t {
      int cols = 16;
      fan::vec2i group_size = {4, 4};
      f32_t auto_scale_min = 0.1f;
      f32_t zoom_speed = 0.1f;
      fan::color col_bg_sel = fan::color(0.5f, 0.1f, 0.1f, 1.f);
      fan::color col_bg_hover = fan::color(0.25f, 0.25f, 0.25f, 0.7f);
      fan::color col_text_addr = fan::color(0.4f, 0.8f, 0.8f, 1.f);
      fan::color col_text_sel = fan::color(1.f, 0.3f, 0.3f, 1.f);
      f32_t spacing_hex_group_mult = 0.5f;
      f32_t spacing_hex_item_mult = 0.0f;
      f32_t spacing_ascii_mult = 0.0f;
      f32_t inner_pad = 2.0f;
      f32_t y_spacing_mult = 0.1f;
    };

    struct metrics_t {
      f32_t scale;
      f32_t cell_w;
      f32_t ascii_w;
      f32_t char_w;
      std::uint64_t size;
      std::uint64_t rows;
    };

    enum class active_panel_t { hex, ascii };

    void render(const std::string_view window_name, fan::io::data_provider_t& data);
    void render(fan::io::data_provider_t& data);
    std::vector<std::uint8_t> get_selected_bytes(fan::io::data_provider_t& data) const;
    std::optional<std::uint64_t> get_active_cell(fan::io::data_provider_t& data) const;
    void set_file_drop_callback(const std::function<void(const fan::bytes_t& data)>& func);

  private:
    void render_cell(fan::io::data_provider_t& data, std::uint64_t idx, f32_t w, f32_t pad, bool is_dragging, bool is_hex);
    void render_data_inspector(fan::io::data_provider_t& data, bool little_endian = true);
    void process_clipboard(fan::io::data_provider_t& data);
    bool has_selection() const;
    std::pair<std::uint64_t, std::uint64_t> get_selection_bounds() const;
    bool is_selected(std::uint64_t idx) const;
    void update_selection(std::uint64_t idx, bool cell_hovered);
    std::uint32_t get_cell_flags(bool is_dragging, bool is_hex) const;
    f32_t get_spacing(std::uint64_t idx, std::uint64_t row_end, bool is_hex) const;

  public:
    config_t config;
    fan::window_t::drop_handle_t file_drop_handle;

  private:
    static constexpr std::uint64_t ascii_id_offset = 0x1000000000000000ULL;

    active_panel_t active_panel = active_panel_t::hex;
    metrics_t metrics {};

    std::optional<std::uint64_t> sel_start;
    std::optional<std::uint64_t> sel_end;
    std::optional<std::uint64_t> active_idx;
    std::optional<std::uint64_t> pending_focus_ascii;
    std::optional<std::uint64_t> pending_focus_hex;
    std::optional<std::uint64_t> hovered_hex_idx;
    std::optional<std::uint64_t> prev_hex_hover_idx;
    std::optional<std::uint64_t> hovered_ascii_idx;
    std::optional<std::uint64_t> prev_ascii_hover_idx;

    f32_t user_zoom = 1.0f;
    bool is_dragging = false;

    std::string active_edit_buf;
    std::optional<std::uint64_t> active_edit_initialized_idx;
  };

  template <FAN_UNIQUE_CALL>
  void hex_editor(const std::string_view window_name, fan::io::data_provider_t& data) {
    static hex_editor_t he;
    he.render(window_name, data);
  }

  template <FAN_UNIQUE_CALL>
  void hex_editor(fan::io::data_provider_t& data) {
    gui::hex_editor<FAN_UNIQUE_CALL_PASS>("", data);
  }

  template <FAN_UNIQUE_CALL>
  void hex_editor(const std::string_view window_name, std::vector<std::uint8_t>& data) {
    fan::io::memory_provider_t provider(data);
    fan::io::data_provider_t& p = provider;
    gui::hex_editor<FAN_UNIQUE_CALL_PASS>(window_name, p);
  }

  template <FAN_UNIQUE_CALL>
  void hex_editor(std::vector<std::uint8_t>& data) {
    fan::io::memory_provider_t provider(data);
    fan::io::data_provider_t& p = provider;
    gui::hex_editor<FAN_UNIQUE_CALL_PASS>("", p);
  }


#if defined(FAN_3D)
  bool terrain_noise_debug(
    fan::graphics::terrain_noise_t& noise,
    f32_t& block_size,
    int& view_dist,
    f32_t& move_speed
  );
#endif

  struct camera_controls_ranges_t {
    f32_t zfar_min = 1.f;
    f32_t zfar_max = 10000.f;

    f32_t znear_min = 0.001f;
    f32_t znear_max = 10.f;

    f32_t fov_min = 1.f;
    f32_t fov_max = 180.f;

    f32_t sensitivity_min = 0.01f;
    f32_t sensitivity_max = 1.f;

    f32_t speed_min = 1.f;
    f32_t speed_max = 10000.f;

    f32_t friction_min = 0.f;
    f32_t friction_max = 30.f;
  };
  
  template <FAN_UNIQUE_CALL>
  void camera_controls(
    const camera_controls_ranges_t& ranges = {},
    fan::graphics::camera_t cam = gloco()->perspective_render_view) {
    fan::graphics::context_camera_t& camera = gloco()->camera_get(cam);
    static f32_t friction = 12.f;
    static f32_t speed = 1000.f;
    static int id = 0;
    bool update = false;
    gui::push_id(&id);
    update |= gui::slider("zfar", &camera.zfar, std::max(ranges.zfar_min, camera.znear + 0.001f), ranges.zfar_max);
    update |= gui::slider("znear", &camera.znear, ranges.znear_min, std::min(ranges.znear_max, camera.zfar - 0.001f));
    update |= gui::slider("fov", &camera.fov, ranges.fov_min, ranges.fov_max);
    update |= gui::slider("sensitivity", &camera.sensitivity, ranges.sensitivity_min, ranges.sensitivity_max);
    update |= gui::slider("speed", &speed, ranges.speed_min, ranges.speed_max);
    update |= gui::slider("friction", &friction, ranges.friction_min, ranges.friction_max);

    gloco()->camera_move(speed, friction);
    if (update) {
      gloco()->camera_set_perspective(cam, camera.fov, gloco()->window.get_size());
    }
    gui::pop_id();
  }
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

#endif
