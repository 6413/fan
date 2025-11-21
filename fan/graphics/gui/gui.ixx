module;

#include <fan/utility.h>
#include <fan/event/types.h>

#if defined(fan_gui)
  #include <fan/imgui/imgui.h>
  #include <fan/imgui/imgui_internal.h>
  #include <fan/imgui/imgui_impl_glfw.h>
  #include <fan/imgui/implot.h>
#endif

import std;

export module fan.graphics.gui;

#if defined(fan_gui)
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

  export import :types;
#endif


#if defined(fan_gui)

#if defined(fan_gui)
export namespace fan::graphics::gui {
  bool begin(const std::string& window_name, bool* p_open = 0, window_flags_t window_flags = 0);
  void end() {
    ImGui::End();
  }
  bool begin_child(const std::string& window_name, const fan::vec2& size = fan::vec2(0, 0), child_window_flags_t child_window_flags = 0, window_flags_t window_flags = 0);
  void end_child();

  bool begin_tab_item(const std::string& label, bool* p_open = 0, window_flags_t window_flags = 0);
  void end_tab_item();
  bool begin_tab_bar(const std::string& tab_bar_name, window_flags_t window_flags = 0);
  void end_tab_bar();

  bool begin_main_menu_bar();
  void end_main_menu_bar();

  bool begin_menu_bar();

  void end_menu_bar();

  bool begin_menu(const std::string& label, bool enabled = true);
  void end_menu();

  void begin_group();
  void end_group();

  void table_setup_column(const std::string& label, ImGuiTableColumnFlags flags = 0, f32_t init_width_or_weight = 0.0f, ImGuiID user_id = 0);
  void table_headers_row();
  bool table_set_column_index(int column_n);

  bool menu_item(const std::string& label, const std::string& shortcut = "", bool selected = false, bool enabled = true);

  void same_line(f32_t offset_from_start_x = 0.f, f32_t spacing_w = -1.f);
  void new_line();

  ImGuiViewport* get_main_viewport();

  f32_t get_frame_height();

  f32_t get_text_line_height_with_spacing();

  fan::vec2 get_mouse_pos();

  bool selectable(const std::string& label, bool selected = false, selectable_flag_t flags = 0, const fan::vec2& size = fan::vec2(0, 0));
  bool selectable(const std::string& label, bool* p_selected, selectable_flag_t flags = 0, const fan::vec2& size = fan::vec2(0, 0));

  bool is_mouse_double_clicked(int button = 0);

  fan::vec2 get_content_region_avail();
  fan::vec2 get_content_region_max();
  fan::vec2 get_item_rect_min();
  fan::vec2 get_item_rect_max();
  void push_item_width(f32_t item_width);
  void pop_item_width();

  void set_cursor_screen_pos(const fan::vec2& pos);

  void push_id(const std::string& str_id);
  void push_id(int int_id);
  void pop_id();

  void set_next_item_width(f32_t width);

  void push_text_wrap_pos(f32_t local_pos = 0);
  void pop_text_wrap_pos();

  bool is_item_hovered(hovered_flag_t flags = 0);
  bool is_any_item_hovered();
  bool is_any_item_active();
  bool is_item_clicked();
  bool is_item_held(int mouse_button = 0);

  void begin_tooltip();
  void end_tooltip();

  void set_tooltip(const std::string& tooltip);

  bool begin_table(const std::string& str_id, int columns, table_flags_t flags = 0, const fan::vec2& outer_size = fan::vec2(0.0f, 0.0f), f32_t inner_width = 0.0f);

  void end_table();

  void table_next_row(table_row_flags_t row_flags = 0, f32_t min_row_height = 0.0f);
  bool table_next_column();

  void columns(int count = 1, const char* id = nullptr, bool borders = true);

  void next_column();

  void push_font(font_t* font);
  void pop_font();
  font_t* get_font();
  f32_t get_font_size();
  f32_t get_text_line_height();

  void indent(f32_t indent_w = 0.0f);
  void unindent(f32_t indent_w = 0.0f);

  fan::vec2 calc_text_size(const std::string& text, const char* text_end = NULL, bool hide_text_after_double_hash = false, f32_t wrap_width = -1.0f);
  fan::vec2 text_size(const std::string& text, const char* text_end = NULL, bool hide_text_after_double_hash = false, f32_t wrap_width = -1.0f);
  void set_cursor_pos_x(f32_t pos);
  void set_cursor_pos_y(f32_t pos);
  void set_cursor_pos(const fan::vec2& pos);
  fan::vec2 get_cursor_pos();
  f32_t get_cursor_pos_x();
  f32_t get_cursor_pos_y();
  fan::vec2 get_cursor_screen_pos();
  fan::vec2 get_cursor_start_pos();

  bool is_window_hovered(hovered_flag_t hovered_flags = 0);
  bool is_window_focused();

  fan::vec2 get_window_content_region_min();
  fan::vec2 get_window_content_region_max();
  f32_t get_column_width(int index = -1);
  void set_column_width(int index, f32_t width);

  bool is_item_active();

  void set_keyboard_focus_here();

  fan::vec2 get_mouse_drag_delta(int button = 0, f32_t lock_threshold = -1.0f);

  void reset_mouse_drag_delta(int button = 0);

  void set_scroll_x(f32_t scroll_x);

  void set_scroll_y(f32_t scroll_y);
  f32_t get_scroll_x();
  f32_t get_scroll_y();


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

  bool button(const std::string& label, const fan::vec2& size = fan::vec2(0, 0));
  bool invisible_button(const std::string& label, const fan::vec2& size = fan::vec2(0, 0));

  /// <summary>
  /// Draws the specified text, with its position influenced by other GUI elements.
  /// </summary>
  /// <param name="text">The text to draw.</param>
  /// <param name="color">The color of the text (defaults to white).</param>
  void text(const std::string& text, const fan::color& color = fan::colors::white);
  /// <summary>
  /// Draws the specified text, with its position influenced by other GUI elements.
  /// </summary>
  /// <param name="text">The text to draw.</param>
  /// <param name="color">The color of the text (defaults to white).</param>
  void text(const char* text, const fan::color& color = fan::colors::white);

  /// <summary>
  /// Draws text constructed from multiple arguments, with optional color.
  /// </summary>
  /// <param name="color">The color of the text.</param>
  /// <param name="args">Arguments to be concatenated with spaces.</param>
  template <typename ...Args>
  void text(const fan::color& color, const Args&... args) {
    std::ostringstream oss;
    int idx = 0;
    ((oss << args << (++idx == sizeof...(args) ? "" : " ")), ...);

    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::Text("%s", oss.str().c_str());
    ImGui::PopStyleColor();
  }

  /// <summary>
  /// Draws text constructed from multiple arguments with default white color.
  /// </summary>
  /// <param name="args">Arguments to be concatenated with spaces.</param>
  template <typename ...Args>
  void text(const Args&... args) {
    text(fan::colors::white, args...);
  }

  /// <summary>
  /// Draws the specified text at a given position on the screen.
  /// </summary>
  /// <param name="text">The text to draw.</param>
  /// <param name="position">The position of the text.</param>
  /// <param name="color">The color of the text (defaults to white).</param>
  void text_at(const std::string& text, const fan::vec2& position = 0, const fan::color& color = fan::colors::white);

  void text_wrapped(const std::string& text, const fan::color& color = fan::colors::white);
  void text_unformatted(const std::string& text, const char* text_end = NULL);

  /// <summary>
  /// Draws text centered horizontally.
  /// </summary>
  /// <param name="text">The text to draw.</param>
  /// <param name="color">The color of the text (defaults to white).</param>
  /// <param name="offset">Offset from center position.</param>
  void text_centered(const std::string& text, const fan::color& color = fan::colors::white);

  /// <summary>
  /// Draws text centered at a specific position.
  /// </summary>
  /// <param name="text">The text to draw.</param>
  /// <param name="center_position">The position where the text should be centered.</param>
  /// <param name="color">The color of the text (defaults to white).</param>
  void text_centered_at(const std::string& text, const fan::vec2& center_position, const fan::color& color = fan::colors::white);

  /// <summary>
  /// Draws text to bottom right.
  /// </summary>
  /// <param name="text">The text to draw.</param>
  /// <param name="color">The color of the text (defaults to white).</param>
  /// <param name="offset">Offset from the bottom-right corner.</param>
  void text_bottom_right(const std::string& text, const fan::color& color = fan::colors::white, const fan::vec2& offset = 0);

  constexpr const f32_t outline_thickness = 1.5f;
  constexpr const fan::vec2 outline_offsets[] = {
    {0, -outline_thickness},  // top
    {-outline_thickness, 0},  // left
    {outline_thickness, 0},   // right
    {0, outline_thickness}    // bottom
  };

  void text_outlined_at(const std::string& text, const fan::vec2& screen_pos, const fan::color& color = fan::colors::white, const fan::color& outline_color = fan::colors::black);

  void text_outlined(const std::string& text, const fan::color& color = fan::colors::white, const fan::color& outline_color = fan::colors::black);

  /// <summary>
  /// Draws outlined text centered at a specific position.
  /// </summary>
  /// <param name="text">The text to draw.</param>
  /// <param name="center_position">The position where the text should be centered.</param>
  /// <param name="color">The color of the text (defaults to white).</param>
  /// <param name="outline_color">The color of the outline (defaults to black).</param>
  void text_centered_outlined_at(const std::string& text, const fan::vec2& center_position, const fan::color& color = fan::colors::white, const fan::color& outline_color = fan::colors::black);

  /// <summary>
  /// Draws outlined text centered horizontally within the current window.
  /// </summary>
  /// <param name="text">The text to draw.</param>
  /// <param name="color">The color of the text (defaults to white).</param>
  /// <param name="outline_color">The color of the outline (defaults to black).</param>
  void text_centered_outlined(const std::string& text, const fan::color& color = fan::colors::white, const fan::color& outline_color = fan::colors::black);

  void text_box(const std::string& text,
    const ImVec2& size = ImVec2(0, 0),
    const fan::color& text_color = fan::colors::white,
    const fan::color& bg_color = fan::color());

  template<typename T>
  constexpr ImGuiDataType get_imgui_data_type() {
    if constexpr (std::is_same_v<T, int8_t>) return ImGuiDataType_S8;
    else if constexpr (std::is_same_v<T, uint8_t>) return ImGuiDataType_U8;
    else if constexpr (std::is_same_v<T, int16_t>) return ImGuiDataType_S16;
    else if constexpr (std::is_same_v<T, uint16_t>) return ImGuiDataType_U16;
    else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int>) return ImGuiDataType_S32;
    else if constexpr (std::is_same_v<T, uint32_t>) return ImGuiDataType_U32;
    else if constexpr (std::is_same_v<T, int64_t>) return ImGuiDataType_S64;
    else if constexpr (std::is_same_v<T, uint64_t>) return ImGuiDataType_U64;
    else if constexpr (std::is_same_v<T, f32_t>) return ImGuiDataType_Float;
    else if constexpr (std::is_same_v<T, double>) return ImGuiDataType_Double;
    else static_assert(false, "Unsupported type for ImGui");
  }

  template<typename T>
  constexpr const char* get_default_format() {
    if constexpr (std::is_integral_v<T>) {
      if constexpr (std::is_signed_v<T>) return "%d";
      else return "%u";
    }
    else return "%.3f";
  }

  template<typename T>
  bool slider(const std::string& label, T* v, auto v_min, auto v_max, ImGuiSliderFlags flags = 0) {
    if constexpr (get_component_count<T>() == 1) {
      T min_val = static_cast<T>(v_min);
      T max_val = static_cast<T>(v_max);
      return ImGui::SliderScalar(label.c_str(), get_imgui_data_type<T>(), v, &min_val, &max_val, get_default_format<T>(), flags);
    }
    else {
      using component_type = component_type_t<T>;
      component_type min_val = static_cast<component_type>(v_min);
      component_type max_val = static_cast<component_type>(v_max);
      return ImGui::SliderScalarN(label.c_str(), get_imgui_data_type<component_type>(), v->data(), get_component_count<T>(), &min_val, &max_val, get_default_format<component_type>(), flags);
    }
  }
  template<typename T>
  bool slider(const std::string& label, T* v, ImGuiSliderFlags flags = 0) {
    if constexpr (std::is_integral_v<T> || std::is_integral_v<component_type_t<T>>) {
      return slider(label, v, 0, 100, flags);
    }
    else {
      return slider(label, v, 0.0, 1.0, flags);
    }
  }


  template<typename T>
  bool drag(const std::string& label, T* v, f32_t v_speed = 1.f, f32_t v_min = 0, f32_t v_max = 0, ImGuiSliderFlags flags = 0) {
    if constexpr (get_component_count<T>() == 1) {
      T speed_val = static_cast<T>(v_speed);
      T min_val = static_cast<T>(v_min);
      T max_val = static_cast<T>(v_max);
      return ImGui::DragScalar(label.c_str(), get_imgui_data_type<T>(), v, speed_val, &min_val, &max_val, get_default_format<T>(), flags);
    }
    else {
      using component_type = component_type_t<T>;
      component_type speed_val = static_cast<component_type>(v_speed);
      component_type min_val = static_cast<component_type>(v_min);
      component_type max_val = static_cast<component_type>(v_max);
      return ImGui::DragScalarN(label.c_str(), get_imgui_data_type<component_type>(), v->data(), get_component_count<T>(), speed_val, &min_val, &max_val, get_default_format<component_type>(), flags);
    }
  }
  //template<typename T>
  //bool drag(const std::string& label, T* v, ImGuiSliderFlags flags = 0) {
  //  return drag(label, v, 1, 0, 0, flags);
  //}


  f32_t calc_item_width();

  f32_t get_item_width();

  //imgui_stdlib.cpp:
  int InputTextCallback(ImGuiInputTextCallbackData* data);

  bool input_text(const std::string& label, std::string* buf, input_text_flags_t flags = 0, input_text_callback_t callback = nullptr, void* user_data = nullptr);
  bool input_text_multiline(const std::string& label, std::string* buf, const ImVec2& size = ImVec2(0, 0), input_text_flags_t flags = 0, input_text_callback_t callback = nullptr, void* user_data = nullptr);

  bool input_float(const std::string& label, f32_t* v, f32_t step = 0.0f, f32_t step_fast = 0.0f, const char* format = "%.3f", input_text_flags_t flags = 0);
  bool input_float(const std::string& label, fan::vec2* v, const char* format = "%.3f", input_text_flags_t flags = 0);
  bool input_float(const std::string& label, fan::vec3* v, const char* format = "%.3f", input_text_flags_t flags = 0);
  bool input_float(const std::string& label, fan::vec4* v, const char* format = "%.3f", input_text_flags_t flags = 0);
  bool input_int(const std::string& label, int* v, int step = 1, int step_fast = 100, input_text_flags_t flags = 0);
  bool input_int(const std::string& label, fan::vec2i* v, input_text_flags_t flags = 0);
  bool input_int(const std::string& label, fan::vec3i* v, input_text_flags_t flags = 0);
  bool input_int(const std::string& label, fan::vec4i* v, input_text_flags_t flags = 0);

  bool color_edit3(const std::string& label, fan::color* color, color_edit_flags_t flags = 0);

  bool color_edit3(const std::string& label, fan::vec3* color, color_edit_flags_t flags= 0);

  bool color_edit4(const std::string& label, fan::color* color, color_edit_flags_t flags = 0);

  fan::vec2 get_window_pos();
  fan::vec2 get_window_size();

  void set_next_window_pos(const fan::vec2& position, cond_t cond = cond_none, const fan::vec2& pivot = 0);

  void set_next_window_size(const fan::vec2& size, cond_t cond = cond_none);

  void set_next_window_bg_alpha(f32_t a);

  void set_window_font_scale(f32_t scale);

  bool is_mouse_dragging(int button = 0, f32_t threshold = -1.0f);

  bool is_item_deactivated_after_edit();

  void set_mouse_cursor(cursor_t type);

  style_t& get_style();
  fan::color get_color(col_t idx);
  uint32_t get_color_u32(col_t idx);

  void separator();
  void spacing();

  io_t& get_io();

  template <typename... Args>
  bool tree_node_ex(void* id, tree_node_flags_t flags, const char* fmt, Args&&... args) {
    return ImGui::TreeNodeEx(id, flags, fmt, std::forward<Args>(args)...);
  }
  void tree_pop();

  bool is_item_toggled_open();

  void push_style_color(col_t index, const fan::color& col);

  void pop_style_color(int n = 1);

  void push_style_var(style_var_t index, f32_t val);

  void push_style_var(style_var_t index, const fan::vec2& val);

  void pop_style_var(int n = 1);

  void dummy(const fan::vec2& size);

  draw_list_t* get_window_draw_list();
  draw_list_t* get_background_draw_list();

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

    void set(const auto& lambda);
  };

  struct imgui_element_t : imgui_element_nr_t {
    imgui_element_t() = default;
    imgui_element_t(const auto& lambda);
  };


  const char* item_getter1(const std::vector<std::string>& items, int index);

  void set_viewport(fan::graphics::viewport_t viewport);
  void set_viewport(const fan::graphics::render_view_t& render_view);

  using window_handle_t = ImGuiWindow;
  window_handle_t* get_current_window();

  /// <summary>
  /// Draws the specified button, with its position influenced by other GUI elements.
  /// Plays default hover and click audio piece if none specified.
  /// </summary>
  /// <param name="label">Name of the button. Draws the given label next to the button. The label is hideable using "##hidden_label".</param>
  /// <param name="piece_hover">Audio piece that is played when hovering the button.</param>
  /// <param name="piece_click">Audio piece that is played when clicking and releasing the button.</param>
  /// <param name="size">Size of the button (defaults to automatic).</param>
  /// <returns></returns>
#if defined (fan_audio)
  bool audio_button(
    const std::string& label, 
    fan::audio::piece_t piece_hover = fan::audio::piece_invalid, 
    fan::audio::piece_t piece_click = fan::audio::piece_invalid, 
    const fan::vec2& size = fan::vec2(0, 0)
  );
#endif

  bool combo(const std::string& label, int* current_item, const char* const items[], int items_count, int popup_max_height_in_items = -1);
  // Separate items with \0 within a string, end item-list with \0\0. e.g. "One\0Two\0Three\0"
  bool combo(const std::string& label, int* current_item, const char* items_separated_by_zeros, int popup_max_height_in_items = -1);
  bool combo(const std::string& label, int* current_item, const char* (*getter)(void* user_data, int idx), void* user_data, int items_count, int popup_max_height_in_items = -1);
  bool checkbox(const std::string& label, bool* v);
  bool list_box(const std::string &label, int* current_item, bool (*old_callback)(void* user_data, int idx, const char** out_text), void* user_data, int items_count, int height_in_items = -1);
  bool list_box(const std::string& label, int* current_item, const char* const items[], int items_count, int height_in_items = -1);
  bool list_box(const std::string& label, int* current_item, const char* (*getter)(void* user_data, int idx), void* user_data, int items_count, int height_in_items = -1);

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

  bool toggle_button(const std::string& str, bool* v);
  bool toggle_image_button(const std::string& char_id, fan::graphics::image_t image, const fan::vec2& size, bool* toggle);

  void text_bottom_right(const std::string& text, uint32_t reverse_yoffset = 0);


  template <std::size_t N>
  bool toggle_image_button(const std::array<fan::graphics::image_t, N>& images, const fan::vec2& size, int* selectedIndex)
  {
    f32_t y_pos = ImGui::GetCursorPosY() + ImGui::GetStyle().WindowPadding.y - ImGui::GetStyle().FramePadding.y / 2;

    bool clicked = false;
    bool pushed = false;

    for (std::size_t i = 0; i < images.size(); ++i) {
      ImVec4 tintColor = ImVec4(0.2, 0.2, 0.2, 1.0);
      if (*selectedIndex == i) {
        tintColor = ImVec4(0.2, 0.2, 0.2, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_Button, tintColor);
        pushed = true;
      }
      /*if (ImGui::IsItemHovered()) {
      tintColor = ImVec4(1, 1, 1, 1.0f);
      }*/
      ImGui::SetCursorPosY(y_pos);
      if (fan::graphics::gui::image_button("##toggle_image_button" + std::to_string(i) + std::to_string((uint64_t)&clicked), images[i], size)) {
        *selectedIndex = i;
        clicked = true;
      }
      if (pushed) {
        ImGui::PopStyleColor();
        pushed = false;
      }

      ImGui::SameLine();
    }

    return clicked;
  }


  fan::vec2 get_position_bottom_corner(const std::string& text = "", uint32_t reverse_yoffset = 0);

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
  void send_drag_drop_item(const std::string& id, const std::wstring& path, const std::string& popup = "");
  void receive_drag_drop_target(const std::string& id, std::function<void(std::string)> receive_func);
  namespace plot {
    bool begin_plot(const std::string& title, const fan::vec2& size = fan::vec2(-1, 0), flags_t flags = flags_none);
    void end_plot();
    void setup_axes(const std::string& x_label, const std::string& y_label,
      axis_flags_t x_flags = axis_flags_none, axis_flags_t y_flags = axis_flags_none);
    void setup_axis(axis_t axis, const std::string& label = "", axis_flags_t flags = axis_flags_none);
    void setup_axis_limits(axis_t axis, double v_min, double v_max, cond_t cond = cond_once);
    void setup_axes_limits(double x_min, double x_max, double y_min, double y_max, cond_t cond = cond_once);
    void setup_axis_format(axis_t idx, const std::string& format);
    void setup_axis_links(ImAxis idx, double* min_lnk, double* max_lnk);
    void setup_axis_format(ImAxis idx, ImPlotFormatter formatter, void* data);
    void setup_legend(location_t location, int flags = 0);
    void setup_finish();

    void plot_line(const std::string& label, const std::vector<f32_t>& values,
      double x_scale = 1.0, double x_start = 0.0, line_flags_t flags = line_flags_none);
    void plot_line(const std::string& label, const std::vector<f32_t>& xs, const std::vector<f32_t>& ys,
      line_flags_t flags = line_flags_none);
    void plot_line(const std::string& label, const f32_t* xs, const f32_t* ys, int count,
      line_flags_t flags = line_flags_none);
    void plot_scatter(const std::string& label, const std::vector<f32_t>& values,
      double x_scale = 1.0, double x_start = 0.0, scatter_flags_t flags = scatter_flags_none);
    void plot_scatter(const std::string& label, const std::vector<f32_t>& xs, const std::vector<f32_t>& ys,
      scatter_flags_t flags = scatter_flags_none);
    void plot_scatter(const std::string& label, const f32_t* xs, const f32_t* ys, int count,
      scatter_flags_t flags = scatter_flags_none);
    void plot_bars(const std::string& label, const std::vector<f32_t>& values,
      double bar_size = 0.67, double shift = 0, bars_flags_t flags = bars_flags_none);
    void plot_bars(const std::string& label, const std::vector<f32_t>& xs, const std::vector<f32_t>& ys,
      double bar_size, bars_flags_t flags = bars_flags_none);
    void plot_shaded(const std::string& label, const std::vector<f32_t>& xs, const std::vector<f32_t>& ys,
      double y_ref = 0.0, int flags = 0);

    void push_style_color(col_t idx, const fan::color& color);
    void pop_style_color(int count = 1);
    void push_style_var(int idx, f32_t val);
    void push_style_var(int idx, const fan::vec2& val);
    void pop_style_var(int count = 1);

    void set_next_line_style(const fan::color& col = fan::color(0, 0, 0, -1), f32_t weight = -1.0f);
    void set_next_fill_style(const fan::color& col = fan::color(0, 0, 0, -1), f32_t alpha_mod = -1.0f);
    void set_next_marker_style(marker_t marker = -1, f32_t size = -1.0f,
      const fan::color& fill = fan::color(0, 0, 0, -1), f32_t weight = -1.0f,
      const fan::color& outline = fan::color(0, 0, 0, -1));

    fan::vec2 get_plot_pos();

    fan::vec2 get_plot_size();

    bool is_plot_hovered();

    bool is_axis_hovered(axis_t axis);

    fan::vec2 pixels_to_plot(const fan::vec2& pix, axis_t x_axis = IMPLOT_AUTO, axis_t y_axis = IMPLOT_AUTO);
    fan::vec2 plot_to_pixels(double x, double y, axis_t x_axis = IMPLOT_AUTO, axis_t y_axis = IMPLOT_AUTO);

    fan::vec2 get_plot_mouse_pos(axis_t x_axis = IMPLOT_AUTO, axis_t y_axis = IMPLOT_AUTO);

    void annotation(double x, double y, const fan::color& col, const fan::vec2& pix_offset, bool clamp, const std::string& text);

    void tag_x(double x, const fan::color& col, const std::string& text = "");
    void tag_y(double y, const fan::color& col, const std::string& text = "");

    void plot_text(const std::string& text, double x, double y, const fan::vec2& pix_offset = fan::vec2(0, 0), int flags = 0);
    void plot_dummy(const std::string& label_id, int flags = 0);

    fan::color next_colormap_color();
    fan::color get_last_item_color();
  } // namespace plot
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

  void shape_properties(const fan::graphics::shape_t& shape);
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
      ImVec2 selection_start;
      ImVec2 selection_end;
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

    bool is_point_in_rect(ImVec2 point, ImVec2 rect_min, ImVec2 rect_max);

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
    void receive_drag_drop_target(auto receive_func);
  };

  ImFont* get_font_impl(f32_t font_size, bool bold = false);

  font_t* get_font(f32_t font_size, bool bold = false);

  bool begin_popup(const std::string& id, window_flags_t flags = 0);
  bool begin_popup_modal(const std::string& id, window_flags_t flags = 0);
  void end_popup();
  void open_popup(const std::string& id);
  void close_current_popup();
  bool is_popup_open(const std::string& id);


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


  struct particle_editor_t {
    fan::graphics::shapes::particles_t::ri_t& get_ri();

    void handle_file_operations();

    void render_menu();

    void render_settings();

    void render();

    fan::graphics::shapes::shape_t particle_shape = fan::graphics::shapes::particles_t::properties_t{
      .position = fan::vec3(32.108f, -1303.084f, 10.0f),
      .size = 28.638f,
      .color = fan::color::from_rgba(0x33333369),
      .alive_time = 1768368768,
      .count = 1191,
      .position_velocity = fan::vec2(0.0f, 9104.127f),
      .begin_angle = 0,
      .end_angle = -0.16f,
      .angle = fan::vec3(0.0f, 0.0f, -0.494f),
      .gap_size = fan::vec2(400.899f, 1.0f),
      .max_spread_size = fan::vec2(2648.021f, 1.0f),
      .shape = fan::graphics::shapes::particles_t::shapes_e::rectangle,
      .image = fan::graphics::image_load("images/waterdrop.webp")
    };

    fan::color bg_color = fan::color::from_rgba(0xB8C4BFFF);
    fan::color base_color = fan::color::from_rgba(0x33333369);
    f32_t color_intensity = 1.0f;
    fan::graphics::file_save_dialog_t save_file_dialog{};
    fan::graphics::file_open_dialog_t open_file_dialog{};
    std::string filename{};
  };


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
    void render(const std::string& window_name, ImFont* font, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing, const auto& inside_window_cb);

    void render(const std::string& window_name, ImFont* font, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing);

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
    auto content_cb,
    const f32_t anim_duration = 0.25f,
    const f32_t hide_delay = 0.5f
  );

  // Text that is added (stacked) to bottom left and fades away after specified time
  //-------------------------------------Floating text-------------------------------------
  template <typename ...Args>
  void print(const Args&... args) {
    fan::graphics::g_render_context_handle.text_logger->print(args...);
  }
  template <typename ...Args>
  void print(const fan::color& color, const Args&... args) {
    fan::graphics::g_render_context_handle.text_logger->print(color, args...);
  }
  template <typename... args_t>
  void printf(std::string_view fmt, args_t&&... args) {
    fan::graphics::g_render_context_handle.text_logger->printf(fmt, std::forward<args_t>(args)...);
  }
  template <typename... args_t>
  void printf(const fan::color& color, std::string_view fmt, args_t&&... args) {
    fan::graphics::g_render_context_handle.text_logger->printf(color, fmt, std::forward<args_t>(args)...);
  }
  template <typename... args_t>
  void printft(std::streamsize tab_width, std::string_view fmt, args_t&&... args) {
    fan::graphics::g_render_context_handle.text_logger->printft(tab_width, fmt, std::forward<args_t>(args)...);
  }
  template <typename... args_t>
  void printft(std::streamsize tab_width, const fan::color& color, std::string_view fmt, args_t&&... args) {
    fan::graphics::g_render_context_handle.text_logger->printft(tab_width, color, fmt, std::forward<args_t>(args)...);
  }
  template <typename ...args_t>
  void print_error(args_t&&... args) {
    fan::graphics::g_render_context_handle.text_logger->print(fan::colors::red, std::forward<args_t>(args)...);
  }
  template <typename ...args_t>
  void print_warning(args_t&&... args) {
    fan::graphics::g_render_context_handle.text_logger->print(fan::colors::yellow, std::forward<args_t>(args)...);
  }
  template <typename ...args_t>
  void print_success(args_t&&... args) {
    fan::graphics::g_render_context_handle.text_logger->print(fan::colors::green, std::forward<args_t>(args)...);
  }
  void set_text_fade_time(f32_t seconds);
  //-------------------------------------Floating text-------------------------------------

  // Text that is added (stacked) to bottom left and it never disappears
  //-------------------------------------Static text-------------------------------------
  template <typename ...Args>
  void print_static(const Args&... args) {
    fan::graphics::g_render_context_handle.text_logger->print_static(args...);
  }
  template <typename ...Args>
  void print_static(const fan::color& color, const Args&... args) {
    fan::graphics::g_render_context_handle.text_logger->print_static(color, args...);
  }
  template <typename... args_t>
  void printf_static(std::string_view fmt, args_t&&... args) {
    fan::graphics::g_render_context_handle.text_logger->printf_static(fmt, std::forward<args_t>(args)...);
  }
  template <typename... args_t>
  void printf_static(const fan::color& color, std::string_view fmt, args_t&&... args) {
    fan::graphics::g_render_context_handle.text_logger->printf_static(color, fmt, std::forward<args_t>(args)...);
  }
  template <typename... args_t>
  void printft_static(std::streamsize tab_width, std::string_view fmt, args_t&&... args) {
    fan::graphics::g_render_context_handle.text_logger->printft_static(tab_width, fmt, std::forward<args_t>(args)...);
  }
  template <typename... args_t>
  void printft_static(std::streamsize tab_width, const fan::color& color, std::string_view fmt, args_t&&... args) {
    fan::graphics::g_render_context_handle.text_logger->printft_static(tab_width, color, fmt, std::forward<args_t>(args)...);
  }
  void clear_static_text();
  //-------------------------------------Static text-------------------------------------

  void text_partial_render(const std::string& text, size_t render_pos, f32_t wrap_width, f32_t line_spacing = 0);
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