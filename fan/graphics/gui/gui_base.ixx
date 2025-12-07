module;

#if defined(fan_gui)
  #include <fan/imgui/imgui.h>
  #include <fan/imgui/imgui_impl_glfw.h>
  #include <fan/imgui/implot.h>
  #include <fan/imgui/imgui_internal.h>

  #include <string>
  #include <sstream>
  #include <functional>
  #include <cstdint>
#endif

#if defined(fan_vulkan)
  #include <vulkan/vulkan.h>
#endif

export module fan.graphics.gui.base;

#if defined(fan_gui)

#if defined (fan_audio)
import fan.audio;
#endif

export import fan.graphics.gui.types;
import fan.types.vector;
import fan.types.color;
import fan.utility;
import fan.math;
import fan.print;

export namespace fan::graphics::gui {
  bool begin(const std::string& window_name, bool* p_open = 0, window_flags_t window_flags = 0);
  void end();
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

  void table_setup_column(const std::string& label, table_column_flags_t flags = 0, f32_t init_width_or_weight = 0.0f, id_t user_id = 0);
  void table_headers_row();
  bool table_set_column_index(int column_n);

  bool menu_item(const std::string& label, const std::string& shortcut = "", bool selected = false, bool enabled = true);
  
  bool begin_combo(const std::string& label, const std::string& preview_value, int flags = 0);
  void end_combo();

  void set_item_default_focus();

  void same_line(f32_t offset_from_start_x = 0.f, f32_t spacing_w = -1.f);
  void new_line();

  viewport_t* get_main_viewport();

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
  void item_size(const fan::vec2& s);
  void item_size(const rect_t& bb, f32_t text_baseline_y = -1.0f);
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

  // gvars
  inline constexpr f32_t font_sizes[] = {
    4, 5, 6, 7, 8, 9, 10, 11, 12, 14,
    16, 18, 20, 22, 24, 28,
    32, 36, 48, 60, 72
  };
  fan::graphics::gui::font_t* fonts[std::size(font_sizes)]{};
  fan::graphics::gui::font_t* fonts_bold[std::size(font_sizes)]{};

  void build_fonts();
  void rebuild_fonts();
  void load_fonts(font_t* (&fonts)[std::size(fan::graphics::gui::font_sizes)], const std::string& name, font_config_t* cfg = nullptr);

  void push_font(font_t* font);
  void pop_font();
  font_t* get_font();
  f32_t get_font_size();
  f32_t get_text_line_height();

  void indent(f32_t indent_w = 0.0f);
  void unindent(f32_t indent_w = 0.0f);

  fan::vec2 calc_text_size(const std::string& text, const char* text_end = NULL, bool hide_text_after_double_hash = false, f32_t wrap_width = -1.0f);
  fan::vec2 get_text_size(const std::string& text, const char* text_end = NULL, bool hide_text_after_double_hash = false, f32_t wrap_width = -1.0f);
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
  void set_next_window_focus();
  void set_window_focus(const std::string& name);

  int render_window_flags();

  fan::vec2 get_window_content_region_min();
  fan::vec2 get_window_content_region_max();
  f32_t get_column_width(int index = -1);
  void set_column_width(int index, f32_t width);

  bool is_item_active();

  bool want_io();
  void set_want_io(bool flag = ImGui::GetIO().WantCaptureMouse | 
    ImGui::GetIO().WantCaptureKeyboard | 
    ImGui::GetIO().WantTextInput, 
    bool op_or = false
  );

  void set_keyboard_focus_here();

  fan::vec2 get_mouse_drag_delta(int button = 0, f32_t lock_threshold = -1.0f);

  void reset_mouse_drag_delta(int button = 0);

  void set_scroll_x(f32_t scroll_x);

  void set_scroll_y(f32_t scroll_y);
  void set_scroll_here_y();
  f32_t get_scroll_x();
  f32_t get_scroll_y();

  template<typename T>
  constexpr data_type_t get_imgui_data_type() {
    if constexpr (std::is_same_v<T, std::int8_t>) return ImGuiDataType_S8;
    else if constexpr (std::is_same_v<T, std::uint8_t>) return ImGuiDataType_U8;
    else if constexpr (std::is_same_v<T, std::int16_t>) return ImGuiDataType_S16;
    else if constexpr (std::is_same_v<T, std::uint16_t>) return ImGuiDataType_U16;
    else if constexpr (std::is_same_v<T, std::int32_t> || std::is_same_v<T, int>) return ImGuiDataType_S32;
    else if constexpr (std::is_same_v<T, std::uint32_t>) return ImGuiDataType_U32;
    else if constexpr (std::is_same_v<T, std::int64_t>) return ImGuiDataType_S64;
    else if constexpr (std::is_same_v<T, std::uint64_t>) return ImGuiDataType_U64;
    else if constexpr (std::is_same_v<T, f32_t>) return ImGuiDataType_Float;
    else if constexpr (std::is_same_v<T, double>) return ImGuiDataType_Double;
    else static_assert(false, "Unsupported type for ImGui");
  }

  void push_style_color(col_t index, const fan::color& col);
  void pop_style_color(int n = 1);

  void push_style_var(style_var_t index, f32_t val);
  void push_style_var(style_var_t index, const fan::vec2& val);
  void pop_style_var(int n = 1);

  bool button(const std::string& label, const fan::vec2& size = fan::vec2(0, 0));
  bool invisible_button(const std::string& label, const fan::vec2& size = fan::vec2(0, 0));
  bool arrow_button(const std::string& label, dir_t dir);

  /// <summary>
  /// Draws the specified text, with its position influenced by other GUI elements.
  /// </summary>
  /// <param name="color">The color of the text.</param>
  /// <param name="text">The text args to draw.</param>
  template <typename ...Args>
  void text(const fan::color& color, const Args&... args) {
    gui::push_style_color(col_text, color);
    ImGui::Text(fan::format_args(args...).c_str());
    gui::pop_style_color();
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
  void text_disabled(const std::string& text);

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

  f32_t calc_item_width();
  f32_t get_item_width();

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
  bool tree_node_ex(const std::string& label, tree_node_flags_t flags = 0);
  void tree_pop();

  bool tree_node(const std::string& label);

  bool is_item_toggled_open();

  void dummy(const fan::vec2& size);

  draw_list_t* get_window_draw_list();
  draw_list_t* get_foreground_draw_list();
  draw_list_t* get_background_draw_list();

  window_handle_t* get_current_window();

  bool combo(const std::string& label, int* current_item, const char* const items[], int items_count, int popup_max_height_in_items = -1);
  // Separate items with \0 within a string, end item-list with \0\0. e.g. "One\0Two\0Three\0"
  bool combo(const std::string& label, int* current_item, const char* items_separated_by_zeros, int popup_max_height_in_items = -1);
  bool combo(const std::string& label, int* current_item, const char* (*getter)(void* user_data, int idx), void* user_data, int items_count, int popup_max_height_in_items = -1);
  bool checkbox(const std::string& label, bool* v);
  bool list_box(const std::string &label, int* current_item, bool (*old_callback)(void* user_data, int idx, const char** out_text), void* user_data, int items_count, int height_in_items = -1);
  bool list_box(const std::string& label, int* current_item, const char* const items[], int items_count, int height_in_items = -1);
  bool list_box(const std::string& label, int* current_item, const char* (*getter)(void* user_data, int idx), void* user_data, int items_count, int height_in_items = -1);

  bool toggle_button(const std::string& str, bool* v);

  void text_bottom_right(const std::string& text, uint32_t reverse_yoffset = 0);

  fan::vec2 get_position_bottom_corner(const std::string& text = "", uint32_t reverse_yoffset = 0);

  void send_drag_drop_item(const std::string& id, const std::wstring& path, const std::string& popup = "");
  void receive_drag_drop_target(const std::string& id, std::function<void(std::string)> receive_func);

  bool slider_scalar(const char* label, data_type_t data_type, void* p_data, const void* p_min, const void* p_max, const char* format = NULL, slider_flags_t flags = 0);
  bool slider_scalar_n(const char* label, data_type_t data_type, void* p_data, int components, const void* p_min, const void* p_max, const char* format = NULL, slider_flags_t flags = 0);

  bool drag_scalar(const char* label, data_type_t data_type, void* p_data, f32_t v_speed = 1.0f, const void* p_min = NULL, const void* p_max = NULL, const char* format = NULL, slider_flags_t flags = 0);
  bool drag_scalar_n(const char* label, data_type_t data_type, void* p_data, int components, f32_t v_speed = 1.0f, const void* p_min = NULL, const void* p_max = NULL, const char* format = NULL, slider_flags_t flags = 0); 

  font_t* get_font_impl(f32_t font_size, bool bold = false);
  font_t* get_font(
    font_t* (&fonts)[std::size(fan::graphics::gui::font_sizes)],
    f32_t font_size
  );
  font_t* get_font(f32_t font_size, bool bold = false);

  void image(texture_id_t texture, const fan::vec2& size, const fan::vec2& uv0 = fan::vec2(0, 0), const fan::vec2& uv1 = fan::vec2(1, 1), const fan::color& tint_col = fan::colors::white, const fan::color& border_col = fan::color(0, 0, 0, 0));
  bool image_button(const std::string& str_id, texture_id_t texture, const fan::vec2& size, const fan::vec2& uv0 = fan::vec2(0, 0), const fan::vec2& uv1 = fan::vec2(1, 1), const fan::color& bg_col = fan::color(0, 0, 0, 0), const fan::color& tint_col = fan::colors::white);

  bool item_add(const rect_t& bb, id_t id, const rect_t* nav_bb = NULL, item_flags_t extra_flags = 0);


  int is_key_pressed(key_t key, bool repeat = true);
  int get_pressed_key();

  void set_next_window_class(const class_t* c);

  // TODO flags
  bool begin_drag_drop_source();
  bool set_drag_drop_payload(const std::string& type, const void* data, size_t sz, cond_t cond = cond_none);
  void end_drag_drop_source();

  bool begin_drag_drop_target();
  const payload_t* accept_drag_drop_payload(const std::string& type);
  void end_drag_drop_target();
  const payload_t* get_drag_drop_payload();

  bool begin_popup(const std::string& id, window_flags_t flags = 0);
  bool begin_popup_modal(const std::string& id, window_flags_t flags = 0);
  void end_popup();
  void open_popup(const std::string& id);
  void close_current_popup();
  bool is_popup_open(const std::string& id);

  id_t get_id(const std::string& str_id);
  storage_t* get_state_storage();

  f32_t get_line_height_with_spacing();

  void seperator();
  void dock_space_over_viewport(id_t dockspace_id = 0, const gui::viewport_t* viewport = NULL, int flags = 0, const void* window_class = NULL);

  context_t* get_context();

  inline bool g_gui_initialized = false;
  void init(
    GLFWwindow* window,
    int renderer,
    int opengl_renderer_definition,  // todo bad
    int vulkan_renderer_definition  //  todo bad
  #if defined(fan_vulkan)
    , VkInstance instance,
    VkPhysicalDevice physical_device,
    VkDevice device,
    uint32_t queue_family,
    VkQueue graphics_queue,
    VkDescriptorPool descriptor_pool,
    VkRenderPass render_pass,
    uint32_t image_count,
    uint32_t min_image_count,
    VkSampleCountFlagBits msaa_samples,
    void (*check_vk_result)(VkResult)
  #endif
  );
  void init_graphics_context(
    GLFWwindow* window,
    int renderer,
    int opengl_renderer_definition,  // todo bad
    int vulkan_renderer_definition  //  todo bad
  #if defined(fan_vulkan)
    , VkInstance instance,
    VkPhysicalDevice physical_device,
    VkDevice device,
    uint32_t queue_family,
    VkQueue graphics_queue,
    VkDescriptorPool descriptor_pool,
    VkRenderPass render_pass,
    uint32_t image_count,
    uint32_t min_image_count,
    VkSampleCountFlagBits msaa_samples,
    void (*check_vk_result)(VkResult)
  #endif
  );
  void init_fonts();
  void load_emojis();

  void shutdown_graphics_context(
    int renderer,
    int opengl_renderer_definition,  // todo bad
    int vulkan_renderer_definition  //  todo bad
  #if defined(fan_vulkan)
    , VkDevice device
  #endif
  );
  void shutdown_window_context();
  void destroy();

  void new_frame(
    int renderer,
    int opengl_renderer_definition,  // todo bad
    int vulkan_renderer_definition  //  todo bad
  );

#if defined(fan_vulkan)
  typedef void (*ImGuiFrameRenderFunc)(void* context, VkResult, fan::color);
#endif

  void render(
    int renderer,
    int opengl_renderer_definition,  // todo bad
    int vulkan_renderer_definition,  //  todo bad
    bool render_shapes_top
  #if defined(fan_vulkan)
    ,
    void* context,
    const fan::color& clear_color,
    VkResult& image_error,
    VkCommandBuffer& cmd_buffer,
    ImGuiFrameRenderFunc render_func
  #endif
  );


  void profile_heap(void* (*dynamic_malloc)(size_t, void*), void (*dynamic_free)(void*, void*));
} // namespace fan::graphics::gui


  // plot
export namespace fan::graphics::gui::plot {
  bool begin_plot(const std::string& title, const fan::vec2& size = fan::vec2(-1, 0), flags_t flags = flags_none);
  void end_plot();
  void setup_axes(const std::string& x_label, const std::string& y_label,
    axis_flags_t x_flags = axis_flags_none, axis_flags_t y_flags = axis_flags_none);
  void setup_axis(axis_t axis, const std::string& label = "", axis_flags_t flags = axis_flags_none);
  void setup_axis_limits(axis_t axis, double v_min, double v_max, cond_t cond = cond_once);
  void setup_axes_limits(double x_min, double x_max, double y_min, double y_max, cond_t cond = cond_once);
  void setup_axis_format(axis_t idx, const std::string& format);
  void setup_axis_links(axis_t idx, double* min_lnk, double* max_lnk);
  void setup_axis_format(axis_t idx, formatter_t formatter, void* data);
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

  fan::vec2 pixels_to_plot(const fan::vec2& pix, axis_t x_axis = plot_auto, axis_t y_axis = plot_auto);
  fan::vec2 plot_to_pixels(double x, double y, axis_t x_axis = plot_auto, axis_t y_axis = plot_auto);

  fan::vec2 get_plot_mouse_pos(axis_t x_axis = plot_auto, axis_t y_axis = plot_auto);

  void annotation(double x, double y, const fan::color& col, const fan::vec2& pix_offset, bool clamp, const std::string& text);

  void tag_x(double x, const fan::color& col, const std::string& text = "");
  void tag_y(double y, const fan::color& col, const std::string& text = "");

  void plot_text(const std::string& text, double x, double y, const fan::vec2& pix_offset = fan::vec2(0, 0), int flags = 0);
  void plot_dummy(const std::string& label_id, int flags = 0);

  fan::color next_colormap_color();
  fan::color get_last_item_color();

  void setup_axis_ticks(axis_t axis, const double* values, int n_ticks, const char* const labels[]=nullptr, bool keep_default=false);
  void setup_axis_ticks(axis_t axis, double v_min, double v_max, int n_ticks, const char* const labels[]=nullptr, bool keep_default=false);

  void push_plot_clip_rect(f32_t expand=0);
  void pop_plot_clip_rect();

} // namespace plot

#if defined (fan_audio)
export namespace fan::graphics::gui {

  /// <summary>
  /// Draws the specified button, with its position influenced by other GUI elements.
  /// Plays default hover and click audio piece if none specified.
  /// </summary>
  /// <param name="label">Name of the button. Draws the given label next to the button. The label is hideable using "##hidden_label".</param>
  /// <param name="piece_hover">Audio piece that is played when hovering the button.</param>
  /// <param name="piece_click">Audio piece that is played when clicking and releasing the button.</param>
  /// <param name="size">Size of the button (defaults to automatic).</param>
  /// <returns></returns>
  bool audio_button(
    const std::string& label, 
    fan::audio::piece_t piece_hover = fan::audio::piece_invalid, 
    fan::audio::piece_t piece_click = fan::audio::piece_invalid, 
    const fan::vec2& size = fan::vec2(0, 0)
  );
}
#endif

// templates
export namespace fan::graphics::gui {
  template<typename T>
  constexpr const char* get_default_format() {
    if constexpr (std::is_integral_v<T>) {
      if constexpr (std::is_signed_v<T>) return "%d";
      else return "%u";
    }
    else return "%.3f";
  }

  template<typename T>
  bool slider(const std::string& label, T* v, auto v_min, auto v_max, slider_flags_t flags = 0) {
    if constexpr (get_component_count<T>() == 1) {
      T min_val = static_cast<T>(v_min);
      T max_val = static_cast<T>(v_max);
      return slider_scalar(label.c_str(), get_imgui_data_type<T>(), v, &min_val, &max_val, get_default_format<T>(), flags);
    }
    else {
      using component_type = component_type_t<T>;
      component_type min_val = static_cast<component_type>(v_min);
      component_type max_val = static_cast<component_type>(v_max);
      return slider_scalar_n(label.c_str(), get_imgui_data_type<component_type>(), v->data(), get_component_count<T>(), &min_val, &max_val, get_default_format<component_type>(), flags);
    }
  }
  template<typename T>
  bool slider(const std::string& label, T* v, slider_flags_t flags = 0) {
    if constexpr (std::is_integral_v<T> || std::is_integral_v<component_type_t<T>>) {
      return slider(label, v, 0, 100, flags);
    }
    else {
      return slider(label, v, 0.0, 1.0, flags);
    }
  }


  template<typename T>
  bool drag(const std::string& label, T* v, f32_t v_speed = 1.f, f32_t v_min = 0, f32_t v_max = 0, slider_flags_t flags = 0) {
    if constexpr (get_component_count<T>() == 1) {
      T min_val, max_val;

      if constexpr (std::is_floating_point_v<T>) {
        min_val = static_cast<T>(v_min);
        max_val = static_cast<T>(v_max);
      }
      else {
        constexpr f32_t safe_min = static_cast<f32_t>(std::numeric_limits<T>::min());
        constexpr f32_t safe_max = static_cast<f32_t>(std::numeric_limits<T>::max());
        min_val = (v_min <= safe_min) ? std::numeric_limits<T>::min() : (v_min >= safe_max) ? std::numeric_limits<T>::max() : static_cast<T>(v_min);
        max_val = (v_max <= safe_min) ? std::numeric_limits<T>::min() : (v_max >= safe_max) ? std::numeric_limits<T>::max() : static_cast<T>(v_max);
      }

      if (min_val > max_val) {
        std::swap(min_val, max_val);
      }

      return drag_scalar(label.c_str(), get_imgui_data_type<T>(), v, v_speed, &min_val, &max_val, get_default_format<T>(), flags);
    }
    else {
      using component_type = component_type_t<T>;
      component_type speed_val = static_cast<component_type>(v_speed);
      component_type min_val, max_val;

      if constexpr (std::is_floating_point_v<component_type>) {
        min_val = static_cast<component_type>(v_min);
        max_val = static_cast<component_type>(v_max);
      }
      else {
        constexpr f32_t safe_min = static_cast<f32_t>(std::numeric_limits<component_type>::min());
        constexpr f32_t safe_max = static_cast<f32_t>(std::numeric_limits<component_type>::max());
        min_val = (v_min <= safe_min) ? std::numeric_limits<component_type>::min() : (v_min >= safe_max) ? std::numeric_limits<component_type>::max() : static_cast<component_type>(v_min);
        max_val = (v_max <= safe_min) ? std::numeric_limits<component_type>::min() : (v_max >= safe_max) ? std::numeric_limits<component_type>::max() : static_cast<component_type>(v_max);
      }

      if (min_val > max_val) {
        std::swap(min_val, max_val);
      }

      return drag_scalar_n(label.c_str(), get_imgui_data_type<component_type>(), v->data(), get_component_count<T>(), speed_val, &min_val, &max_val, get_default_format<component_type>(), flags);
    }
  }

  namespace plot {
    template <typename T>
    void plot_bars(const std::string& label_id, const T* values, int count, double bar_size = 0.67, double shift = 0, bars_flags_t flags = 0, int offset = 0, int stride = sizeof(T)) {
      ImPlot::PlotBars<T>(label_id.c_str(), values, count, bar_size, shift, flags, offset, stride);
    }
    template <typename T>
    void plot_bars(const std::string& label_id, const T* xs, const T* ys, int count, double bar_size, bars_flags_t flags = 0, int offset = 0, int stride = sizeof(T)) {
      ImPlot::PlotBars<T>(label_id.c_str(), xs, ys, count, bar_size, flags, offset, stride);
    }

    template <typename T>
    void plot_line(const std::string& label_id, const T* values, int count, double xscale = 1, double xstart = 0, int flags = 0, int offset = 0, int stride = sizeof(T)) {
      ImPlot::PlotLine(label_id.c_str(), values, count, xscale, xstart, flags, offset, stride);
    }
    template <typename T>
    void plot_line(const std::string& label_id, const T* xs, const T* ys, int count, int flags = 0, int offset = 0, int stride = sizeof(T)) {
      ImPlot::PlotLine(label_id.c_str(), xs, ys, count, flags, offset, stride);
    }
  }
}

#endif