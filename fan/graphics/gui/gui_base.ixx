module;

#if defined(FAN_GUI)
  #include <fan/imgui/imgui.h>
  #include <fan/imgui/imgui_impl_glfw.h>
  #include <fan/imgui/implot.h>
  #include <fan/imgui/imgui_internal.h>
  #include <fan/imgui/ImGuizmo.h>
#endif
#if defined(FAN_VULKAN)
  #include <vulkan/vulkan.h>
#endif

#include <fan/utility.h>

export module fan.graphics.gui.base;

import std;

#if defined(FAN_GUI)

#if defined (FAN_AUDIO)
  import fan.audio;
#endif

import fan.types;
import fan.graphics.gui.types;
import fan.types.vector;
import fan.types.color;
import fan.types.matrix;
import fan.types.compile_time_string;
import fan.utility;
import fan.math;
import fan.formatter;
import fan.print;
import fan.graphics.common_context;

export namespace fan::graphics::gui {

  inline constexpr std::size_t default_font_size_index = 9;
  inline constexpr f32_t font_sizes[] = {
    4, 5, 6, 7, 8, 9, 10, 11, 12, 14,
    16, 18, 20, 22, 24, 28,
    32, 36, 48, 60, 72
  };

  namespace font {
    enum type_t { regular, bold, mono, count };
  }

  inline auto& get_font_paths() {
    static const char* paths[font::count] = {
      "fonts/Inter/Inter_18pt-Regular.ttf",
      "fonts/Inter/Inter_18pt-Bold.ttf",
      "fonts/JetBrainsMono/JetBrainsMono-Regular.ttf"
    };
    return paths;
  }

  topmost_window_data_t& topmost_data();
  void enforce_topmost();

  // for print
  inline std::ostream& operator<<(std::ostream& os, const str_view_t& label) {
    return os << static_cast<std::string_view>(label);
  }
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
    else static_assert(!sizeof(T), "Unsupported type for ImGui");
  }

  void push_style_color(col_t index, const fan::color& col);
  void pop_style_color(int n = 1);

  void push_style_var(style_var_t index, f32_t val);
  void push_style_var(style_var_t index, const fan::vec2& val);
  void pop_style_var(int n = 1);

  bool button(str_view_t label, const fan::vec2& size = fan::vec2(0, 0));
  bool button(str_view_t label, const fan::vec2& size, f32_t font_size, font::type_t type = font::regular);
  struct button_centered_args_t {
    fan::vec2 size = fan::vec2(0, 0);
    fan::vec2i affects_axis = fan::vec2i(1, 1);
    fan::vec2 offset = 0.f;
  };
  bool button_centered(str_view_t label, const button_centered_args_t& args = {});


  bool button_fill(str_view_t label);

  f32_t calc_button_width(str_view_t label);

  bool invisible_button(str_view_t label, const fan::vec2& size = fan::vec2(0, 0));
  bool invisible_button(const fan::vec2& size = fan::vec2(0, 0));

  bool arrow_button(str_view_t label, dir_t dir);
  
  struct text_style_t {
    fan::color color = fan::colors::white;
    fan::color outline_color = fan::colors::black;
    f32_t font_size = 0.f;
    fan::vec2 pos = std::numeric_limits<f32_t>::max();
    fan::vec2 offset = 0.f;
    fan::vec2 text_offset = 0.f;
    fan::vec2 window_offset = std::numeric_limits<f32_t>::max();
    enum class align_e { left, center, bottom_right } align = align_e::left;
    bool outlined = false;
    bool wrapped = false;
    bool overlay = false;
  };

  /// <summary>
  /// Draws the specified text, with its position influenced by other GUI elements.
  /// </summary>
  /// <param name="color">The color of the text.</param>
  /// <param name="text">The text args to draw.</param>
  template <typename first_t, typename ...args_t>
  requires (!std::is_same_v<std::decay_t<first_t>, text_style_t>)
  void text(const fan::color& color, const first_t& first, const args_t&... args) {
    gui::push_style_color(col_text, color);
    std::string txt = fan::format_args(first, args...);
    ImGui::TextUnformatted(txt.c_str());
    gui::pop_style_color();
  }

  /// <summary>
  /// Draws text constructed from multiple arguments with default white color.
  /// </summary>
  /// <param name="args">Arguments to be concatenated with spaces.</param>
  template <typename first_t, typename ...args_t>
  requires (!std::is_same_v<std::decay_t<first_t>, text_style_t> &&
            !std::is_same_v<std::decay_t<first_t>, fan::color> &&
            (!std::is_same_v<std::decay_t<args_t>, text_style_t> && ...))
  void text(const first_t& first, const args_t&... args) {
    text(fan::colors::white, first, args...);
  }

  void text_disabled(std::string_view text);

  void text(std::string_view str, const text_style_t& style = {});
  void text_outlined(std::string_view str, text_style_t style = {});
  void text_wrapped(std::string_view str, text_style_t style = {});

  template <typename... args_t>
  void text(const text_style_t& style, const args_t&... args) {
    text(fan::format_args(args...), style);
  }

  template <typename... args_t>
  void text_outlined(const text_style_t& style, const args_t&... args) {
    text_outlined(fan::format_args(args...), style);
  }

  void text_box(
    std::string_view text,
    const fan::vec2& size = fan::vec2(0, 0),
    const fan::color& text_color = fan::colors::white,
    const fan::color& bg_color = fan::color()
  );

  void text_box_at(
    std::string_view text,
    const fan::vec2& pos,
    const fan::vec2& size = fan::vec2(0, 0),
    const fan::color& text_color = fan::colors::white,
    const fan::color& bg_color = fan::color()
  );

  f32_t calc_item_width();
  f32_t get_item_width();

  bool input_text(str_view_t label, std::string* buf, input_text_flags_t flags = 0, input_text_callback_t callback = nullptr, void* user_data = nullptr);
  bool input_text(std::string* buf, input_text_flags_t flags = 0, input_text_callback_t callback = nullptr, void* user_data = nullptr);

  bool input_text_multiline(str_view_t label, std::string* buf, const fan::vec2& size = fan::vec2(0, 0), input_text_flags_t flags = 0, input_text_callback_t callback = nullptr, void* user_data = nullptr);
  bool input_text_multiline(std::string* buf, const fan::vec2& size = fan::vec2(0, 0), input_text_flags_t flags = 0, input_text_callback_t callback = nullptr, void* user_data = nullptr);

  bool input_float(str_view_t label, f32_t* v, f32_t step = 0.0f, f32_t step_fast = 0.0f, const char* format = "%.3f", input_text_flags_t flags = 0);
  bool input_float(str_view_t label, fan::vec2* v, const char* format = "%.3f", input_text_flags_t flags = 0);
  bool input_float(str_view_t label, fan::vec3* v, const char* format = "%.3f", input_text_flags_t flags = 0);
  bool input_float(str_view_t label, fan::vec4* v, const char* format = "%.3f", input_text_flags_t flags = 0);
  bool input_float(f32_t* v, f32_t step = 0.0f, f32_t step_fast = 0.0f, const char* format = "%.3f", input_text_flags_t flags = 0);
  bool input_float(fan::vec2* v, const char* format = "%.3f", input_text_flags_t flags = 0);
  bool input_float(fan::vec3* v, const char* format = "%.3f", input_text_flags_t flags = 0);
  bool input_float(fan::vec4* v, const char* format = "%.3f", input_text_flags_t flags = 0);

  bool input_int(str_view_t label, int* v, int step = 1, int step_fast = 100, input_text_flags_t flags = 0);
  bool input_int(str_view_t label, fan::vec2i* v, input_text_flags_t flags = 0);
  bool input_int(str_view_t label, fan::vec3i* v, input_text_flags_t flags = 0);
  bool input_int(str_view_t label, fan::vec4i* v, input_text_flags_t flags = 0);
  bool input_int(int* v, int step = 1, int step_fast = 100, input_text_flags_t flags = 0);
  bool input_int(fan::vec2i* v, input_text_flags_t flags = 0);
  bool input_int(fan::vec3i* v, input_text_flags_t flags = 0);
  bool input_int(fan::vec4i* v, input_text_flags_t flags = 0);

  bool color_edit3(str_view_t label, fan::color* color, color_edit_flags_t flags = 0);
  bool color_edit3(str_view_t label, fan::vec3* color, color_edit_flags_t flags = 0);
  bool color_edit3(fan::color* color, color_edit_flags_t flags = 0);
  bool color_edit3(fan::vec3* color, color_edit_flags_t flags = 0);
  bool color_edit4(str_view_t label, fan::color* color, color_edit_flags_t flags = 0);
  bool color_edit4(fan::color* color, color_edit_flags_t flags = 0);

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
  fan::vec2 get_frame_padding();

  fan::color get_color(col_t idx);
  std::uint32_t get_color_u32(col_t idx);

  void separator();
  void spacing();

  io_t& get_io();

  template <typename... Args>
  bool tree_node_ex(void* id, tree_node_flags_t flags, const char* fmt, Args&&... args) {
    return ImGui::TreeNodeEx(id, flags, fmt, std::forward<Args>(args)...);
  }

  bool tree_node_ex(str_view_t label, tree_node_flags_t flags = 0);

  void tree_pop();

  bool tree_node(str_view_t label);

  bool is_item_toggled_open();

  void dummy(const fan::vec2& size);

  void fill_width();
  void fill_width_except(str_view_t next_label);
  f32_t fill_width_between(f32_t left_w, f32_t right_w);
  f32_t calc_item_spacing_x();
  void spacing(f32_t px);

  draw_list_t* get_window_draw_list();
  draw_list_t* get_foreground_draw_list();
  draw_list_t* get_background_draw_list();

  window_data_t* get_current_window();
  // finds window by name
  window_data_t* find_window(const fan::str_view_t name);

  bool combo(str_view_t label, int* current_item, const char* const items[], int items_count, int popup_max_height_in_items = -1);

  
  bool combo(str_view_t label, int* current_item, const char* items_separated_by_zeros, int popup_max_height_in_items = -1);

  bool combo(str_view_t label, int* current_item, const char* (*getter)(void* user_data, int idx), void* user_data, int items_count, int popup_max_height_in_items = -1);

  bool combo(str_view_t id, int* current, int count, auto&& getter) {
    using F = std::decay_t<decltype(getter)>;
    return ImGui::Combo(id, current, [](void* d, int i, const char** out) -> bool {
      *out = (*static_cast<F*>(d))(i); return true;
    }, static_cast<void*>(&getter), count);
  }

  bool combo(int* current_item, const char* const items[], int items_count, int popup_max_height_in_items = -1);
  bool combo(int* current_item, const char* items_separated_by_zeros, int popup_max_height_in_items = -1);
  bool combo(int* current_item, const char* (*getter)(void* user_data, int idx), void* user_data, int items_count, int popup_max_height_in_items = -1);

  bool checkbox(str_view_t label, bool* v);
  bool checkbox(bool* v);

  bool list_box(str_view_t label, int* current_item, bool (*old_callback)(void* user_data, int idx, const char** out_text), void* user_data, int items_count, int height_in_items = -1);

  bool list_box(str_view_t label, int* current_item, const char* const items[], int items_count, int height_in_items = -1);

  bool list_box(str_view_t label, int* current_item, const char* (*getter)(void* user_data, int idx), void* user_data, int items_count, int height_in_items = -1);

  bool toggle_button(str_view_t label, bool* v);

  fan::vec2 get_position_bottom_corner(std::string_view text = {}, std::uint32_t reverse_yoffset = 0);

  void send_drag_drop_item(str_view_t id, const std::wstring& path, str_view_t popup = "");

  void receive_drag_drop_target(str_view_t id, std::function<void(std::string)> receive_func);

  bool slider_scalar(str_view_t label, data_type_t data_type, void* p_data, const void* p_min, const void* p_max, const char* format = NULL, slider_flags_t flags = 0);
  bool slider_scalar_n(str_view_t label, data_type_t data_type, void* p_data, int components, const void* p_min, const void* p_max, const char* format = NULL, slider_flags_t flags = 0);

  bool drag_scalar(str_view_t label, data_type_t data_type, void* p_data, f32_t v_speed = 1.0f, const void* p_min = NULL, const void* p_max = NULL, const char* format = NULL, slider_flags_t flags = 0);
  bool drag_scalar_n(str_view_t label, data_type_t data_type, void* p_data, int components, f32_t v_speed = 1.0f, const void* p_min = NULL, const void* p_max = NULL, const char* format = NULL, slider_flags_t flags = 0);

  struct font_scope_t {
    font_scope_t() = default;
    font_scope_t(f32_t size, font::type_t type = font::regular);
    font_scope_t& operator=(font_scope_t&& o) noexcept;
    ~font_scope_t();
    bool active = false;
  };
  
  struct crisp_font_scope_t {
    crisp_font_scope_t(f32_t base_size, f32_t ideal_scale, font::type_t type = font::regular);
    ~crisp_font_scope_t();
    f32_t actual_scale = 1.0f;
  };

  struct zoom_scope_t {
    zoom_scope_t(f32_t& zoom_factor, f32_t base_font_size, f32_t auto_scale, f32_t speed = 0.1f, f32_t user_max = 5.f);
  };

  void image(
    texture_id_t texture,
    const fan::vec2& size,
    const fan::vec2& uv0 = fan::vec2(0, 0),
    const fan::vec2& uv1 = fan::vec2(1, 1),
    const fan::color& tint_col = fan::colors::white,
    const fan::color& border_col = fan::color(0, 0, 0, 0)
  );

  void image(
    fan::graphics::image_t img,
    const fan::vec2& size,
    const fan::vec2& uv0 = fan::vec2(0, 0),
    const fan::vec2& uv1 = fan::vec2(1, 1),
    const fan::color& tint_col = fan::color(1, 1, 1, 1),
    const fan::color& border_col = fan::color(0, 0, 0, 0)
  );

  bool image_button(
    str_view_t str_id,
    fan::graphics::image_t img,
    const fan::vec2& size,
    const fan::vec2& uv0 = fan::vec2(0, 0),
    const fan::vec2& uv1 = fan::vec2(1, 1),
    int frame_padding = -1,
    const fan::color& bg_col = fan::color(0, 0, 0, 0),
    const fan::color& tint_col = fan::color(1, 1, 1, 1)
  );

  bool image_text_button(
    fan::graphics::image_t img,
    std::string_view text,
    const fan::color& color,
    const fan::vec2& size,
    const fan::vec2& uv0 = fan::vec2(0, 0),
    const fan::vec2& uv1 = fan::vec2(1, 1),
    int frame_padding = -1,
    const fan::color& bg_col = fan::color(0, 0, 0, 0),
    const fan::color& tint_col = fan::color(1, 1, 1, 1)
  );

  bool image_button(
    str_view_t str_id,
    texture_id_t texture,
    const fan::vec2& size,
    const fan::vec2& uv0 = fan::vec2(0, 0),
    const fan::vec2& uv1 = fan::vec2(1, 1),
    const fan::color& bg_col = fan::color(0, 0, 0, 0),
    const fan::color& tint_col = fan::colors::white
  );

  bool item_add(const rect_t& bb, id_t id, const rect_t* nav_bb = NULL, item_flags_t extra_flags = 0);

  int is_key_clicked(key_t key, bool repeat = true);
  int get_pressed_key();
  wcharacter_t get_char_pressed();

  void set_next_window_class(const class_t* c);

  bool begin_drag_drop_source();
  bool set_drag_drop_payload(str_view_t type, const void* data, std::size_t sz, cond_t cond = cond_none);

  void end_drag_drop_source();

  bool begin_drag_drop_target();

  const payload_t* accept_drag_drop_payload(str_view_t type);

  void end_drag_drop_target();

  const payload_t* get_drag_drop_payload();

  bool begin_popup(str_view_t id, window_flags_t flags = 0);

  bool begin_popup_modal(str_view_t id, window_flags_t flags = 0);

  void end_popup();

  void open_popup(str_view_t id);

  void close_current_popup();

  bool is_popup_open(str_view_t id);

  void begin_disabled(bool disabled = true);
  void end_disabled();

  id_t get_id(str_view_t str_id);
  id_t get_id(int id);

  storage_t* get_state_storage();

  f32_t get_line_height_with_spacing();

  void dock_space_over_viewport(id_t dockspace_id = 0, const gui::viewport_t* viewport = NULL, int flags = 0, const void* window_class = NULL);

  bool is_mouse_hovering_rect(const fan::vec2& min, const fan::vec2& max, bool clip = true);

  context_t* get_context();

  bool& is_gui_initialized() {
    static bool g_gui_initialized = false;
    return g_gui_initialized;
  }

  void init(
    GLFWwindow* window,
    int renderer,
    int opengl_renderer_definition,
    int vulkan_renderer_definition
  #if defined(FAN_VULKAN)
    , VkInstance instance,
    VkPhysicalDevice physical_device,
    VkDevice device,
    std::uint32_t queue_family,
    VkQueue graphics_queue,
    VkDescriptorPool descriptor_pool,
    VkRenderPass render_pass,
    std::uint32_t image_count,
    std::uint32_t min_image_count,
    VkSampleCountFlagBits msaa_samples,
    void (*check_vk_result)(VkResult)
  #endif
  );

  void init_graphics_context(
    GLFWwindow* window,
    int renderer,
    int opengl_renderer_definition,
    int vulkan_renderer_definition
  #if defined(FAN_VULKAN)
    , VkInstance instance,
    VkPhysicalDevice physical_device,
    VkDevice device,
    std::uint32_t queue_family,
    VkQueue graphics_queue,
    VkDescriptorPool descriptor_pool,
    VkRenderPass render_pass,
    std::uint32_t image_count,
    std::uint32_t min_image_count,
    VkSampleCountFlagBits msaa_samples,
    void (*check_vk_result)(VkResult)
  #endif
  );

  void init_fonts();
  void load_emojis();

  void shutdown_graphics_context(
    int renderer,
    int opengl_renderer_definition,
    int vulkan_renderer_definition
  #if defined(FAN_VULKAN)
    , VkDevice device
  #endif
  );

  void shutdown_window_context();
  void destroy();

  void new_frame(
    int renderer,
    int opengl_renderer_definition,
    int vulkan_renderer_definition
  );

#if defined(FAN_VULKAN)
  typedef void (*ImGuiFrameRenderFunc)(void* context, VkResult, fan::color);
#endif

  void render(
    int renderer,
    int opengl_renderer_definition,
    int vulkan_renderer_definition,
    bool render_shapes_top
  #if defined(FAN_VULKAN)
    ,
    void* context,
    const fan::color& clear_color,
    VkResult& image_error,
    VkCommandBuffer& cmd_buffer,
    ImGuiFrameRenderFunc render_func
  #endif
  );

  void profile_heap(void* (*dynamic_malloc)(std::size_t, void*), void (*dynamic_free)(void*, void*));

  bool begin(str_view_t window_name, bool* p_open = 0, window_flags_t window_flags = 0);
  void end();

  bool begin_child(str_view_t window_name, const fan::vec2& size = fan::vec2(0, 0), child_window_flags_t child_window_flags = 0, window_flags_t window_flags = 0);
  void end_child();

  bool begin_tab_item(str_view_t label, bool* p_open = 0, window_flags_t window_flags = 0);
  void end_tab_item();

  bool begin_tab_bar(str_view_t tab_bar_name, window_flags_t window_flags = 0);
  void end_tab_bar();

  bool begin_main_menu_bar();
  void end_main_menu_bar();

  bool begin_menu_bar();
  void end_menu_bar();

  bool begin_menu(str_view_t label, bool enabled = true);
  void end_menu();

  void begin_group();
  void end_group();

  void table_setup_column(str_view_t label, table_column_flags_t flags = 0, f32_t init_width_or_weight = 0.0f, id_t user_id = 0);

  void table_headers_row();
  bool table_set_column_index(int column_n);
  f32_t table_get_column_offset(int column_n = -1);
  f32_t table_get_cell_width(f32_t init_width = 0.f);

  bool collapsing_header(str_view_t label, bool* p_open = nullptr, tree_node_flags_t flags = 0);

  void push_clip_rect(const fan::vec2& min, const fan::vec2& max, bool intersect_with_current_clip_rect);
  void pop_clip_rect();

  bool button_behavior(const gui::rect_t& bb, gui::id_t id, bool* out_hovered, bool* out_held, int flags = 0);

  void rect_with_border(
    const fan::vec2& size = 0.f,
    const fan::color border_color = -1.f,
    f32_t border_thickness = 1.f,
    f32_t rounding = 0.f
  );

  gui::table_data_t* get_current_table();

  void show_debug_log_window(bool* p_open = nullptr);

  bool menu_item(str_view_t label, std::string_view shortcut = {}, bool selected = false, bool enabled = true);

  bool begin_combo(str_view_t label, std::string_view preview_value, int flags = 0);

  void end_combo();

  /// <summary>
  /// RAII containers for gui windows.
  /// </summary>
  struct window_t {
    window_t(str_view_t window_name, bool* p_open = 0, window_flags_t window_flags = 0);

    ~window_t();

    explicit operator bool() const;

    bool is_open;
  };

  /// <summary>
  /// RAII containers for gui child windows.
  /// </summary>
  struct child_window_t {
    child_window_t(str_view_t window_name, const fan::vec2& size = fan::vec2(0, 0), child_window_flags_t window_flags = 0);

    ~child_window_t();

    explicit operator bool() const;

    bool is_open;
  };

  /// <summary>
  /// RAII containers for gui tables.
  /// </summary>
  struct table_t {
    table_t(str_view_t str_id, int columns, table_flags_t flags = 0, const fan::vec2& outer_size = fan::vec2(0.0f, 0.0f), f32_t inner_width = 0.0f);

    ~table_t();

    explicit operator bool() const;

    bool is_open;
  };

  void set_item_default_focus();

  void same_line(f32_t offset_from_start_x = 0.f, f32_t spacing_w = -1.f);
  void new_line();
  void align_text_to_frame_padding();

  struct viewport_rect_t {
    fan::vec2 position;
    fan::vec2 size;
  };

  viewport_rect_t get_viewport_rect();
  viewport_t* get_main_viewport();

  f32_t get_frame_height();
  f32_t get_text_line_height_with_spacing();

  fan::vec2 get_mouse_pos();

  bool selectable(str_view_t label, bool selected = false, selectable_flag_t flags = 0, const fan::vec2& size = fan::vec2(0, 0));

  bool selectable(str_view_t label, bool* p_selected, selectable_flag_t flags = 0, const fan::vec2& size = fan::vec2(0, 0));
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

  void push_id(str_view_t str_id);
  void push_id(int int_id);
  void push_id(const void* ptr_id);

  void pop_id();

  void set_next_item_width(f32_t width);

  void push_text_wrap_pos(f32_t local_pos = 0);
  void pop_text_wrap_pos();

  bool is_item_hovered(hovered_flag_t flags = 0);
  bool is_any_item_hovered();
  bool is_any_item_active();
  bool is_item_clicked(int mouse_button = 0);
  bool is_item_held(int mouse_button = 0);

  void begin_tooltip();
  void end_tooltip();

  void set_tooltip(std::string_view tooltip);

  void tooltip_on_hover(std::string_view text, const fan::color& color = fan::colors::white);

  bool begin_table(str_view_t str_id, int columns, table_flags_t flags = 0, const fan::vec2& outer_size = fan::vec2(0.0f, 0.0f), f32_t inner_width = 0.0f);

  void end_table();

  void table_next_row(table_row_flags_t row_flags = 0, f32_t min_row_height = 0.0f);
  bool table_next_column();

  void columns(int count = 1, const char* id = nullptr, bool borders = true);

  void next_column();

  template <typename T>
  void table_cell(T&& v) {
    if constexpr (std::is_arithmetic_v<std::remove_cvref_t<T>>)
      gui::text(std::to_string(v));
    else
      gui::text(std::forward<T>(v));
  }

  template <typename... Columns>
  void table_row(Columns&&... cols) {
    gui::table_next_row();
    (( gui::table_next_column(), table_cell(std::forward<Columns>(cols)) ), ...);
  }

  void clear_active_id();

  void build_fonts();
  void rebuild_fonts();

  void push_font(font_t* font);
  void pop_font();

  void set_font(f32_t size);

  font_t* get_font();
  font_t* get_font(f32_t font_size, font::type_t type = font::regular);
  f32_t get_font_size();
  f32_t get_text_line_height();

  void indent(f32_t indent_w = 0.0f);
  void unindent(f32_t indent_w = 0.0f);

  fan::vec2 calc_text_size(std::string_view text, const char* text_end = NULL, bool hide_text_after_double_hash = false, f32_t wrap_width = -1.0f);

  fan::vec2 get_text_size(std::string_view text, const char* text_end = NULL, bool hide_text_after_double_hash = false, f32_t wrap_width = -1.0f);

  fan::vec2 text_size(std::string_view text, const char* text_end = NULL, bool hide_text_after_double_hash = false, f32_t wrap_width = -1.0f);

  fan::vec2 calc_input_size(std::string_view text);

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

  void set_window_focus(str_view_t name);

  int render_window_flags();

  fan::vec2 get_window_content_region_min();
  fan::vec2 get_window_content_region_max();

  f32_t get_column_width(int index = -1);
  void set_column_width(int index, f32_t width);

  bool is_item_active();

  bool& want_io();
  bool& force_want_io_for_frame();

  void set_want_io(
    bool flag = ImGui::GetIO().WantCaptureMouse |
      ImGui::GetIO().WantCaptureKeyboard |
      ImGui::GetIO().WantTextInput,
    bool op_or = false
  );

  void set_keyboard_focus_here(int offset = 0);

  fan::vec2 get_mouse_drag_delta(int button = 0, f32_t lock_threshold = -1.0f);
  void reset_mouse_drag_delta(int button = 0);

  void set_scroll_x(f32_t scroll_x);
  void set_scroll_y(f32_t scroll_y);

  void set_scroll_here_y();

  f32_t get_scroll_x();
  f32_t get_scroll_y();

  struct window {
    window(str_view_t title, fan::graphics::gui::window_flags_t flags = 0);
    window(str_view_t title, bool* p_open, fan::graphics::gui::window_flags_t flags = 0);

    explicit operator bool() const;

  private:
    fan::graphics::gui::window_t wnd;
  };

  struct child_window {
    child_window(str_view_t id, const fan::vec2& size = {0, 0}, bool border = false, window_flags_t flags = 0);
    ~child_window();

    explicit operator bool() const;
  private:
    bool visible;
  };

  struct hud {
    hud(str_view_t name, bool* p_open = 0);
    operator bool() const;
  private:
    window wnd;
  };

  struct hud_interactive {
    hud_interactive(str_view_t name, f32_t transparency = 0.f, bool* p_open = 0);
    operator bool() const;
  private:
    window wnd;
  };

  struct hud_interactive_custom {
    hud_interactive_custom(str_view_t name, f32_t transparency = 0.f, bool* p_open = 0);
    operator bool() const;
    private:
    window wnd;
  };

  struct table {
    table(str_view_t str_id, int columns,
      table_flags_t flags = 0,
      const fan::vec2& outer_size = fan::vec2(0.0f, 0.0f),
      f32_t inner_width = 0.0f);
    operator bool() const;
  private:
    table_t tbl;
  };
  
  gui::window fullscreen_window(str_view_t id, window_flags_t extra_flags = 0);

  gui::window centered_window(
    str_view_t id,
    fan::vec2 size,
    bool* open = nullptr,
    window_flags_t extra_flags = 0
  );

  gui::window overlay_window(str_view_t id, fan::vec2 size, f32_t alpha = 0.7f);

  struct scale_context_t {
    scale_context_t() = default;
    scale_context_t(const fan::vec2& current_size, const fan::vec2& design_size);

    f32_t scale(f32_t val) const;
    fan::vec2 scale(const fan::vec2& val) const;

    f32_t factor = 1.f;
  };

  struct style_scope_t {
    style_scope_t() = default;
    style_scope_t(gui::col_t idx, const fan::color& col);
    style_scope_t(gui::style_var_t idx, f32_t val);
    style_scope_t(gui::style_var_t idx, const fan::vec2& val);

    style_scope_t(const style_scope_t&) = delete;
    style_scope_t& operator=(const style_scope_t&) = delete;
    style_scope_t(style_scope_t&&) noexcept;
    style_scope_t& operator=(style_scope_t&&) noexcept;

    style_scope_t& color(gui::col_t idx, const fan::color& col);
    style_scope_t& var(gui::style_var_t idx, f32_t val);
    style_scope_t& var(gui::style_var_t idx, const fan::vec2& val);

    ~style_scope_t();

    int color_count = 0, var_count = 0;
  };

  style_scope_t make_invisible_input_style();

  struct scale_window_t {
    scale_window_t(str_view_t title, window_flags_t flags = 0);
    ~scale_window_t();

    explicit operator bool() const;

    window_t wnd;
    style_scope_t style_scope;
  };

  struct fit_window_t {
    fit_window_t(str_view_t title, const fan::vec2& design_size, window_flags_t flags = 0);
    ~fit_window_t();

    explicit operator bool() const;

    window_t wnd;
    style_scope_t style_scope;
  };

  template <typename container_t, typename mapper_t>
  bool combo_mapped(str_view_t label, int* current, const container_t& items, mapper_t mapper) {
    const char* labels[256];
    int count = 0;

    for (const auto& item : items) {
      labels[count++] = mapper(item);
    }

    return gui::combo(label, current, labels, count);
  }

  template<typename T>
  constexpr const char* get_default_format() {
    if constexpr (std::is_integral_v<T>) {
      if constexpr (std::is_signed_v<T>) {
        return "%d";
      }
      else {
        return "%u";
      }
    }
    else {
      return "%.3f";
    }
  }

  template<typename T>
  bool slider(str_view_t label, T* v, auto v_min, auto v_max, slider_flags_t flags = 0) {
    if constexpr (get_component_count<T>() == 1) {
      T min_val = static_cast<T>(v_min);
      T max_val = static_cast<T>(v_max);
      return slider_scalar(label, get_imgui_data_type<T>(), static_cast<void*>(v), &min_val, &max_val, get_default_format<T>(), flags);
    }
    else {
      using component_type = component_type_t<T>;
      component_type min_val = static_cast<component_type>(v_min);
      component_type max_val = static_cast<component_type>(v_max);
      return slider_scalar_n(label, get_imgui_data_type<component_type>(), static_cast<void*>(v->data()), get_component_count<T>(), &min_val, &max_val, get_default_format<component_type>(), flags);
    }
  }

  template<typename T>
  bool slider(str_view_t label, T* v, slider_flags_t flags = 0) {
    if constexpr (std::is_integral_v<T> || std::is_integral_v<component_type_t<T>>) {
      return slider(label, v, 0, 100, flags);
    }
    else {
      return slider(label, v, 0.0, 1.0, flags);
    }
  }

  template<typename T>
  requires (!std::convertible_to<T*, str_view_t>)
  bool slider(T* v, auto v_min, auto v_max, slider_flags_t flags = 0) {
    push_id(v);
    bool changed = slider("##", v, v_min, v_max, flags);
    pop_id();
    return changed;
  }

  template<typename T>
  requires (!std::convertible_to<T*, str_view_t>)
  bool slider(T* v, slider_flags_t flags = 0) {
    push_id(v);
    bool changed = slider("##", v, flags);
    pop_id();
    return changed;
  }

  template<typename T>
  bool drag(str_view_t label, T* v, f32_t v_speed = 0.1f, f32_t v_min = 0, f32_t v_max = 0, slider_flags_t flags = 0) {
    if constexpr (get_component_count<T>() == 1) {
      if (std::is_integral<T>::value) { v_speed = std::max(v_speed, 1.f); }
      T min_val;
      T max_val;

      if constexpr (std::is_floating_point_v<T>) {
        min_val = static_cast<T>(v_min);
        max_val = static_cast<T>(v_max);
      }
      else {
        constexpr f32_t safe_min = static_cast<f32_t>(std::numeric_limits<T>::min());
        constexpr f32_t safe_max = static_cast<f32_t>(std::numeric_limits<T>::max());

        min_val =
          (v_min <= safe_min) ? std::numeric_limits<T>::min() :
          (v_min >= safe_max) ? std::numeric_limits<T>::max() :
          static_cast<T>(v_min);

        max_val =
          (v_max <= safe_min) ? std::numeric_limits<T>::min() :
          (v_max >= safe_max) ? std::numeric_limits<T>::max() :
          static_cast<T>(v_max);
      }

      if (min_val > max_val) {
        std::swap(min_val, max_val);
      }

      return drag_scalar(label, get_imgui_data_type<T>(), v, v_speed, &min_val, &max_val, get_default_format<T>(), flags);
    }
    else {
      using component_type = component_type_t<T>;

      if (std::is_integral<component_type>::value) { v_speed = std::max(v_speed, 1.f); }
      component_type speed_val = static_cast<component_type>(v_speed);
      component_type min_val;
      component_type max_val;


      if constexpr (std::is_floating_point_v<component_type>) {
        min_val = static_cast<component_type>(v_min);
        max_val = static_cast<component_type>(v_max);
      }
      else {
        constexpr f32_t safe_min = static_cast<f32_t>(std::numeric_limits<component_type>::min());
        constexpr f32_t safe_max = static_cast<f32_t>(std::numeric_limits<component_type>::max());

        min_val =
          (v_min <= safe_min) ? std::numeric_limits<component_type>::min() :
          (v_min >= safe_max) ? std::numeric_limits<component_type>::max() :
          static_cast<component_type>(v_min);

        max_val =
          (v_max <= safe_min) ? std::numeric_limits<component_type>::min() :
          (v_max >= safe_max) ? std::numeric_limits<component_type>::max() :
          static_cast<component_type>(v_max);
      }

      if (min_val > max_val) {
        std::swap(min_val, max_val);
      }

      return drag_scalar_n(label, get_imgui_data_type<component_type>(), v->data(), get_component_count<T>(), speed_val, &min_val, &max_val, get_default_format<component_type>(), flags);
    }
  }
  template<typename T>
  bool drag(T* v, f32_t v_speed = 0.1f, f32_t v_min = 0, f32_t v_max = 0, slider_flags_t flags = 0) {
    push_id(v);
    bool changed = drag("##", v, v_speed, v_min, v_max, flags);
    pop_id();
    return changed;
  }
  template <typename T = f32_t, FAN_UNIQUE_CALL>
  gui::ret_t<T> drag(f32_t v_speed = 0.1f, f32_t v_min = 0, f32_t v_max = 0, slider_flags_t flags = 0) {
    static T val{};
    return gui::ret_t{val, drag(&val, v_speed, v_min, v_max, flags)};
  }
  std::optional<fan::str_view_t> button_grid(
    std::initializer_list<fan::str_view_t> labels,
    int columns,
    fan::vec2 size,
    f32_t font_size,
    font::type_t font_type = font::regular
  ) {
    font_scope_t fs(font_size, font_type);
    int i = 0;
    for (fan::str_view_t label : labels) {
      if (button(label, size)) {
        return label;
      }
      if (i != (int)labels.size() - 1 && (i + 1) % columns != 0)
        same_line();

      ++i;
    }
    return std::nullopt;
  }
  template <typename label_fn_t, typename enabled_fn_t, typename on_click_t>
  void button_row(std::ranges::range auto&& data, label_fn_t&& label_fn, enabled_fn_t&& enabled_fn, const fan::vec2& size, on_click_t&& on_click) {
    int i = 0;
    for (auto&& item : data) {
      if (i) { same_line(); }
      bool enabled = enabled_fn(item);
      if (!enabled) { begin_disabled(); }
      auto label = label_fn(item);
      if (button(label, size) && enabled) { on_click(i); }
      if (!enabled) { end_disabled(); }
      ++i;
    }
  }

  template <typename on_click_t>
  void button_row(std::ranges::range auto&& labels, const fan::vec2& size, f32_t font_size, on_click_t&& on_click) {
    int i = 0;
    for (auto&& lbl : labels) {
      if (i) { same_line(); }
      if (button(lbl, size, font_size)) { on_click(i, lbl); }
      ++i;
    }
  }

  template <typename on_click_t>
  void button_row(std::initializer_list<fan::str_view_t> labels, const fan::vec2& size, f32_t font_size, on_click_t&& on_click) {
    button_row(std::span(labels), size, font_size, std::forward<on_click_t>(on_click));
  }
  void healthbar(int value, int max, fan::vec2 size, const fan::color& fill = fan::colors::green, const fan::color& bg = fan::color(0.2f, 0.2f, 0.2f, 1.f));
  void healthbar_labeled(
    fan::str_view_t label,
    int value, int max,
    fan::vec2 size,
    const fan::color& label_color = fan::colors::white,
    const fan::color& fill = fan::colors::green,
    const fan::color& bg = fan::color(0.2f, 0.2f, 0.2f, 1.f)
  );

  void disabled_button_row(const std::string* labels, const bool* enabled, int count, fan::vec2 size, std::function<void(int)> on_click);
  void disabled_button_row(
    std::span<const fan::str_view_t> labels,
    std::span<const bool> enabled,
    fan::vec2 size,
    std::function<void(int)> on_click
  );

  void table_row_edit(const char* label, std::string& buf, auto on_enter) {
    gui::table_next_row();
    gui::table_next_column(); gui::text(label);
    gui::table_next_column(); gui::set_next_item_width(-1);
    gui::push_id(label);
    if (gui::input_text("##v", &buf, gui::input_text_flags_enter_returns_true))
      try { on_enter(); }
    catch (...) {}
    gui::pop_id();
  }

  void table_row_edit(const char* label, std::string& bufu, std::string& bufs, auto on_u, auto on_s) {
    gui::table_next_row();
    gui::table_next_column(); gui::text(label);
    gui::push_id(label);
    gui::table_next_column(); gui::set_next_item_width(-1);
    if (gui::input_text("##u", &bufu, gui::input_text_flags_enter_returns_true))
      try { on_u(); }
    catch (...) {}
    gui::table_next_column(); gui::set_next_item_width(-1);
    if (gui::input_text("##s", &bufs, gui::input_text_flags_enter_returns_true))
      try { on_s(); }
    catch (...) {}
    gui::pop_id();
  }

  void window_anchor_top_left(const fan::vec2& offset);
  void window_anchor_top_center(const fan::vec2& offset);
  void window_anchor_top_right(const fan::vec2& offset);
  void window_anchor_center_left(const fan::vec2& offset);
  void window_anchor_center(const fan::vec2& offset);
  void window_anchor_center_right(const fan::vec2& offset);
  void window_anchor_bottom_left(const fan::vec2& offset);
  void window_anchor_bottom_center(const fan::vec2& offset);
  void window_anchor_bottom_right(const fan::vec2& offset);

  void anchor_top_left(const fan::vec2& offset);
  void anchor_top_center(const fan::vec2& offset);
  void anchor_top_right(const fan::vec2& offset);
  void anchor_center_left(const fan::vec2& offset);
  void anchor_screen_center(const fan::vec2& offset);
  void anchor_center_right(const fan::vec2& offset);
  void anchor_bottom_left(const fan::vec2& offset);
  void anchor_bottom_center(const fan::vec2& offset);
  void anchor_bottom_right(const fan::vec2& offset);
  void anchor_center(const fan::vec2& window_size, const fan::vec2& item_size = 0.f, int item_count = 1);
  fan::vec2 get_display_size();

  void align_center_x(f32_t item_width);

  void window_move_title_bar_only();
} // namespace fan::graphics::gui

export namespace fan::graphics::gui::plot {

  bool begin_plot(str_view_t title, const fan::vec2& size = fan::vec2(-1, 0), flags_t flags = flags_none);

  void end_plot();

  void setup_axes(
    str_view_t x_label,
    str_view_t y_label,
    axis_flags_t x_flags = axis_flags_none,
    axis_flags_t y_flags = axis_flags_none
  );

  void setup_axis(axis_t axis, str_view_t label, axis_flags_t flags = axis_flags_none);

  void setup_axis_limits(axis_t axis, double v_min, double v_max, cond_t cond = cond_once);
  void setup_axes_limits(double x_min, double x_max, double y_min, double y_max, cond_t cond = cond_once);

  void setup_axis_format(axis_t idx, std::string_view format);

  void setup_axis_links(axis_t idx, double* min_lnk, double* max_lnk);

  void setup_axis_format(axis_t idx, formatter_t formatter, void* data);

  void setup_legend(location_t location, int flags = 0);
  void setup_finish();

  void plot_line(
    str_view_t label,
    const std::vector<f32_t>& values,
    double x_scale = 1.0,
    double x_start = 0.0,
    line_flags_t flags = line_flags_none
  );

  void plot_line(
    str_view_t label,
    const std::vector<f32_t>& xs,
    const std::vector<f32_t>& ys,
    line_flags_t flags = line_flags_none
  );

  void plot_line(
    str_view_t label,
    const f32_t* xs,
    const f32_t* ys,
    int count,
    line_flags_t flags = line_flags_none
  );

  void plot_scatter(
    str_view_t label,
    const std::vector<f32_t>& values,
    double x_scale = 1.0,
    double x_start = 0.0,
    scatter_flags_t flags = scatter_flags_none
  );

  void plot_scatter(
    str_view_t label,
    const std::vector<f32_t>& xs,
    const std::vector<f32_t>& ys,
    scatter_flags_t flags = scatter_flags_none
  );

  void plot_scatter(
    str_view_t label,
    const f32_t* xs,
    const f32_t* ys,
    int count,
    scatter_flags_t flags = scatter_flags_none
  );

  void plot_bars(
    str_view_t label,
    const std::vector<f32_t>& values,
    double bar_size = 0.67,
    double shift = 0,
    bars_flags_t flags = bars_flags_none
  );

  void plot_bars(
    str_view_t label,
    const std::vector<f32_t>& xs,
    const std::vector<f32_t>& ys,
    double bar_size,
    bars_flags_t flags = bars_flags_none
  );

  void plot_shaded(
    str_view_t label,
    const std::vector<f32_t>& xs,
    const std::vector<f32_t>& ys,
    double y_ref = 0.0,
    int flags = 0
  );

  void push_style_color(col_t idx, const fan::color& color);
  void pop_style_color(int count = 1);

  void push_style_var(int idx, f32_t val);
  void push_style_var(int idx, const fan::vec2& val);
  void pop_style_var(int count = 1);

  void set_next_line_style(const fan::color& col = fan::color(0, 0, 0, -1), f32_t weight = -1.0f);
  void set_next_fill_style(const fan::color& col = fan::color(0, 0, 0, -1), f32_t alpha_mod = -1.0f);

  void set_next_marker_style(
    marker_t marker = -1,
    f32_t size = -1.0f,
    const fan::color& fill = fan::color(0, 0, 0, -1),
    f32_t weight = -1.0f,
    const fan::color& outline = fan::color(0, 0, 0, -1)
  );

  fan::vec2 get_plot_pos();
  fan::vec2 get_plot_size();

  bool is_plot_hovered();
  bool is_axis_hovered(axis_t axis);

  fan::vec2 pixels_to_plot(const fan::vec2& pix, axis_t x_axis = plot_auto, axis_t y_axis = plot_auto);
  fan::vec2 plot_to_pixels(double x, double y, axis_t x_axis = plot_auto, axis_t y_axis = plot_auto);

  fan::vec2 get_plot_mouse_pos(axis_t x_axis = plot_auto, axis_t y_axis = plot_auto);

  void annotation(
    double x,
    double y,
    const fan::color& col,
    const fan::vec2& pix_offset,
    bool clamp,
    std::string_view text
  );

  void tag_x(double x, const fan::color& col, std::string_view text = {});

  void tag_y(double y, const fan::color& col, std::string_view text = {});

  void plot_text(std::string_view text, double x, double y, const fan::vec2& pix_offset = fan::vec2(0, 0), int flags = 0);

  void plot_dummy(str_view_t label_id, int flags = 0);

  fan::color next_colormap_color();
  fan::color get_last_item_color();

  void setup_axis_ticks(axis_t axis, const double* values, int n_ticks, const char* const labels[] = nullptr, bool keep_default = false);
  void setup_axis_ticks(axis_t axis, double v_min, double v_max, int n_ticks, const char* const labels[] = nullptr, bool keep_default = false);

  void push_plot_clip_rect(f32_t expand = 0);
  void pop_plot_clip_rect();

  template <typename T>
  void plot_bars(str_view_t label_id, const T* values, int count, double bar_size = 0.67, double shift = 0, bars_flags_t flags = 0, int offset = 0, int stride = sizeof(T)) {
    ImPlot::PlotBars<T>(label_id, values, count, bar_size, shift, flags, offset, stride);
  }

  template <typename T>
  void plot_bars(str_view_t label_id, const T* xs, const T* ys, int count, double bar_size, bars_flags_t flags = 0, int offset = 0, int stride = sizeof(T)) {
    ImPlot::PlotBars<T>(label_id, xs, ys, count, bar_size, flags, offset, stride);
  }

  template <typename T>
  void plot_line(str_view_t label_id, const T* values, int count, double xscale = 1, double xstart = 0, int flags = 0, int offset = 0, int stride = sizeof(T)) {
    ImPlot::PlotLine(label_id, values, count, xscale, xstart, flags, offset, stride);
  }

  template <typename T>
  void plot_line(str_view_t label_id, const T* xs, const T* ys, int count, int flags = 0, int offset = 0, int stride = sizeof(T)) {
    ImPlot::PlotLine(label_id, xs, ys, count, flags, offset, stride);
  }

} // namespace fan::graphics::gui::plot

#if defined (FAN_AUDIO)
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
    str_view_t label,
    fan::audio::piece_t piece_hover = fan::audio::piece_invalid,
    fan::audio::piece_t piece_click = fan::audio::piece_invalid,
    const fan::vec2& size = fan::vec2(0, 0)
  );
} // namespace fan::graphics::gui
#endif

export namespace fan::graphics::gui::gizmo {

  struct operation {
    enum {
      translate = ImGuizmo::TRANSLATE,
      rotate = ImGuizmo::ROTATE,
      scale = ImGuizmo::SCALE,
      bounds = ImGuizmo::BOUNDS
    };
  };

  struct mode {
    enum {
      local = ImGuizmo::LOCAL,
      world = ImGuizmo::WORLD
    };
  };

  void begin_frame();
  void set_orthographic(bool ortho);
  void set_drawlist();

  void set_rect(const fan::vec2& pos, const fan::vec2& size);

  bool manipulate(
    const fan::mat4& view,
    const fan::mat4& projection,
    int op,
    int m,
    fan::mat4& transform,
    const fan::mat4* delta = nullptr,
    const fan::mat4* snap = nullptr,
    const fan::mat4* bounds = nullptr,
    const fan::mat4* bounds_snap = nullptr
  );

  bool is_using();
  bool is_over();
  bool is_using_any();

  void draw_grid(
    const fan::mat4& view,
    const fan::mat4& projection,
    const fan::mat4& matrix,
    float size
  );

} // namespace fan::graphics::gui::gizmo

export namespace fan::graphics::gui::slot {

  void background(
    gui::draw_list_t* dl,
    const fan::vec2& p_min,
    const fan::vec2& p_max,
    const fan::color& color,
    f32_t rounding
  );

  void border(
    gui::draw_list_t* dl,
    const fan::vec2& p_min,
    const fan::vec2& p_max,
    const fan::color& color,
    f32_t rounding,
    f32_t thickness
  );

  void selected_border(
    gui::draw_list_t* dl,
    const fan::vec2& p_min,
    const fan::vec2& p_max,
    const fan::color& color,
    f32_t thickness,
    f32_t expand = 3.0f
  );

  void icon(
    const fan::graphics::image_t& img,
    const fan::vec2& p_min,
    const fan::vec2& p_max,
    const fan::vec2& padding
  );

  void stack_count(
    std::uint32_t count,
    const fan::vec2& p_min,
    const fan::vec2& p_max
  );

  void tooltip(std::string_view text, bool show);

} // namespace fan::graphics::gui::slot

#endif