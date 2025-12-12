module;

#if defined(fan_gui)
  #include <fan/imgui/imgui.h>
  #include <fan/imgui/imgui_internal.h>
  #include <fan/imgui/imgui_impl_glfw.h>
  #include <fan/imgui/implot.h>
  #include <fan/graphics/gui/imgui_themes.h>

  #include <fan/imgui/misc/freetype/imgui_freetype.h>
  #include <fan/imgui/imgui_impl_opengl3.h>
  #if defined(fan_vulkan)
  #include <fan/imgui/imgui_impl_vulkan.h>
  #endif

  #include <string>
  #include <functional>
  #include <cmath>

  #define GLFW_INCLUDE_NONE
  #include <GLFW/glfw3.h>
#endif

module fan.graphics.gui.base;

import fan.utility;

//imgui_stdlib.cpp:
static int InputTextCallback(ImGuiInputTextCallbackData* data) {
  using namespace fan::graphics::gui;
  InputTextCallback_UserData* user_data = (InputTextCallback_UserData*)data->UserData;
  if ( data->EventFlag == ImGuiInputTextFlags_CallbackResize ) {
    // Resize string callback
    // If for some reason we refuse the new length (BufTextLen) and/or capacity (BufSize) we need to set them back to what we want.
    std::string* str = user_data->Str;
    IM_ASSERT(data->Buf == str->c_str());
    str->resize(data->BufTextLen);
    data->Buf = (char*)str->c_str();
  }
  else if ( user_data->ChainCallback ) {
    // Forward to user callback, if any
    data->UserData = user_data->ChainCallbackUserData;
    return user_data->ChainCallback(data);
  }
  return 0;
}

#if defined(fan_gui)
namespace fan::graphics::gui {
  std::unordered_map<std::string, bool> want_io_ignore_list;
  bool begin(const std::string& window_name, bool* p_open, window_flags_t window_flags) {

    if (window_flags & window_flags_no_title_bar) {
      ImGuiWindowClass window_class;
      window_class.DockNodeFlagsOverrideSet = ImGuiDockNodeFlags_NoTabBar;
      ImGui::SetNextWindowClass(&window_class);
    }
    if (window_flags & window_flags_override_input) {
      want_io_ignore_list[window_name] = true;
    }

    return ImGui::Begin(window_name.c_str(), p_open, window_flags);
  }
  void end() {
    ImGui::End();
  }
  bool begin_child(const std::string& window_name, const fan::vec2& size, child_window_flags_t child_window_flags, window_flags_t window_flags) {
    return ImGui::BeginChild(window_name.c_str(), size, child_window_flags, window_flags);
  }
  void end_child() {
    ImGui::EndChild();
  }
  bool begin_tab_item(const std::string& label, bool* p_open, window_flags_t window_flags) {
    return ImGui::BeginTabItem(label.c_str(), p_open, window_flags);
  }
  void end_tab_item() {
    ImGui::EndTabItem();
  }
  bool begin_tab_bar(const std::string& tab_bar_name, window_flags_t window_flags) {
    return ImGui::BeginTabBar(tab_bar_name.c_str(), window_flags);
  }
  void end_tab_bar() {
    ImGui::EndTabBar();
  }
  bool begin_main_menu_bar() {
    return ImGui::BeginMainMenuBar();
  }
  void end_main_menu_bar() {
    ImGui::EndMainMenuBar();
  }
  bool begin_menu_bar() {
    return ImGui::BeginMenuBar();
  }
  void end_menu_bar() {
    ImGui::EndMenuBar();
  }
  bool begin_menu(const std::string& label, bool enabled) {
    return ImGui::BeginMenu(label.c_str(), enabled);
  }
  void end_menu() {
    ImGui::EndMenu();
  }
  void begin_group() {
    ImGui::BeginGroup();
  }
  void end_group() {
    ImGui::EndGroup();
  }
  void table_setup_column(const std::string& label, table_column_flags_t flags, f32_t init_width_or_weight, id_t user_id) {
    ImGui::TableSetupColumn(label.c_str(), flags, init_width_or_weight, user_id);
  }
  void table_headers_row() {
    ImGui::TableHeadersRow();
  }
  bool table_set_column_index(int column_n) {
    return ImGui::TableSetColumnIndex(column_n);
  }
  bool menu_item(const std::string& label, const std::string& shortcut, bool selected, bool enabled) {
    return ImGui::MenuItem(label.c_str(), shortcut.empty() ? nullptr : shortcut.c_str(), selected, enabled);
  }
  bool begin_combo(const std::string& label, const std::string& preview_value, int flags) {
    return ImGui::BeginCombo(label.c_str(), preview_value.c_str(), flags);
  }
  void end_combo() {
    ImGui::EndCombo();
  }
  void set_item_default_focus() {
    ImGui::SetItemDefaultFocus();
  }
  void same_line(f32_t offset_from_start_x, f32_t spacing_w) {
    ImGui::SameLine(offset_from_start_x, spacing_w);
  }
  void new_line() {
    ImGui::NewLine();
  }
  viewport_t* get_main_viewport() {
    return ImGui::GetMainViewport();
  }
  f32_t get_frame_height() {
    return ImGui::GetFrameHeight();
  }
  f32_t get_text_line_height_with_spacing() {
    return ImGui::GetTextLineHeightWithSpacing();
  }
  fan::vec2 get_mouse_pos() {
    ImVec2 pos = ImGui::GetMousePos();
    return fan::vec2(pos.x, pos.y);
  }
  bool selectable(const std::string& label, bool selected, selectable_flag_t flags, const fan::vec2& size) {
    return ImGui::Selectable(label.c_str(), selected, flags, size);
  }
  bool selectable(const std::string& label, bool* p_selected, selectable_flag_t flags, const fan::vec2& size) {
    return ImGui::Selectable(label.c_str(), p_selected, flags, size);
  }
  bool is_mouse_double_clicked(int button) {
    return ImGui::IsMouseDoubleClicked(button);
  }
  fan::vec2 get_content_region_avail() {
    return ImGui::GetContentRegionAvail();
  }
  fan::vec2 get_content_region_max() {
    return ImGui::GetContentRegionMax();
  }
  fan::vec2 get_item_rect_min() {
    return ImGui::GetItemRectMin();
  }
  fan::vec2 get_item_rect_max() {
    return ImGui::GetItemRectMax();
  }
  void item_size(const fan::vec2& v) {
    ImGui::ItemSize(v);
  }
  void item_size(const rect_t& bb, f32_t text_baseline_y) {
    ImGui::ItemSize(bb.GetSize(), text_baseline_y); 
  } 
  void push_item_width(f32_t item_width) {
    ImGui::PushItemWidth(item_width);
  }
  void pop_item_width() {
    ImGui::PopItemWidth();
  }
  void set_cursor_screen_pos(const fan::vec2& pos) {
    ImGui::SetCursorScreenPos(ImVec2(pos.x, pos.y));
  }
  void push_id(const std::string& str_id) {
    ImGui::PushID(str_id.c_str());
  }
  void push_id(int int_id) {
    ImGui::PushID(int_id);
  }
  void pop_id() {
    ImGui::PopID();
  }
  void set_next_item_width(f32_t width) {
    ImGui::SetNextItemWidth(width);
  }
  void push_text_wrap_pos(f32_t local_pos) {
    ImGui::PushTextWrapPos(local_pos);
  }
  void pop_text_wrap_pos() {
    ImGui::PopTextWrapPos();
  }
  bool is_item_hovered(hovered_flag_t flags) {
    return ImGui::IsItemHovered(flags);
  }
  bool is_any_item_hovered() {
    return ImGui::IsAnyItemHovered();
  }
  bool is_any_item_active() {
    return ImGui::IsAnyItemActive();
  }
  bool is_item_clicked() {
    return ImGui::IsItemClicked();
  }
  bool is_item_held(int mouse_button) {
    if (ImGui::IsDragDropActive()) {
      return false;
    }
    return ImGui::IsMouseDown(mouse_button) && is_item_hovered(hovered_flags_rect_only);
  }
  void begin_tooltip() {
    ImGui::BeginTooltip();
  }
  void end_tooltip() {
    ImGui::EndTooltip();
  }
  void set_tooltip(const std::string& tooltip) {
    ImGui::SetTooltip(tooltip.c_str());
  }
  bool begin_table(const std::string& str_id, int columns, table_flags_t flags, const fan::vec2& outer_size, f32_t inner_width) {
    return ImGui::BeginTable(str_id.c_str(), columns, flags, outer_size, inner_width);
  }
  void end_table() {
    ImGui::EndTable();
  }
  void table_next_row(table_row_flags_t row_flags, f32_t min_row_height) {
    ImGui::TableNextRow(row_flags, min_row_height);
  }
  bool table_next_column() {
    return ImGui::TableNextColumn();
  }
  void columns(int count, const char* id, bool borders) {
    ImGui::Columns(count, id, borders);
  }
  void next_column() {
    ImGui::NextColumn();
  }
  void push_font(font_t* font) {
    ImGui::PushFont(font);
  }
  void pop_font() {
    ImGui::PopFont();
  }
  void set_font(f32_t size) {
    push_font(get_font(size, false));
  }
  font_t* get_font() {
    return ImGui::GetFont();
  }
  f32_t get_font_size() {
    return ImGui::GetFontSize();
  }
  f32_t get_text_line_height() {
    return ImGui::GetTextLineHeight();
  }
  void indent(f32_t indent_w) {
    ImGui::Indent(indent_w);
  }
  void unindent(f32_t indent_w) {
    ImGui::Indent(indent_w);
  }
  fan::vec2 calc_text_size(const std::string& text, const char* text_end, bool hide_text_after_double_hash, f32_t wrap_width) {
    return ImGui::CalcTextSize(text.c_str(), text_end, hide_text_after_double_hash, wrap_width);
  }
  fan::vec2 get_text_size(const std::string& text, const char* text_end, bool hide_text_after_double_hash, f32_t wrap_width) {
    return calc_text_size(text, text_end, hide_text_after_double_hash, wrap_width);
  }
  fan::vec2 text_size(const std::string& text, const char* text_end, bool hide_text_after_double_hash, f32_t wrap_width) {
    return calc_text_size(text, text_end, hide_text_after_double_hash, wrap_width);
  }
  void set_cursor_pos_x(f32_t pos) {
    ImGui::SetCursorPosX(pos);
  }
  void set_cursor_pos_y(f32_t pos) {
    ImGui::SetCursorPosY(pos);
  }
  void set_cursor_pos(const fan::vec2& pos) {
    ImGui::SetCursorPos(pos);
  }
  fan::vec2 get_cursor_pos() {
    return ImGui::GetCursorPos();
  }
  f32_t get_cursor_pos_x() {
    return ImGui::GetCursorPosX();
  }
  f32_t get_cursor_pos_y() {
    return ImGui::GetCursorPosY();
  }
  fan::vec2 get_cursor_screen_pos() {
    ImVec2 pos = ImGui::GetCursorScreenPos();
    return fan::vec2(pos.x, pos.y);
  }
  fan::vec2 get_cursor_start_pos() {
    return ImGui::GetCursorStartPos();
  }
  bool is_window_hovered(hovered_flag_t hovered_flags) {
    return ImGui::IsWindowHovered(hovered_flags);
  }
  bool is_window_focused() {
    return ImGui::IsWindowFocused();
  }
  void set_next_window_focus() {
    return ImGui::SetNextWindowFocus();
  }
  void set_window_focus(const std::string& name) {
    ImGui::SetWindowFocus(name.c_str());
  }
  int render_window_flags() {
    return gui::window_flags_no_title_bar | gui::window_flags_no_background | gui::window_flags_override_input;
  }
  fan::vec2 get_window_content_region_min() {
    return ImGui::GetWindowContentRegionMin();
  }
  fan::vec2 get_window_content_region_max() {
    return ImGui::GetWindowContentRegionMax();
  }
  f32_t get_column_width(int index) {
    return ImGui::GetColumnWidth(index);
  }
  void set_column_width(int index, f32_t width) {
    ImGui::SetColumnWidth(index, width);
  }
  bool is_item_active() {
    return ImGui::IsItemActive();
  }

  // TODO need gui storage
  bool g_want_io = false;
  bool want_io() {
    return g_want_io;
  }
  void set_want_io(bool flag, bool op_or) {
    ImGuiContext* g = ImGui::GetCurrentContext();
    if (g->NavWindow) {
      std::string nav_window_name = g->NavWindow->Name;
      if (nav_window_name.find("WindowOverViewport_") == 0) {
        g_want_io = false;
        return;
      }
    }
    if (g->NavWindow && want_io_ignore_list.find(g->NavWindow->Name) != want_io_ignore_list.end()) {
      g_want_io = false;
      return;
    }

    if (g->HoveredWindow && want_io_ignore_list.find(g->HoveredWindow->Name) != want_io_ignore_list.end()
      ) {
      g_want_io = false;
      return;
    }
    /*
    printf("WantCapture: flag=%d keyboard=%d mouse=%d text=%d\n", 
      flag, 
      ImGui::GetIO().WantCaptureKeyboard, 
      ImGui::GetIO().WantCaptureMouse, 
      ImGui::GetIO().WantTextInput
    );
    if (g->HoveredWindow) {
      printf("Hovered window: %s\n", g->HoveredWindow->Name);
    }
    if (g->NavWindow) {
      printf("Nav window: %s\n", g->NavWindow->Name);
    }
    */
    g_want_io = op_or ? g_want_io | flag : flag;
  }
  void set_keyboard_focus_here() {
    ImGui::SetKeyboardFocusHere();
  }
  fan::vec2 get_mouse_drag_delta(int button, f32_t lock_threshold) {
    ImVec2 delta = ImGui::GetMouseDragDelta(button, lock_threshold);
    return fan::vec2(delta.x, delta.y);
  }
  void reset_mouse_drag_delta(int button) {
    ImGui::ResetMouseDragDelta(button);
  }
  void set_scroll_x(f32_t scroll_x) {
    ImGui::SetScrollX(scroll_x);
  }
  void set_scroll_y(f32_t scroll_y) {
    ImGui::SetScrollY(scroll_y);
  }
  void set_scroll_here_y() {
    ImGui::SetScrollHereY();
  }
  f32_t get_scroll_x() {
    return ImGui::GetScrollX();
  }
  f32_t get_scroll_y() {
    return ImGui::GetScrollY();
  }

  void push_style_color(col_t index, const fan::color& col) {
    ImGui::PushStyleColor(index, col);
  }
  void pop_style_color(int n) {
    ImGui::PopStyleColor(n);
  }
  void push_style_var(style_var_t index, f32_t val) {
    ImGui::PushStyleVar(index, val);
  }
  void push_style_var(style_var_t index, const fan::vec2& val) {
    ImGui::PushStyleVar(index, val);
  }
  void pop_style_var(int n) {
    ImGui::PopStyleVar(n);
  }

  bool button(const std::string& label, const fan::vec2& size) {
    return ImGui::Button(label.c_str(), size);
  }
  bool invisible_button(const std::string& label, const fan::vec2& size) {
    return ImGui::InvisibleButton(label.c_str(), size);
  }
  bool arrow_button(const std::string& label, dir_t dir) {
    return ImGui::ArrowButton(label.c_str(), dir);
  }

  /// <summary>
  /// Draws the specified text, with its position influenced by other GUI elements.
  /// </summary>
  /// <param name="text">The text to draw.</param>
  /// <param name="color">The color of the text (defaults to white).</param>
  void text_colored(const char* text, const fan::color& color) {
    push_style_color(col_text, color);
    ImGui::Text(text);
    pop_style_color();
  }
  /// <summary>
  /// Draws the specified text, with its position influenced by other GUI elements.
  /// </summary>
  /// <param name="text">The text to draw.</param>
  /// <param name="color">The color of the text (defaults to white).</param>
  void text_colored(const std::string& text, const fan::color& color) {
    gui::text_colored(text.c_str(), color);
  }

  constexpr const f32_t outline_thickness = 1.5f;
  constexpr const fan::vec2 outline_offsets[] = {
    {0, -outline_thickness},  // top
    {-outline_thickness, 0},  // left
    {outline_thickness, 0},   // right
    {0, outline_thickness}    // bottom
  };

  /// <summary>
  /// Draws the specified text at a given position on the screen.
  /// </summary>
  /// <param name="text">The text to draw.</param>
  /// <param name="position">The position of the text.</param>
  /// <param name="color">The color of the text (defaults to white).</param>
  void text_at(const std::string& text, const fan::vec2& position, const fan::color& color) {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    draw_list->AddText(position, color.get_gui_color(), text.c_str());
  }

  void text_wrapped(const std::string& text, const fan::color& color) {
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::TextWrapped("%s", text.c_str());
    ImGui::PopStyleColor();
  }

  void text_unformatted(const std::string& text, const char* text_end) {
    ImGui::TextUnformatted(text.c_str(), text_end);
  }

  void text_disabled(const std::string& text) {
    ImGui::TextDisabled(text.c_str());
  }
  /// <summary>
  /// Draws text centered horizontally.
  /// </summary>
  /// <param name="text">The text to draw.</param>
  /// <param name="color">The color of the text (defaults to white).</param>
  /// <param name="offset">Offset from center position.</param>
  void text_centered(const std::string& text, const fan::color& color) {
    fan::vec2 text_size = ImGui::CalcTextSize(text.c_str());
    fan::vec2 window_size = ImGui::GetWindowSize();
    fan::vec2 current_pos = ImGui::GetCursorPos();

    current_pos.x -= text_size.x * 0.5f;
    current_pos.y -= text_size.y * 0.5f;

    ImGui::SetCursorPos(current_pos);
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::Text("%s", text.c_str());
    ImGui::PopStyleColor();
  }

  /// <summary>
  /// Draws text centered at a specific position.
  /// </summary>
  /// <param name="text">The text to draw.</param>
  /// <param name="center_position">The position where the text should be centered.</param>
  /// <param name="color">The color of the text (defaults to white).</param>
  void text_centered_at(const std::string& text, const fan::vec2& center_position, const fan::color& color) {
    fan::vec2 text_size = ImGui::CalcTextSize(text.c_str());
    fan::vec2 draw_position = center_position;
    draw_position.x -= text_size.x * 0.5f;
    draw_position.y -= text_size.y * 0.5f;

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    draw_list->AddText(draw_position, color.get_gui_color(), text.c_str());
  }

  /// <summary>
  /// Draws text to bottom right.
  /// </summary>
  /// <param name="text">The text to draw.</param>
  /// <param name="color">The color of the text (defaults to white).</param>
  /// <param name="offset">Offset from the bottom-right corner.</param>
  void text_bottom_right(const std::string& text, const fan::color& color, const fan::vec2& offset) {
    fan::vec2 text_pos;
    fan::vec2 text_size = ImGui::CalcTextSize(text.c_str());
    fan::vec2 window_pos = ImGui::GetWindowPos();
    fan::vec2 window_size = ImGui::GetWindowSize();

    text_pos.x = window_pos.x + window_size.x - text_size.x - ImGui::GetStyle().WindowPadding.x;
    text_pos.y = window_pos.y + window_size.y - text_size.y - ImGui::GetStyle().WindowPadding.y;
    text_at(text, text_pos + offset, color);
  }

  void text_outlined_at(const std::string& text, const fan::vec2& screen_pos, const fan::color& color, const fan::color& outline_color) {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    // Draw outline
    for ( const auto& offset : outline_offsets ) {
      draw_list->AddText(screen_pos + offset, outline_color.get_gui_color(), text.c_str());
    }

    draw_list->AddText(screen_pos, color.get_gui_color(), text.c_str());
  }

  void text_outlined(const std::string& text, const fan::color& color, const fan::color& outline_color) {
    fan::vec2 cursor_pos = ImGui::GetCursorPos();

    // Draw outline
    ImGui::PushStyleColor(ImGuiCol_Text, outline_color);
    for ( const auto& offset : outline_offsets ) {
      ImGui::SetCursorPos(cursor_pos + offset);
      ImGui::Text("%s", text.c_str());
    }
    ImGui::PopStyleColor();

    ImGui::SetCursorPos(cursor_pos);
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::Text("%s", text.c_str());
    ImGui::PopStyleColor();
  }

  /// <summary>
  /// Draws outlined text centered at a specific position.
  /// </summary>
  /// <param name="text">The text to draw.</param>
  /// <param name="center_position">The position where the text should be centered.</param>
  /// <param name="color">The color of the text (defaults to white).</param>
  /// <param name="outline_color">The color of the outline (defaults to black).</param>
  void text_centered_outlined_at(const std::string& text, const fan::vec2& center_position, const fan::color& color, const fan::color& outline_color) {
    fan::vec2 text_size = ImGui::CalcTextSize(text.c_str());
    fan::vec2 draw_position = center_position;
    draw_position.x -= text_size.x * 0.5f;
    draw_position.y -= text_size.y * 0.5f;

    text_outlined_at(text, draw_position, color, outline_color);
  }

  /// <summary>
  /// Draws outlined text centered horizontally within the current window.
  /// </summary>
  /// <param name="text">The text to draw.</param>
  /// <param name="color">The color of the text (defaults to white).</param>
  /// <param name="outline_color">The color of the outline (defaults to black).</param>
  void text_centered_outlined(const std::string& text, const fan::color& color, const fan::color& outline_color) {
    fan::vec2 text_size = ImGui::CalcTextSize(text.c_str());
    fan::vec2 window_size = ImGui::GetWindowSize();
    fan::vec2 current_pos = ImGui::GetCursorPos();
    current_pos.x = (window_size.x - text_size.x) * 0.5f;
    current_pos.y -= text_size.y * 0.5f;

    for ( const auto& offset : outline_offsets ) {
      ImGui::SetCursorPos(current_pos + offset);
      ImGui::PushStyleColor(ImGuiCol_Text, outline_color);
      ImGui::Text("%s", text.c_str());
      ImGui::PopStyleColor();
    }

    ImGui::SetCursorPos(current_pos);
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::Text("%s", text.c_str());
    ImGui::PopStyleColor();
  }

  void text_box(const std::string& text, const ImVec2& size, const fan::color& text_color, const fan::color& bg_color) {

    ImVec2 text_size = ImGui::CalcTextSize(text.c_str());
    ImVec2 padding = ImGui::GetStyle().FramePadding;
    ImVec2 box_size = size;

    if ( box_size.x <= 0 ) {
      box_size.x = text_size.x + padding.x * 2;
    }
    if ( box_size.y <= 0 ) {
      box_size.y = text_size.y + padding.y * 2;
    }

    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    fan::color actual_bg_color = bg_color;
    if ( bg_color.r == 0 && bg_color.g == 0 && bg_color.b == 0 && bg_color.a == 0 ) {
      ImVec4 default_bg = ImGui::GetStyleColorVec4(ImGuiCol_Button);
      actual_bg_color = fan::color(default_bg.x, default_bg.y, default_bg.z, default_bg.w);
    }

    ImU32 bg_color_u32 = ImGui::ColorConvertFloat4ToU32(ImVec4(actual_bg_color.r, actual_bg_color.g, actual_bg_color.b, actual_bg_color.a));
    ImU32 border_color = ImGui::GetColorU32(ImGuiCol_Border);
    f32_t rounding = ImGui::GetStyle().FrameRounding;

    draw_list->AddRectFilled(pos, ImVec2(pos.x + box_size.x, pos.y + box_size.y), bg_color_u32, rounding);
    draw_list->AddRect(pos, ImVec2(pos.x + box_size.x, pos.y + box_size.y), border_color, rounding);

    ImVec2 text_pos = ImVec2(
      pos.x + (box_size.x - text_size.x) * 0.5f,
      pos.y + (box_size.y - text_size.y) * 0.5f
    );

    ImU32 text_color_u32 = ImGui::ColorConvertFloat4ToU32(ImVec4(text_color.r, text_color.g, text_color.b, text_color.a));
    draw_list->AddText(text_pos, text_color_u32, text.c_str());

    ImGui::Dummy(box_size);
  }

  f32_t calc_item_width() {
    return ImGui::CalcItemWidth();
  }

  f32_t get_item_width() {
    return calc_item_width();
  }

  bool input_text(const std::string& label, std::string* buf, input_text_flags_t flags, input_text_callback_t callback, void* user_data) {
    IM_ASSERT((flags & ImGuiInputTextFlags_CallbackResize) == 0);
    flags |= ImGuiInputTextFlags_CallbackResize;

    InputTextCallback_UserData cb_user_data;
    cb_user_data.Str = buf;
    cb_user_data.ChainCallback = callback;
    cb_user_data.ChainCallbackUserData = user_data;
    return ImGui::InputText(label.c_str(), (char*)buf->c_str(), buf->capacity() + 1, flags, InputTextCallback, &cb_user_data);
  }

  bool input_text_multiline(const std::string& label, std::string* buf, const ImVec2& size, input_text_flags_t flags, input_text_callback_t callback, void* user_data) {
    IM_ASSERT((flags & ImGuiInputTextFlags_CallbackResize) == 0);
    flags |= ImGuiInputTextFlags_CallbackResize;

    InputTextCallback_UserData cb_user_data;
    cb_user_data.Str = buf;
    cb_user_data.ChainCallback = callback;
    cb_user_data.ChainCallbackUserData = user_data;
    return ImGui::InputTextMultiline(label.c_str(), (char*)buf->c_str(), buf->capacity() + 1, size, flags, InputTextCallback, &cb_user_data);
  }

  bool input_float(const std::string& label, f32_t* v, f32_t step, f32_t step_fast, const char* format, input_text_flags_t flags) {
    return ImGui::InputFloat(label.c_str(), v, step, step_fast, format, flags);
  }

  bool input_float(const std::string& label, fan::vec2* v, const char* format, input_text_flags_t flags) {
    return ImGui::InputFloat2(label.c_str(), v->data(), format, flags);
  }

  bool input_float(const std::string& label, fan::vec3* v, const char* format, input_text_flags_t flags) {
    return ImGui::InputFloat3(label.c_str(), v->data(), format, flags);
  }

  bool input_float(const std::string& label, fan::vec4* v, const char* format, input_text_flags_t flags) {
    return ImGui::InputFloat4(label.c_str(), v->data(), format, flags);
  }

  bool input_int(const std::string& label, int* v, int step, int step_fast, input_text_flags_t flags) {
    return ImGui::InputInt(label.c_str(), v, step, step_fast, flags);
  }

  bool input_int(const std::string& label, fan::vec2i* v, input_text_flags_t flags) {
    return ImGui::InputInt2(label.c_str(), v->data(), flags);
  }

  bool input_int(const std::string& label, fan::vec3i* v, input_text_flags_t flags) {
    return ImGui::InputInt3(label.c_str(), v->data(), flags);
  }

  bool input_int(const std::string& label, fan::vec4i* v, input_text_flags_t flags) {
    return ImGui::InputInt4(label.c_str(), v->data(), flags);
  }

  bool color_edit3(const std::string& label, fan::color* color, color_edit_flags_t flags) {
    return ImGui::ColorEdit3(label.c_str(), color->data(), flags);
  }

  bool color_edit3(const std::string& label, fan::vec3* color, color_edit_flags_t flags) {
    return ImGui::ColorEdit3(label.c_str(), color->data(), flags);
  }

  bool color_edit4(const std::string& label, fan::color* color, color_edit_flags_t flags) {
    return ImGui::ColorEdit4(label.c_str(), color->data(), flags);
  }

  fan::vec2 get_window_pos() {
    return ImGui::GetWindowPos();
  }

  fan::vec2 get_window_size() {
    return ImGui::GetWindowSize();
  }

  void set_next_window_pos(const fan::vec2& position, cond_t cond, const fan::vec2& pivot) {
    ImGui::SetNextWindowPos(position, cond, pivot);
  }

  void set_next_window_size(const fan::vec2& size, cond_t cond) {
    ImGui::SetNextWindowSize(size, cond);
  }

  void set_next_window_bg_alpha(f32_t a) {
    ImGui::SetNextWindowBgAlpha(a);
  }

  void set_window_font_scale(f32_t scale) {
    ImGui::SetWindowFontScale(scale);
  }

  bool is_mouse_dragging(int button, f32_t threshold) {
    return ImGui::IsMouseDragging(button, threshold);
  }

  bool is_item_deactivated_after_edit() {
    return ImGui::IsItemDeactivatedAfterEdit();
  }

  void set_mouse_cursor(cursor_t type) {
    ImGui::SetMouseCursor(type);
  }

  style_t& get_style() {
    return ImGui::GetStyle();
  }

  fan::color get_color(col_t idx) {
    return get_style().Colors[idx];
  }

  uint32_t get_color_u32(col_t idx) {
    return ImGui::GetColorU32(idx);
  }

  void separator() {
    ImGui::Separator();
  }

  void dock_space_over_viewport(id_t dockspace_id, const gui::viewport_t* viewport, int flags, const void* window_class) {
    ImGui::DockSpaceOverViewport(dockspace_id, viewport, flags, static_cast<const ImGuiWindowClass*>(window_class));
  }

  context_t* get_context() {
    return ImGui::GetCurrentContext();
  }

  void spacing() {
    ImGui::Spacing();
  }

  io_t& get_io() {
    return ImGui::GetIO();
  }

  bool tree_node_ex(const std::string& label, tree_node_flags_t flags) {
    return ImGui::TreeNodeEx(label.c_str(), flags);
  }
  void tree_pop() {
    ImGui::TreePop();
  }

  bool tree_node(const std::string& label) {
    return ImGui::TreeNode(label.c_str());
  }

  bool is_item_toggled_open() {
    return ImGui::IsItemToggledOpen();
  }

  void dummy(const fan::vec2& size) {
    ImGui::Dummy(size);
  }

  draw_list_t* get_window_draw_list() {
    return ImGui::GetWindowDrawList();
  }
  draw_list_t* get_foreground_draw_list() {
    return ImGui::GetForegroundDrawList();
  }
  draw_list_t* get_background_draw_list() {
    return ImGui::GetBackgroundDrawList();
  }

  window_handle_t* get_current_window() {
    return ImGui::GetCurrentWindow();
  }

  bool combo(const std::string& label, int* current_item, const char* const items[], int items_count, int popup_max_height_in_items) {
    return ImGui::Combo(label.c_str(), current_item, items, items_count, popup_max_height_in_items);
  }

  // Separate items with \0 within a string, end item-list with \0\0. e.g. "One\0Two\0Three\0"
  bool combo(const std::string& label, int* current_item, const char* items_separated_by_zeros, int popup_max_height_in_items) {
    return ImGui::Combo(label.c_str(), current_item, items_separated_by_zeros, popup_max_height_in_items);
  }

  bool combo(const std::string& label, int* current_item, const char* (*getter)(void* user_data, int idx), void* user_data, int items_count, int popup_max_height_in_items) {
    return ImGui::Combo(label.c_str(), current_item, getter, user_data, items_count, popup_max_height_in_items);
  }

  bool checkbox(const std::string& label, bool* v) {
    return ImGui::Checkbox(label.c_str(), v);
  }

  bool list_box(const std::string& label, int* current_item, bool(*old_callback)(void* user_data, int idx, const char** out_text), void* user_data, int items_count, int height_in_items) {
    return ImGui::ListBox(label.c_str(), current_item, old_callback, user_data, items_count, height_in_items);
  }

  bool list_box(const std::string& label, int* current_item, const char* const items[], int items_count, int height_in_items) {
    return ImGui::ListBox(label.c_str(), current_item, items, items_count, height_in_items);
  }

  bool list_box(const std::string& label, int* current_item, const char* (*getter)(void* user_data, int idx), void* user_data, int items_count, int height_in_items) {
    return ImGui::ListBox(label.c_str(), current_item, getter, user_data, items_count, height_in_items);
  }

  bool toggle_button(const std::string& str, bool* v) {
    ImGui::Text("%s", str.c_str());
    ImGui::SameLine();

    ImVec2 p = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    f32_t height = ImGui::GetFrameHeight();
    f32_t width = height * 1.55f;
    f32_t radius = height * 0.50f;

    bool changed = ImGui::InvisibleButton(("##" + str).c_str(), ImVec2(width, height));
    if ( changed )
      *v = !*v;
    ImU32 col_bg;
    if ( ImGui::IsItemHovered() )
      col_bg = *v ? IM_COL32(145 + 20, 211, 68 + 20, 255) : IM_COL32(218 - 20, 218 - 20, 218 - 20, 255);
    else
      col_bg = *v ? IM_COL32(145, 211, 68, 255) : IM_COL32(218, 218, 218, 255);

    draw_list->AddRectFilled(p, ImVec2(p.x + width, p.y + height), col_bg, height * 0.5f);
    draw_list->AddCircleFilled(ImVec2(*v ? (p.x + width - radius) : (p.x + radius), p.y + radius), radius - 1.5f, IM_COL32(255, 255, 255, 255));

    return changed;
  }


  void text_bottom_right(const std::string& text, uint32_t reverse_yoffset) {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    ImVec2 window_pos = ImGui::GetWindowPos();
    ImVec2 window_size = ImGui::GetWindowSize();

    ImVec2 text_size = ImGui::CalcTextSize(text.c_str());

    ImVec2 text_pos;
    text_pos.x = window_pos.x + window_size.x - text_size.x - ImGui::GetStyle().WindowPadding.x;
    text_pos.y = window_pos.y + window_size.y - text_size.y - ImGui::GetStyle().WindowPadding.y;

    text_pos.y -= reverse_yoffset * ImGui::GetTextLineHeightWithSpacing();

    draw_list->AddText(text_pos, IM_COL32(255, 255, 255, 255), text.c_str());
  }

  fan::vec2 get_position_bottom_corner(const std::string& text, uint32_t reverse_yoffset) {
    fan::vec2 window_pos = ImGui::GetWindowPos();
    fan::vec2 window_size = ImGui::GetWindowSize();

    fan::vec2 text_size = ImGui::CalcTextSize(text.c_str());

    fan::vec2 text_pos;
    text_pos.x = window_pos.x + window_size.x - text_size.x - ImGui::GetStyle().WindowPadding.x;
    text_pos.y = window_pos.y + window_size.y - text_size.y - ImGui::GetStyle().WindowPadding.y;

    text_pos.y -= reverse_yoffset * ImGui::GetTextLineHeightWithSpacing();

    return text_pos;
  }

  void send_drag_drop_item(const std::string& id, const std::wstring& path, const std::string& popup) {
    std::string popup_ = popup;
    if ( popup.empty() ) {
      popup_ = {path.begin(), path.end()};
    }
    if ( ImGui::BeginDragDropSource() ) {
      ImGui::SetDragDropPayload(id.c_str(), path.data(), (path.size() + 1) * sizeof(wchar_t));
      ImGui::Text("%s", popup_.c_str());
      ImGui::EndDragDropSource();
    }
  }

  void receive_drag_drop_target(const std::string& id, std::function<void(std::string)> receive_func) {
    if ( ImGui::BeginDragDropTarget() ) {
      if ( const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(id.c_str()) ) {
        const wchar_t* path = (const wchar_t*)payload->Data;
        std::wstring wstr = path;
        receive_func(std::string(wstr.begin(), wstr.end()));
      }
      ImGui::EndDragDropTarget();
    }
  }

  bool slider_scalar(const char* label, data_type_t data_type, void* p_data, const void* p_min, const void* p_max, const char* format, slider_flags_t flags) {
    return ImGui::SliderScalar(label, data_type, p_data, p_min, p_max, format, flags);
  }
  bool slider_scalar_n(const char* label, data_type_t data_type, void* p_data, int components, const void* p_min, const void* p_max, const char* format, slider_flags_t flags) {
    return ImGui::SliderScalarN(label, data_type, p_data, components, p_min, p_max, format, flags);
  }

  bool drag_scalar(const char* label, data_type_t data_type, void* p_data, f32_t v_speed, const void* p_min, const void* p_max, const char* format, slider_flags_t flags) {
    return ImGui::DragScalar(label, data_type, p_data, v_speed, p_min, p_max, format, flags);
  }
  bool drag_scalar_n(const char* label, data_type_t data_type, void* p_data, int components, f32_t v_speed, const void* p_min, const void* p_max, const char* format, slider_flags_t flags) {
    return ImGui::DragScalarN(label, data_type, p_data, components, v_speed, p_min, p_max, format, flags);
  }


  font_t* get_font_impl(f32_t font_size, bool bold) {
    return get_font(bold ? fonts_bold : fonts, font_size);
  }
  font_t* get_font(
    font_t* (&fonts)[std::size(fan::graphics::gui::font_sizes)],
    f32_t font_size
  ) {
    font_size /= 2;

    int best_index = 0;
    f32_t best_diff = std::abs(fan::graphics::gui::font_sizes[0] - font_size);

    for (std::size_t i = 1; i < std::size(fan::graphics::gui::font_sizes); ++i) {
      f32_t diff = std::abs(fan::graphics::gui::font_sizes[i] - font_size);
      if (diff < best_diff) {
        best_diff = diff;
        best_index = i;
      }
    }

    return fonts[best_index];
  }
  font_t* get_font(f32_t font_size, bool bold) {
    return get_font_impl(font_size, bold);
  }

  void image(texture_id_t user_texture_id, const fan::vec2& size, const fan::vec2& uv0, const fan::vec2& uv1, const fan::color& tint_col, const fan::color& border_col) {
    ImGui::Image(user_texture_id, size, uv0, uv1, tint_col, border_col);
  }
  bool image_button(const std::string& str_id, texture_id_t user_texture_id, const fan::vec2& size, const fan::vec2& uv0, const fan::vec2& uv1, const fan::color& bg_col, const fan::color& tint_col) {
    return ImGui::ImageButton(str_id.c_str(), user_texture_id, size, uv0, uv1, bg_col, tint_col);
  }

  bool item_add(const rect_t& bb, id_t id, const rect_t* nav_bb, item_flags_t extra_flags) {
    return ImGui::ItemAdd(bb, id, nav_bb, extra_flags);
  }


  int is_key_pressed(key_t key, bool repeat) {
    return ImGui::IsKeyPressed(key, repeat);
  }
  int get_pressed_key() {
    auto& style = get_style();
    if (style.DisabledAlpha != style.Alpha && is_window_focused()) {
      for (int i = ImGuiKey_A; i <= ImGuiKey_Z; ++i) {
        if (ImGui::IsKeyPressed((ImGuiKey)i, false)) {
          return (i - ImGuiKey_A) + 'A';
        }
      }
    }
    return -1;
  }

  void set_next_window_class(const class_t* c) {
    ImGui::SetNextWindowClass(c);
  }

  bool begin_drag_drop_source() {
    return ImGui::BeginDragDropSource();
  }
  bool set_drag_drop_payload(const std::string& type, const void* data, size_t sz, cond_t cond) {
    return ImGui::SetDragDropPayload(type.c_str(), data, sz, cond);
  }
  void end_drag_drop_source() {
    ImGui::EndDragDropSource();
  }

  bool begin_drag_drop_target() {
    return ImGui::BeginDragDropTarget();

  }
  const payload_t* accept_drag_drop_payload(const std::string& type) {
    return ImGui::AcceptDragDropPayload(type.c_str());
  }
  void end_drag_drop_target() {
    ImGui::EndDragDropTarget();
  }
  const payload_t* get_drag_drop_payload() {
    return ImGui::GetDragDropPayload();
  }

  bool begin_popup(const std::string& id, window_flags_t flags) {
    return ImGui::BeginPopup(id.c_str(), flags);
  }

  bool begin_popup_modal(const std::string& id, window_flags_t flags) {
    return ImGui::BeginPopupModal(id.c_str(), 0, flags);
  }

  void end_popup() {
    ImGui::EndPopup();
  }

  void open_popup(const std::string& id) {
    ImGui::OpenPopup(id.c_str());
  }

  void close_current_popup() {
    ImGui::CloseCurrentPopup();
  }

  bool is_popup_open(const std::string& id) {
    return ImGui::IsPopupOpen(id.c_str());
  }
  id_t get_id(const std::string& str_id) {
    return ImGui::GetID(str_id.c_str());
  }
  storage_t* get_state_storage() {
    return ImGui::GetStateStorage();
  }

  f32_t get_line_height_with_spacing() {
    return ImGui::GetTextLineHeightWithSpacing();
  }

  void seperator() {
    ImGui::Separator();
  }

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
  ) {
    ImGui::CreateContext();
    ImPlot::CreateContext();
    auto& input_map = ImPlot::GetInputMap();
    input_map.Pan = ImGuiMouseButton_Middle;

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ///    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    ImGuiStyle& style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
      style.WindowRounding = 0.;
    }
    style.FrameRounding = 5.f;
    style.FramePadding = ImVec2(12.f, 5.f);
    style.Colors[ImGuiCol_WindowBg].w = 1.0f;

    init_graphics_context(
      window, 
      renderer,
      opengl_renderer_definition,
      vulkan_renderer_definition
    #if defined(fan_vulkan)
      , 
      instance, 
      physical_device, 
      device, 
      queue_family, 
      graphics_queue, 
      descriptor_pool, 
      render_pass, 
      image_count, 
      min_image_count, 
      msaa_samples,
      check_vk_result
    #endif
    );

    imgui_themes::dark();

    fan::graphics::gui::init_fonts();

    g_gui_initialized = true;
  }

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
  ) {
    if (renderer == opengl_renderer_definition) {
      glfwMakeContextCurrent(window);
      ImGui_ImplGlfw_InitForOpenGL(window, true);
      const char* glsl_version = "#version 120";
      ImGui_ImplOpenGL3_Init(glsl_version);
    }

  #if defined(fan_vulkan)
    else if (renderer == vulkan_renderer_definition) {
      ImGui_ImplGlfw_InitForVulkan(window, true);

      ImGui_ImplVulkan_InitInfo init_info = {};
      init_info.Instance = instance;  
      init_info.PhysicalDevice = physical_device;
      init_info.Device = device;  
      init_info.QueueFamily = queue_family;
      init_info.Queue = graphics_queue;
      init_info.DescriptorPool = descriptor_pool;  
      init_info.RenderPass = render_pass;
      init_info.Subpass = 0;
      init_info.MinImageCount = min_image_count;
      init_info.ImageCount = image_count;  
      init_info.MSAASamples = msaa_samples;
      init_info.CheckVkResultFn = check_vk_result;

      ImGui_ImplVulkan_Init(&init_info);
    }
  #endif
  }

  void build_fonts() {
    auto& io = get_io();
    io.Fonts->Build();
  }
  void rebuild_fonts() {
    auto& io = get_io();
    io.Fonts->Clear();
    build_fonts();
  }

  void init_fonts() {
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->FontBuilderIO = ImGuiFreeType::GetBuilderForFreeType();
    io.Fonts->FontBuilderFlags |= ImGuiFreeTypeBuilderFlags_LoadColor;

    load_fonts(fan::graphics::gui::fonts, "fonts/SourceCodePro-Regular.ttf");

    build_fonts();

    io.FontDefault = fan::graphics::gui::fonts[default_font_size_index];
  }


  void load_fonts(font_t* (&fonts)[std::size(fan::graphics::gui::font_sizes)], const std::string& name, font_config_t* cfg) {
    ImGuiIO& io = ImGui::GetIO();

    font_config_t internal_cfg;
    internal_cfg.FontBuilderFlags |= ImGuiFreeTypeBuilderFlags_LoadColor;
    if (cfg) {
      internal_cfg = *cfg;
    }
    for (std::size_t i = 0; i < std::size(fonts); ++i) {
      fonts[i] = io.Fonts->AddFontFromFileTTF(name.c_str(), fan::graphics::gui::font_sizes[i] * 2, &internal_cfg);

      if (fonts[i] == nullptr) {
        fan::throw_error_impl((std::string("failed to load font:") + name).c_str());
      }
    }
  }
  
  void load_emojis() {
    ImFontConfig emoji_cfg;
    emoji_cfg.FontBuilderFlags |= ImGuiFreeTypeBuilderFlags_LoadColor | ImGuiFreeTypeBuilderFlags_Bitmap;

    // TODO: expand ranges if needed
    static const ImWchar emoji_ranges[] = {
      //0x2600, 0x26FF,    // Miscellaneous Symbols
      //0x2700, 0x27BF,    // Dingbats
      //0x2B00, 0x2BFF,    
      //0x1F300, 0x1F5FF,  // Miscellaneous Symbols and Pictographs
      //0x1F600, 0x1F64F,  // Emoticons
      //0x1F680, 0x1F6FF,  // Transport and Map Symbols
      //0x1F900, 0x1F9FF,  // Supplemental Symbols and Pictographs
      //0x1FA70, 0x1FAFF,  // Symbols and Pictographs Extended-A
      //0

      0x2600, 0x26FF,    // Miscellaneous Symbols
      0x2B00, 0x2BFF,    // Miscellaneous Symbols and Arrows
      0x1F600, 0x1F64F,  // Emoticons
      0
    };

    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->Clear();
    io.Fonts->FontBuilderIO = ImGuiFreeType::GetBuilderForFreeType();
    io.Fonts->FontBuilderFlags |= ImGuiFreeTypeBuilderFlags_LoadColor;

    for (std::size_t i = 0; i < std::size(fan::graphics::gui::fonts); ++i) {
      f32_t font_size = fan::graphics::gui::font_sizes[i] * 2;
      // load 2x font size and possibly downscale for better quality

      ImFontConfig main_cfg;
      fan::graphics::gui::fonts[i] = io.Fonts->AddFontFromFileTTF(
        "fonts/SourceCodePro-Regular.ttf", font_size, &main_cfg
      );

      ImFontConfig emoji_cfg;
      emoji_cfg.MergeMode = true;
      emoji_cfg.FontBuilderFlags |= ImGuiFreeTypeBuilderFlags_LoadColor;
      emoji_cfg.SizePixels = 0;
      emoji_cfg.RasterizerDensity = 1.0f;
      emoji_cfg.GlyphMinAdvanceX = font_size;

      io.Fonts->AddFontFromFileTTF(
        "fonts/seguiemj.ttf", font_size, &emoji_cfg, emoji_ranges
      );
    }

    build_fonts();
    io.FontDefault = fan::graphics::gui::fonts[9];
  }

  void shutdown_graphics_context(
    int renderer,
    int opengl_renderer_definition,  // todo bad
    int vulkan_renderer_definition  //  todo bad
  #if defined(fan_vulkan)
    , VkDevice device
  #endif
  ) {
    if (renderer == opengl_renderer_definition) {
      ImGui_ImplOpenGL3_Shutdown();
    }
  #if defined(fan_vulkan)
    else if (renderer == vulkan_renderer_definition) {
      vkDeviceWaitIdle(device);
      ImGui_ImplVulkan_Shutdown();
    }
  #endif
  }
  void shutdown_window_context() {
    ImGui_ImplGlfw_Shutdown();
  }
  void destroy() {
    shutdown_window_context();
    ImGui::DestroyContext();
    ImPlot::DestroyContext();
    g_gui_initialized = false;
  }

  void new_frame(
    int renderer,
    int opengl_renderer_definition,  // todo bad
    int vulkan_renderer_definition  //  todo bad
  ) {
    if (renderer == opengl_renderer_definition) {
      ImGui_ImplOpenGL3_NewFrame();
    }
  #if defined(fan_vulkan)
    else if (renderer == vulkan_renderer_definition) {
      ImGui_ImplVulkan_NewFrame();
    }
  #endif
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
  }

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
  ) {
    ImGui::Render();

    if (renderer == opengl_renderer_definition) {
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }
  #if defined(fan_vulkan)
    else if (renderer == vulkan_renderer_definition) {
      if (image_error == (VkResult)-0xfff) {
        image_error = VK_SUCCESS;
      }
      if (!render_shapes_top) {
        vkCmdEndRenderPass(cmd_buffer);
      }
      ImDrawData* draw_data = ImGui::GetDrawData();
      const bool is_minimized = (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f);
      if (!is_minimized) {
        render_func(context, image_error, clear_color);
      }
    }
  #endif
  }


  void profile_heap(void* (*dynamic_malloc)(size_t, void*), void (*dynamic_free)(void*, void*)) {
    ImGui::SetAllocatorFunctions(dynamic_malloc, dynamic_free, nullptr);
  }
} // namespace fan::graphics::gui

// plot
namespace fan::graphics::gui {
  bool plot::begin_plot(const std::string& title, const fan::vec2& size, flags_t flags) {
    return ImPlot::BeginPlot(title.c_str(), size, flags);
  }

  void plot::end_plot() {
    ImPlot::EndPlot();
  }

  void plot::setup_axes(const std::string& x_label, const std::string& y_label, axis_flags_t x_flags, axis_flags_t y_flags) {
    ImPlot::SetupAxes(x_label.c_str(), y_label.c_str(), x_flags, y_flags);
  }

  void plot::setup_axis(axis_t axis, const std::string& label, axis_flags_t flags) {
    ImPlot::SetupAxis(axis, label.empty() ? nullptr : label.c_str(), flags);
  }

  void plot::setup_axis_limits(axis_t axis, double v_min, double v_max, cond_t cond) {
    ImPlot::SetupAxisLimits(axis, v_min, v_max, cond);
  }

  void plot::setup_axes_limits(double x_min, double x_max, double y_min, double y_max, cond_t cond) {
    ImPlot::SetupAxesLimits(x_min, x_max, y_min, y_max, cond);
  }

  void plot::setup_axis_format(axis_t idx, const std::string& format) {
    ImPlot::SetupAxisFormat(idx, format.c_str());
  }

  void plot::setup_axis_links(axis_t idx, double* min_lnk, double* max_lnk) {
    ImPlot::SetupAxisLinks(idx, min_lnk, max_lnk);
  }

  void plot::setup_axis_format(axis_t idx, formatter_t formatter, void* data) {
    ImPlot::SetupAxisFormat(idx, formatter, data);
  }

  void plot::setup_legend(location_t location, int flags) {
    ImPlot::SetupLegend(location, flags);
  }

  void plot::setup_finish() {
    ImPlot::SetupFinish();
  }

  void plot::plot_line(const std::string& label, const std::vector<f32_t>& values, double x_scale, double x_start, line_flags_t flags) {
    ImPlot::PlotLine(label.c_str(), values.data(), (int)values.size(), x_scale, x_start, flags);
  }

  void plot::plot_line(const std::string& label, const std::vector<f32_t>& xs, const std::vector<f32_t>& ys, line_flags_t flags) {
    if ( xs.size() != ys.size() || xs.empty() ) return;
    ImPlot::PlotLine(label.c_str(), xs.data(), ys.data(), (int)xs.size(), flags);
  }

  void plot::plot_line(const std::string& label, const f32_t* xs, const f32_t* ys, int count, line_flags_t flags) {
    ImPlot::PlotLine(label.c_str(), xs, ys, count, flags);
  }

  void plot::plot_scatter(const std::string& label, const std::vector<f32_t>& values, double x_scale, double x_start, scatter_flags_t flags) {
    ImPlot::PlotScatter(label.c_str(), values.data(), (int)values.size(), x_scale, x_start, flags);
  }

  void plot::plot_scatter(const std::string& label, const std::vector<f32_t>& xs, const std::vector<f32_t>& ys, scatter_flags_t flags) {
    if ( xs.size() != ys.size() || xs.empty() ) return;
    ImPlot::PlotScatter(label.c_str(), xs.data(), ys.data(), (int)xs.size(), flags);
  }

  void plot::plot_scatter(const std::string& label, const f32_t* xs, const f32_t* ys, int count, scatter_flags_t flags) {
    ImPlot::PlotScatter(label.c_str(), xs, ys, count, flags);
  }

  void plot::plot_bars(const std::string& label, const std::vector<f32_t>& values, double bar_size, double shift, bars_flags_t flags) {
    ImPlot::PlotBars(label.c_str(), values.data(), (int)values.size(), bar_size, shift, flags);
  }

  void plot::plot_bars(const std::string& label, const std::vector<f32_t>& xs, const std::vector<f32_t>& ys, double bar_size, bars_flags_t flags) {
    if ( xs.size() != ys.size() || xs.empty() ) return;
    ImPlot::PlotBars(label.c_str(), xs.data(), ys.data(), (int)xs.size(), bar_size, flags);
  }

  void plot::plot_shaded(const std::string& label, const std::vector<f32_t>& xs, const std::vector<f32_t>& ys, double y_ref, int flags) {
    if ( xs.size() != ys.size() || xs.empty() ) return;
    ImPlot::PlotShaded(label.c_str(), xs.data(), ys.data(), (int)xs.size(), y_ref, flags);
  }

  void plot::push_style_color(col_t idx, const fan::color& color) {
    ImPlot::PushStyleColor(idx, color);
  }

  void plot::pop_style_color(int count) {
    ImPlot::PopStyleColor(count);
  }

  void plot::push_style_var(int idx, f32_t val) {
    ImPlot::PushStyleVar(idx, val);
  }

  void plot::push_style_var(int idx, const fan::vec2& val) {
    ImPlot::PushStyleVar(idx, val);
  }

  void plot::pop_style_var(int count) {
    ImPlot::PopStyleVar(count);
  }

  void plot::set_next_line_style(const fan::color& col, f32_t weight) {
    ImPlot::SetNextLineStyle(col, weight);
  }

  void plot::set_next_fill_style(const fan::color& col, f32_t alpha_mod) {
    ImPlot::SetNextFillStyle(col, alpha_mod);
  }

  void plot::set_next_marker_style(marker_t marker, f32_t size, const fan::color& fill, f32_t weight, const fan::color& outline) {
    ImPlot::SetNextMarkerStyle(marker, size, fill, weight, outline);
  }

  fan::vec2 plot::get_plot_pos() {
    auto pos = ImPlot::GetPlotPos();
    return fan::vec2(pos.x, pos.y);
  }

  fan::vec2 plot::get_plot_size() {
    auto size = ImPlot::GetPlotSize();
    return fan::vec2(size.x, size.y);
  }

  bool plot::is_plot_hovered() {
    return ImPlot::IsPlotHovered();
  }

  bool plot::is_axis_hovered(axis_t axis) {
    return ImPlot::IsAxisHovered(axis);
  }

  fan::vec2 plot::pixels_to_plot(const fan::vec2& pix, axis_t x_axis, axis_t y_axis) {
    return ImPlot::PixelsToPlot(ImVec2(pix.x, pix.y), x_axis, y_axis);
  }

  fan::vec2 plot::plot_to_pixels(double x, double y, axis_t x_axis, axis_t y_axis) {
    auto result = ImPlot::PlotToPixels(x, y, x_axis, y_axis);
    return fan::vec2(result.x, result.y);
  }

  fan::vec2 plot::get_plot_mouse_pos(axis_t x_axis, axis_t y_axis) {
    return ImPlot::GetPlotMousePos(x_axis, y_axis);
  }

  void plot::annotation(double x, double y, const fan::color& col, const fan::vec2& pix_offset, bool clamp, const std::string& text) {
    ImPlot::Annotation(x, y, col, ImVec2(pix_offset.x, pix_offset.y), clamp, "%s", text.c_str());
  }

  void plot::tag_x(double x, const fan::color& col, const std::string& text) {
    if ( text.empty() ) {
      ImPlot::TagX(x, col, true);
    }
    else {
      ImPlot::TagX(x, col, "%s", text.c_str());
    }
  }

  void plot::tag_y(double y, const fan::color& col, const std::string& text) {
    if ( text.empty() ) {
      ImPlot::TagY(y, col, true);
    }
    else {
      ImPlot::TagY(y, col, "%s", text.c_str());
    }
  }

  void plot::plot_text(const std::string& text, double x, double y, const fan::vec2& pix_offset, int flags) {
    ImPlot::PlotText(text.c_str(), x, y, ImVec2(pix_offset.x, pix_offset.y), flags);
  }

  void plot::plot_dummy(const std::string& label_id, int flags) {
    ImPlot::PlotDummy(label_id.c_str(), flags);
  }

  fan::color plot::next_colormap_color() {
    auto color = ImPlot::NextColormapColor();
    return fan::color(color.x, color.y, color.z, color.w);
  }

  fan::color plot::get_last_item_color() {
    auto color = ImPlot::GetLastItemColor();
    return fan::color(color.x, color.y, color.z, color.w);
  }
  void plot::setup_axis_ticks(plot::axis_t axis, const double* values, int n_ticks, const char* const labels[], bool keep_default) {
    ImPlot::SetupAxisTicks(axis, values, n_ticks, labels, keep_default);
  }
  void plot::setup_axis_ticks(plot::axis_t axis, double v_min, double v_max, int n_ticks, const char* const labels[], bool keep_default) {
    ImPlot::SetupAxisTicks(axis, v_min, v_max, n_ticks, labels, keep_default);
  }

  void plot::push_plot_clip_rect(f32_t expand) {
    ImPlot::PushPlotClipRect(expand);
  }
  void plot::pop_plot_clip_rect() {
    ImPlot::PopPlotClipRect();
  }
} // namespace fan::graphics::gui

#if defined(fan_audio)
namespace fan::graphics::gui {
  bool audio_button(const std::string& label, fan::audio::piece_t piece_hover, fan::audio::piece_t piece_click, const fan::vec2& size) {
    ImGui::PushID(label.c_str());
    ImGuiStorage* storage = ImGui::GetStateStorage();
    ImGuiID id = ImGui::GetID("audio_button_prev_hovered");
    bool previously_hovered = storage->GetBool(id);

    bool pressed = ImGui::Button(label.c_str(), size);
    bool currently_hovered = ImGui::IsItemHovered();

    if ( currently_hovered && !previously_hovered ) {
      fan::audio::piece_t& piece = fan::audio::is_piece_valid(piece_hover) ? piece_hover : fan::audio::piece_hover;
      fan::audio::play(piece);
    }
    if ( pressed ) {
      fan::audio::piece_t& piece = fan::audio::is_piece_valid(piece_click) ? piece_click : fan::audio::piece_click;
      fan::audio::play(piece);
    }
    storage->SetBool(id, currently_hovered);

    ImGui::PopID();
    return pressed;
  }
}
#endif

#endif