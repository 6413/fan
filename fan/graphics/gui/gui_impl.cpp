module;

#include <fan/utility.h>
#include <fan/event/types.h>

#if defined(fan_gui)
  #include <fan/imgui/imgui.h>
  #include <fan/imgui/imgui_internal.h>
  #include <fan/imgui/imgui_impl_glfw.h>
  #include <fan/imgui/implot.h>
#endif

#include <string>
#include <functional>
#include <filesystem>
#include <coroutine>
#include <algorithm>

module fan.graphics.gui;

#if defined(fan_audio)
import fan.audio;
#endif

namespace fan::graphics::gui {
  bool begin(const std::string& window_name, bool* p_open, window_flags_t window_flags) {

    if (window_flags & window_flags_no_title_bar) {
      ImGuiWindowClass window_class;
      window_class.DockNodeFlagsOverrideSet = ImGuiDockNodeFlags_NoTabBar;
      ImGui::SetNextWindowClass(&window_class);
    }

    return ImGui::Begin(window_name.c_str(), p_open, window_flags);
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
  void table_setup_column(const std::string& label, ImGuiTableColumnFlags flags, f32_t init_width_or_weight, ImGuiID user_id) {
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
  void same_line(f32_t offset_from_start_x, f32_t spacing_w) {
    ImGui::SameLine(offset_from_start_x, spacing_w);
  }
  void new_line() {
    ImGui::NewLine();
  }
  ImGuiViewport* get_main_viewport() {
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
    return fan::window::is_mouse_down(mouse_button) && is_item_hovered(hovered_flags_rect_only);
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
  f32_t get_scroll_x() {
    return ImGui::GetScrollX();
  }
  f32_t get_scroll_y() {
    return ImGui::GetScrollY();
  }

  window_t::window_t(const std::string& window_name, bool* p_open, window_flags_t window_flags)
    : is_open(begin(window_name.c_str(), p_open, window_flags)) {}

  window_t::~window_t() {
    end();
  }

  window_t::operator bool() const {
    return is_open;
  }

  child_window_t::child_window_t(const std::string& window_name, const fan::vec2& size, child_window_flags_t window_flags)
    : is_open(ImGui::BeginChild(window_name.c_str(), size, window_flags)) {}

  child_window_t::~child_window_t() {
    ImGui::EndChild();
  }

  child_window_t::operator bool() const {
    return is_open;
  }

  table_t::table_t(const std::string& str_id, int columns, table_flags_t flags, const fan::vec2& outer_size, f32_t inner_width)
    : is_open(ImGui::BeginTable(str_id.c_str(), columns, flags, outer_size, inner_width)) {}

  table_t::~table_t() {
    ImGui::EndTable();
  }

  table_t::operator bool() const {
    return is_open;
  }

  bool button(const std::string& label, const fan::vec2& size) {
    return ImGui::Button(label.c_str(), size);
  }

  bool invisible_button(const std::string& label, const fan::vec2& size) {
    return ImGui::InvisibleButton(label.c_str(), size);
  }

  /// <summary>
  /// Draws the specified text, with its position influenced by other GUI elements.
  /// </summary>
  /// <param name="text">The text to draw.</param>
  /// <param name="color">The color of the text (defaults to white).</param>
  void text(const std::string& text, const fan::color& color) {
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::Text("%s", text.c_str());
    ImGui::PopStyleColor();
  }

  /// <summary>
  /// Draws the specified text, with its position influenced by other GUI elements.
  /// </summary>
  /// <param name="text">The text to draw.</param>
  /// <param name="color">The color of the text (defaults to white).</param>
  void text(const char* text, const fan::color& color) {
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::Text("%s", text);
    ImGui::PopStyleColor();
  }

  /// <summary>
  /// Draws the specified text at a given position on the screen.
  /// </summary>
  /// <param name="text">The text to draw.</param>
  /// <param name="position">The position of the text.</param>
  /// <param name="color">The color of the text (defaults to white).</param>
  void text_at(const std::string& text, const fan::vec2& position, const fan::color& color) {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    draw_list->AddText(position, color.get_imgui_color(), text.c_str());
  }

  void text_wrapped(const std::string& text, const fan::color& color) {
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::TextWrapped("%s", text.c_str());
    ImGui::PopStyleColor();
  }

  void text_unformatted(const std::string& text, const char* text_end) {
    ImGui::TextUnformatted(text.c_str(), text_end);
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
    draw_list->AddText(draw_position, color.get_imgui_color(), text.c_str());
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
      draw_list->AddText(screen_pos + offset, outline_color.get_imgui_color(), text.c_str());
    }

    draw_list->AddText(screen_pos, color.get_imgui_color(), text.c_str());
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

  //imgui_stdlib.cpp:
  int InputTextCallback(ImGuiInputTextCallbackData* data) {
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

  void spacing() {
    ImGui::Spacing();
  }

  io_t& get_io() {
    return ImGui::GetIO();
  }

  void tree_pop() {
    ImGui::TreePop();
  }

  bool is_item_toggled_open() {
    return ImGui::IsItemToggledOpen();
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

  void dummy(const fan::vec2& size) {
    ImGui::Dummy(size);
  }

  draw_list_t* get_window_draw_list() {
    return ImGui::GetWindowDrawList();
  }

  draw_list_t* get_background_draw_list() {
    return ImGui::GetBackgroundDrawList();
  }

  const char* item_getter1(const std::vector<std::string>& items, int index) {
    if ( index >= 0 && index < (int)items.size() ) {
      return items[index].c_str();
    }
    return "N/A";
  }

  void set_viewport(fan::graphics::viewport_t viewport) {
    ImVec2 child_pos = ImGui::GetWindowPos();
    ImVec2 child_size = ImGui::GetWindowSize();
    ImVec2 mainViewportPos = ImGui::GetMainViewport()->Pos;

    fan::vec2 windowPosRelativeToMainViewport;
    windowPosRelativeToMainViewport.x = child_pos.x - mainViewportPos.x;
    windowPosRelativeToMainViewport.y = child_pos.y - mainViewportPos.y;

    fan::vec2 viewport_size = fan::vec2(child_size.x, child_size.y);
    fan::vec2 viewport_pos = windowPosRelativeToMainViewport;

    fan::vec2 window_size = fan::graphics::get_window().get_size();
    fan::graphics::viewport_set(
      viewport,
      viewport_pos,
      viewport_size
    );
  }

  void set_viewport(const fan::graphics::render_view_t& render_view) {
    set_viewport(render_view.viewport);

    ImVec2 child_size = ImGui::GetWindowSize();
    fan::vec2 viewport_size = fan::vec2(child_size.x, child_size.y);
    fan::graphics::camera_set_ortho(
      render_view.camera,
      fan::vec2(0, viewport_size.x),
      fan::vec2(0, viewport_size.y)
    );
  }

  window_handle_t* get_current_window() {
    return ImGui::GetCurrentWindow();
  }

#if defined(fan_audio)
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
#endif

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

  void image(fan::graphics::image_t img, const fan::vec2& size, const fan::vec2& uv0, const fan::vec2& uv1, const fan::color& tint_col, const fan::color& border_col) {
    ImGui::Image((ImTextureID)fan::graphics::image_get_handle(img), size, uv0, uv1, tint_col, border_col);
  }

  bool image_button(const std::string& str_id, fan::graphics::image_t img, const fan::vec2& size, const fan::vec2& uv0, const fan::vec2& uv1, int frame_padding, const fan::color& bg_col, const fan::color& tint_col) {
    return ImGui::ImageButton(str_id.c_str(), (ImTextureID)fan::graphics::image_get_handle(img), size, uv0, uv1, bg_col, tint_col);
  }

  bool image_text_button(fan::graphics::image_t img, const std::string& text, const fan::color& color, const fan::vec2& size, const fan::vec2& uv0, const fan::vec2& uv1, int frame_padding, const fan::color& bg_col, const fan::color& tint_col) {
    bool ret = ImGui::ImageButton(text.c_str(), (ImTextureID)fan::graphics::image_get_handle(img), size, uv0, uv1, bg_col, tint_col);
    ImVec2 text_size = ImGui::CalcTextSize(text.c_str());
    ImVec2 pos = ImGui::GetItemRectMin() + (ImGui::GetItemRectMax() - ImGui::GetItemRectMin()) / 2 - text_size / 2;
    ImGui::GetWindowDrawList()->AddText(pos, ImGui::GetColorU32(color), text.c_str());
    return ret;
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

  bool toggle_image_button(const std::string& char_id, fan::graphics::image_t image, const fan::vec2& size, bool* toggle) {
    bool clicked = false;

    ImVec4 tintColor = ImVec4(1, 1, 1, 1);
    if ( *toggle ) {
      tintColor = ImVec4(0.3f, 0.3f, 0.3f, 1.0f);
    }

    if ( image_button(char_id, image, size, ImVec2(0, 0), ImVec2(1, 1), -1, ImVec4(0, 0, 0, 0), tintColor) ) {
      *toggle = !(*toggle);
      clicked = true;
    }

    return clicked;
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

  // untested
  void image_rotated(fan::graphics::image_t image, const fan::vec2& size, int angle, const fan::vec2& uv0, const fan::vec2& uv1, const fan::color& tint_col, const fan::color& border_col) {
    IM_ASSERT(angle % 90 == 0);
    fan::vec2 _uv0, _uv1, _uv2, _uv3;

    switch ( angle % 360 ) {
    case 0:
      gui::image(image, size, uv0, uv1, tint_col, border_col);
      return;
    case 180:
      gui::image(image, size, uv1, uv0, tint_col, border_col);
      return;
    case 90:
      _uv3 = uv0;
      _uv1 = uv1;
      _uv0 = fan::vec2(uv1.x, uv0.y);
      _uv2 = fan::vec2(uv0.x, uv1.y);
      break;
    case 270:
      _uv1 = uv0;
      _uv3 = uv1;
      _uv0 = fan::vec2(uv0.x, uv1.y);
      _uv2 = fan::vec2(uv1.x, uv0.y);
      break;
    }

    ImGuiWindow* window = ImGui::GetCurrentWindow();
    if ( window->SkipItems )
      return;

    fan::vec2 _size(size.y, size.x); // swapped for rotation
    fan::vec2 cursor_pos = *(fan::vec2*)&window->DC.CursorPos;
    fan::vec2 bb_max = cursor_pos + _size;
    if ( border_col.a > 0.0f ) {
      bb_max += fan::vec2(2, 2);
    }

    ImRect bb(*(ImVec2*)&cursor_pos, *(ImVec2*)&bb_max);
    ImGui::ItemSize(bb);
    if ( !ImGui::ItemAdd(bb, 0) )
      return;

    if ( border_col.a > 0.0f ) {
      window->DrawList->AddRect(*(ImVec2*)&bb.Min, *(ImVec2*)&bb.Max, ImGui::GetColorU32(border_col), 0.0f);
      fan::vec2 x0 = cursor_pos + fan::vec2(1, 1);
      fan::vec2 x2 = bb_max - fan::vec2(1, 1);
      fan::vec2 x1 = fan::vec2(x2.x, x0.y);
      fan::vec2 x3 = fan::vec2(x0.x, x2.y);

      window->DrawList->AddImageQuad(
        (ImTextureID)fan::graphics::image_get_handle(image),
        *(ImVec2*)&x0, *(ImVec2*)&x1, *(ImVec2*)&x2, *(ImVec2*)&x3,
        *(ImVec2*)&_uv0, *(ImVec2*)&_uv1, *(ImVec2*)&_uv2, *(ImVec2*)&_uv3,
        ImGui::GetColorU32(tint_col)
      );
    }
    else {
      fan::vec2 x0 = cursor_pos;
      fan::vec2 x1 = fan::vec2(bb_max.x, cursor_pos.y);
      fan::vec2 x2 = bb_max;
      fan::vec2 x3 = fan::vec2(cursor_pos.x, bb_max.y);

      window->DrawList->AddImageQuad(
        (ImTextureID)fan::graphics::image_get_handle(image),
        *(ImVec2*)&x0, *(ImVec2*)&x1, *(ImVec2*)&x2, *(ImVec2*)&x3,
        *(ImVec2*)&_uv0, *(ImVec2*)&_uv1, *(ImVec2*)&_uv2, *(ImVec2*)&_uv3,
        ImGui::GetColorU32(tint_col)
      );
    }
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

  imgui_element_nr_t::imgui_element_nr_t(const imgui_element_nr_t& nr) : imgui_element_nr_t() {
    if ( nr.is_invalid() ) {
      return;
    }
    init();
  }

  imgui_element_nr_t::imgui_element_nr_t(imgui_element_nr_t&& nr) {
    NRI = nr.NRI;
    nr.invalidate_soft();
  }

  imgui_element_nr_t::~imgui_element_nr_t() {
    invalidate();
  }

  imgui_element_nr_t& imgui_element_nr_t::operator=(const imgui_element_nr_t& id) {
    if ( !is_invalid() ) {
      invalidate();
    }
    if ( id.is_invalid() ) {
      return *this;
    }

    if ( this != &id ) {
      init();
    }
    return *this;
  }

  imgui_element_nr_t& imgui_element_nr_t::operator=(imgui_element_nr_t&& id) {
    if ( !is_invalid() ) {
      invalidate();
    }
    if ( id.is_invalid() ) {
      return *this;
    }

    if ( this != &id ) {
      if ( !is_invalid() ) {
        invalidate();
      }
      NRI = id.NRI;

      id.invalidate_soft();
    }
    return *this;
  }

  void imgui_element_nr_t::init() {
    *(base_t*)this = fan::graphics::get_gui_draw_cbs().NewNodeLast();
  }

  bool imgui_element_nr_t::is_invalid() const {
    return fan::graphics::gui_draw_cb_inric(*this);
  }

  void imgui_element_nr_t::invalidate_soft() {
    *(base_t*)this = fan::graphics::get_gui_draw_cbs().gnric();
  }

  void imgui_element_nr_t::invalidate() {
    if ( is_invalid() ) {
      return;
    }
    fan::graphics::get_gui_draw_cbs().unlrec(*this);
    *(base_t*)this = fan::graphics::get_gui_draw_cbs().gnric();
  }

  void imgui_element_nr_t::set(const auto& lambda) {
    fan::graphics::get_gui_draw_cbs()[*this] = lambda;
  }

  imgui_element_t::imgui_element_t(const auto& lambda) {
    imgui_element_nr_t::init();
    imgui_element_nr_t::set(lambda);
  }

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

  void plot::setup_axis_links(ImAxis idx, double* min_lnk, double* max_lnk) {
    ImPlot::SetupAxisLinks(idx, min_lnk, max_lnk);
  }

  void plot::setup_axis_format(ImAxis idx, ImPlotFormatter formatter, void* data) {
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

  void shape_properties(const fan::graphics::shape_t& shape) {
    switch ( shape.get_shape_type() ) {
    case fan::graphics::shape_type_t::particles:
    {
      auto& ri = *(fan::graphics::shapes::particles_t::ri_t*)shape.GetData(fan::graphics::g_shapes->shaper);

      const char* items[] = {"circle", "rectangle"};
      int current_shape = ri.shape;
      if ( combo("shape", &current_shape, items, std::size(items)) ) {
        ri.shape = current_shape;
      }
      if ( color_edit4("particle color", &ri.color) ) {

      }
      if ( slider("position", &ri.position, -2000.0f, 2000.0f) ) {
      }
      if ( slider("size", &ri.size, 0.0f, 1000.0f) ) {
      }
      if ( slider("alive_time", &ri.alive_time, 0.0f, 10e+9f) ) {
      }
      if ( ri.shape == fan::graphics::shapes::particles_t::shapes_e::rectangle ) {
        if ( slider("gap_size", &ri.gap_size, -1000.0f, 1000.0f) ) {
        }
      }
      if ( ri.shape == fan::graphics::shapes::particles_t::shapes_e::rectangle ) {
        if ( slider("max_spread_size", &ri.max_spread_size, -10000.0f, 10000.0f) ) {
        }
      }
      if ( slider("position_velocity", &ri.position_velocity, -10000.0f, 10000.0f) ) {
      }
      if ( slider("size_velocity", &ri.size_velocity, -100.0f, 100.0f) ) {
      }
      if ( slider("angle_velocity", &ri.angle_velocity, -fan::math::pi / 2, fan::math::pi / 2) ) {
      }
      if ( slider("count", &ri.count, 1.0f, 5000.0f) ) {
      }
      if ( slider("begin_angle", &ri.begin_angle, -fan::math::pi / 2, fan::math::pi / 2) ) {
      }
      if ( slider("end_angle", &ri.end_angle, -fan::math::pi / 2, fan::math::pi / 2) ) {
      }
      if ( slider("angle", &ri.angle, -fan::math::pi / 2, fan::math::pi / 2) ) {
      }
      break;
    }
    }
  }



  content_browser_t::content_browser_t() {
    search_buffer.resize(32);
    current_directory = std::filesystem::path(asset_path);
    update_directory_cache();
  }

  content_browser_t::content_browser_t(bool no_init) {

  }

  content_browser_t::content_browser_t(const std::wstring& path) {
    init(path);
  }

  void content_browser_t::init(const std::wstring& path) {
    search_buffer.resize(32);
    current_directory = asset_path / std::filesystem::path(path);
    update_directory_cache();
  }

  void content_browser_t::clear_selection() {
    selection_state.selected_indices.clear();
    for ( auto& file : directory_cache ) {
      file.is_selected = false;
    }
    for ( auto& file : search_state.found_files ) {
      file.is_selected = false;
    }
  }

  bool content_browser_t::is_point_in_rect(ImVec2 point, ImVec2 rect_min, ImVec2 rect_max) {
    return point.x >= rect_min.x && point.x <= rect_max.x &&
      point.y >= rect_min.y && point.y <= rect_max.y;
  }

  void content_browser_t::handle_rectangular_selection() {
    ImGuiIO& io = ImGui::GetIO();
    selection_state.ctrl_held = io.KeyCtrl;

    if ( ImGui::IsMouseClicked(ImGuiMouseButton_Left) ) {
      bool can_start_selection = !ImGui::IsAnyItemActive() &&
        ImGui::IsWindowHovered() &&
        !ImGui::IsAnyItemHovered();

      if ( can_start_selection ) {
        selection_state.is_selecting = true;
        selection_state.selection_start = ImGui::GetMousePos();
        selection_state.selection_end = selection_state.selection_start;

        clear_selection();
      }
    }

    if ( selection_state.is_selecting && ImGui::IsMouseDown(ImGuiMouseButton_Left) ) {
      selection_state.selection_end = ImGui::GetMousePos();

      bool showing_search_results = !search_state.found_files.empty() && !search_buffer.empty();

      ImVec2 rect_min = ImVec2(
        std::min(selection_state.selection_start.x, selection_state.selection_end.x),
        std::min(selection_state.selection_start.y, selection_state.selection_end.y)
      );
      ImVec2 rect_max = ImVec2(
        std::max(selection_state.selection_start.x, selection_state.selection_end.x),
        std::max(selection_state.selection_start.y, selection_state.selection_end.y)
      );

      if ( showing_search_results ) {
        update_search_sorted_cache();
      }
      else {
        update_sorted_cache();
      }
    }

    if ( selection_state.is_selecting && ImGui::IsMouseReleased(ImGuiMouseButton_Left) ) {
      selection_state.is_selecting = false;
    }

    if ( selection_state.is_selecting ) {
      ImDrawList* draw_list = ImGui::GetWindowDrawList();
      ImVec2 rect_min = ImVec2(
        std::min(selection_state.selection_start.x, selection_state.selection_end.x),
        std::min(selection_state.selection_start.y, selection_state.selection_end.y)
      );
      ImVec2 rect_max = ImVec2(
        std::max(selection_state.selection_start.x, selection_state.selection_end.x),
        std::max(selection_state.selection_start.y, selection_state.selection_end.y)
      );

      draw_list->AddRect(rect_min, rect_max, IM_COL32(100, 150, 255, 200), 0.0f, 0, 2.0f);
      draw_list->AddRectFilled(rect_min, rect_max, IM_COL32(100, 150, 255, 50));
    }
  }

  void content_browser_t::update_directory_cache() {
    search_iterator.stop();
    search_state.is_searching = false;
    search_state.found_files.clear();
    search_state.search_cache_dirty = true;
    search_state.cache_dirty = true;
    while ( !search_state.pending_directories.empty() ) {
      search_state.pending_directories.pop();
    }

    std::fill(search_buffer.begin(), search_buffer.end(), '\0');

    clear_selection();

    invalidate_cache();

    for ( auto& img : directory_cache ) {
      if ( fan::graphics::is_image_valid(img.preview_image) ) {
        fan::graphics::image_unload(img.preview_image);
      }
    }
    directory_cache.clear();

    if ( !directory_iterator.callback ) {
      directory_iterator.sort_alphabetically = true;
      directory_iterator.callback = [this](const std::filesystem::directory_entry& entry) -> fan::event::task_t {
        file_info_t file_info;
        std::filesystem::path relative_path;
        try {
          // SLOW
          relative_path = std::filesystem::relative(entry, asset_path);
        }
        catch ( const std::exception& e ) {
          fan::print("exception came", e.what());
        }

        file_info.filename = relative_path.filename().string();
        file_info.item_path = relative_path.wstring();
        file_info.is_directory = entry.is_directory();
        file_info.is_selected = false;
        //fan::print(get_file_extension(path.path().string()));
        if ( fan::image::valid(entry.path().string()) ) {
          file_info.preview_image = fan::graphics::image_load(entry.path().string());
        }
        directory_cache.push_back(file_info);
        invalidate_cache();
        co_return;
      };
    }

    fan::io::async_directory_iterate(
      &directory_iterator,
      current_directory.string()
    );
  }

  void content_browser_t::invalidate_cache() {
    search_state.cache_dirty = true;
    search_state.sorted_cache.clear();
  }

  int content_browser_t::get_pressed_key() {
    auto& style = ImGui::GetStyle();
    if ( style.DisabledAlpha != style.Alpha && ImGui::IsWindowFocused() ) {
      for ( int i = ImGuiKey_A; i <= ImGuiKey_Z; ++i ) {
        if ( ImGui::IsKeyPressed((ImGuiKey)i, false) ) {
          return (i - ImGuiKey_A) + 'A';
        }
      }
    }
    return -1;
  }

  void content_browser_t::handle_keyboard_navigation(const std::string& filename, int pressed_key) {
    if ( pressed_key != -1 && !filename.empty() ) {
      if ( std::tolower(filename[0]) == std::tolower(pressed_key) ) {
        ImGui::SetScrollHereY();
      }
    }
  }

  void content_browser_t::handle_right_click(const std::string& filename) {
    if ( ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right) ) {
      item_right_clicked = true;
      item_right_clicked_name = filename;
      item_right_clicked_name.erase(
        std::remove_if(item_right_clicked_name.begin(), item_right_clicked_name.end(),
          [](unsigned char c) { return std::isspace(c); }),
        item_right_clicked_name.end()
      );
    }
  }

  void content_browser_t::process_next_directory() {
    if ( !search_state.pending_directories.empty() ) {
      std::string next_dir = search_state.pending_directories.front();
      search_state.pending_directories.pop();
      fan::io::async_directory_iterate(&search_iterator, next_dir);
    }
    else {
      search_state.is_searching = false;
    }
  }

  void content_browser_t::start_search(const std::string& query, bool recursive) {
    search_iterator.stop();

    search_state.query = query;
    search_state.is_recursive = recursive;
    search_state.found_files.clear();
    search_state.search_cache_dirty = true;

    while ( !search_state.pending_directories.empty() ) {
      search_state.pending_directories.pop();
    }

    if ( query.empty() ) {
      search_state.is_searching = false;
      search_state.found_files.clear();
      return;
    }

    search_state.is_searching = true;

    search_iterator.sort_alphabetically = true;
    search_iterator.callback = [this](const std::filesystem::directory_entry& entry) -> fan::event::task_t {
      try {
        std::string filename = entry.path().filename().string();

        std::string query_lower = search_state.query;
        std::string filename_lower = filename;
        static auto tl = [](unsigned char c) { return std::tolower(c); };
        std::transform(query_lower.begin(), query_lower.end(), query_lower.begin(), tl);
        std::transform(filename_lower.begin(), filename_lower.end(), filename_lower.begin(), tl);

        bool matches = filename_lower.find(query_lower) != std::string::npos;

        if ( matches ) {
          file_info_t file_info;
          std::filesystem::path relative_path;
          try {
            relative_path = std::filesystem::relative(entry.path(), asset_path);
          }
          catch ( const std::exception& ) {
            relative_path = entry.path().filename();
          }

          file_info.filename = filename;
          file_info.item_path = relative_path.wstring();
          file_info.is_directory = entry.is_directory();
          file_info.is_selected = false;

          if ( fan::image::valid(entry.path().string()) ) {
            file_info.preview_image = fan::graphics::image_load(entry.path().string());
          }

          search_state.found_files.push_back(file_info);
          search_state.search_cache_dirty = true;
        }

        if ( entry.is_directory() && search_state.is_recursive ) {
          search_state.pending_directories.push(entry.path().string());
        }
      }
      catch ( ... ) {

      }

      co_return;
    };

    std::string search_root = current_directory.string();
    fan::io::async_directory_iterate(&search_iterator, search_root);
  }

  void content_browser_t::update_sorted_cache() {
    if ( !search_state.cache_dirty ) return;

    search_state.sorted_cache.clear();
    for ( std::size_t i = 0; i < directory_cache.size(); ++i ) {
      auto& file_info = directory_cache[i];
      if ( search_buffer.empty() || file_info.filename.find(search_buffer.data()) != std::string::npos ) {
        search_state.sorted_cache.push_back({file_info, i});
      }
    }

    std::sort(search_state.sorted_cache.begin(), search_state.sorted_cache.end(),
      [](const auto& a, const auto& b) {
      return a.first.is_directory > b.first.is_directory;
    });

    search_state.cache_dirty = false;
  }

  void content_browser_t::update_search_sorted_cache() {
    if ( !search_state.search_cache_dirty ) return;

    search_state.sorted_search_cache.clear();
    for ( std::size_t i = 0; i < search_state.found_files.size(); ++i ) {
      search_state.sorted_search_cache.push_back({search_state.found_files[i], i});
    }

    std::sort(search_state.sorted_search_cache.begin(), search_state.sorted_search_cache.end(),
      [](const auto& a, const auto& b) {
      return a.first.is_directory > b.first.is_directory;
    });

    search_state.search_cache_dirty = false;
  }

  void content_browser_t::render() {
    item_right_clicked = false;
    item_right_clicked_name.clear();

    if ( search_state.is_searching && !search_iterator.operation_in_progress ) {
      if ( search_iterator.current_index >= search_iterator.entries.size() ) {
        if ( !search_state.pending_directories.empty() ) {
          process_next_directory();
        }
        else {
          search_state.is_searching = false;
        }
      }
    }

    ImGuiStyle& style = ImGui::GetStyle();
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 16.0f));

    ImGuiWindowClass window_class;
    ImGui::SetNextWindowClass(&window_class);

    if ( ImGui::Begin("Content Browser", 0, ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoTitleBar) ) {
      if ( ImGui::BeginMenuBar() ) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.f, 0.f, 0.f, 0.f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.f, 0.f, 0.f, 0.f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.3f, 0.3f));

        if ( image_button("##icon_arrow_left", icon_arrow_left, fan::vec2(32)) ) {
          /*if (!std::filesystem::equivalent(current_directory, asset_path)) {
          current_directory = current_directory.parent_path();
          update_directory_cache();
          }*/
          auto absolute_path = std::filesystem::canonical(std::filesystem::absolute(current_directory));
          if ( absolute_path.has_parent_path() ) {
            current_directory = std::filesystem::canonical(absolute_path.parent_path());
            update_directory_cache();
          }
        }
        ImGui::SameLine();
        image_button("##icon_arrow_right", icon_arrow_right, fan::vec2(32));
        ImGui::SameLine();
        ImGui::PopStyleColor(3);

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.f, 0.f, 0.f, 0.f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.f, 0.f, 0.f, 0.f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.3f, 0.3f));

        auto image_list = std::to_array({icon_files_list, icon_files_big_thumbnail});
        fan::vec2 bc = get_position_bottom_corner();
        bc.x -= ImGui::GetWindowPos().x;
        ImGui::SetCursorPosX(bc.x / 2);

        fan::vec2 button_sizes = 32;
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - (button_sizes.x * 2 + style.ItemSpacing.x) * image_list.size());

        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 20.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 7.0f));
        f32_t y_pos = ImGui::GetCursorPosY() + ImGui::GetStyle().WindowPadding.y;
        ImGui::SetCursorPosY(y_pos);

        static char old_search[256] = {0};
        bool search_changed = false;
        if ( ImGui::InputText("##content_browser_search", search_buffer.data(), search_buffer.size()) ) {
          search_changed = true;
        }

        if ( search_changed || strcmp(old_search, search_buffer.data()) != 0 ) {
          strcpy(old_search, search_buffer.data());
          start_search(search_buffer.data(), true);
        }

        if ( search_state.is_searching ) {
          ImGui::SameLine();
          ImGui::Text("Searching... (%zu found)", search_state.found_files.size());
        }

        ImGui::PopStyleVar(2);
        toggle_image_button(image_list, button_sizes, (int*)&current_view_mode);
        ImGui::PopStyleColor(3);
        ImGui::EndMenuBar();
      }

      handle_rectangular_selection();

      switch ( current_view_mode ) {
      case view_mode_large_thumbnails:
        render_large_thumbnails_view();
        break;
      case view_mode_list:
        render_list_view();
        break;
      default:
        break;
      }
    }

    ImGui::PopStyleVar(1);
    ImGui::End();
  }

  void content_browser_t::render_large_thumbnails_view() {
    f32_t thumbnail_size = 128.0f;
    f32_t panel_width = ImGui::GetContentRegionAvail().x;
    int column_count = std::max((int)(panel_width / (thumbnail_size + padding)), 1);

    ImGui::Columns(column_count, 0, false);
    int pressed_key = get_pressed_key();

    bool showing_search_results = !search_state.found_files.empty() && !search_buffer.empty();

    if ( showing_search_results ) {
      update_search_sorted_cache();
      auto& sorted_files = search_state.sorted_search_cache;

      for ( const auto& [file_info, original_index] : sorted_files ) {
        handle_keyboard_navigation(file_info.filename, pressed_key);

        std::string unique_id = "search_" + std::to_string(original_index) + "_" + file_info.filename;

        ImGui::PushID(unique_id.c_str());

        ImVec2 item_pos = ImGui::GetCursorScreenPos();

        bool is_currently_selected = search_state.found_files[original_index].is_selected;

        if ( is_currently_selected ) {
          ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.4f, 0.6f, 1.0f, 0.3f));
          ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.4f, 0.6f, 1.0f, 0.4f));
          ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.4f, 0.6f, 1.0f, 0.5f));
        }
        else {
          ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
          ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.3f, 0.2f));
          ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.5f, 0.5f, 0.5f, 0.3f));
        }

        bool item_clicked = image_button(
          "##" + unique_id,
          fan::graphics::is_image_valid(file_info.preview_image) ? file_info.preview_image
          : file_info.is_directory ? icon_directory : file_info.filename.ends_with(".json") ? icon_object : icon_file,
          ImVec2(thumbnail_size, thumbnail_size)
        );

        if ( item_clicked ) {
          ImGuiIO& io = ImGui::GetIO();
          if ( io.KeyCtrl ) {
            search_state.found_files[original_index].is_selected = !search_state.found_files[original_index].is_selected;
          }
          else {
            for ( auto& f : search_state.found_files ) f.is_selected = false;
            search_state.found_files[original_index].is_selected = true;
          }
        }

        if ( selection_state.is_selecting ) {
          ImVec2 rect_min = ImVec2(
            std::min(selection_state.selection_start.x, selection_state.selection_end.x),
            std::min(selection_state.selection_start.y, selection_state.selection_end.y)
          );
          ImVec2 rect_max = ImVec2(
            std::max(selection_state.selection_start.x, selection_state.selection_end.x),
            std::max(selection_state.selection_start.y, selection_state.selection_end.y)
          );

          ImVec2 item_min = item_pos;
          ImVec2 item_max = ImVec2(item_pos.x + thumbnail_size, item_pos.y + thumbnail_size);

          bool overlaps = !(rect_max.x < item_min.x || rect_min.x > item_max.x ||
            rect_max.y < item_min.y || rect_min.y > item_max.y);

          if ( overlaps ) {
            search_state.found_files[original_index].is_selected = true;
          }
          else if ( !selection_state.ctrl_held ) {
            search_state.found_files[original_index].is_selected = false;
          }
        }

        handle_right_click(file_info.filename);
        handle_item_interaction(file_info, original_index);

        ImGui::PopStyleColor(3);
        ImGui::TextWrapped("%s", file_info.filename.c_str());
        ImGui::NextColumn();
        ImGui::PopID();
      }
    }
    else {
      update_sorted_cache();
      auto& sorted_files = search_state.sorted_cache;

      for ( const auto& [file_info, original_index] : sorted_files ) {
        handle_keyboard_navigation(file_info.filename, pressed_key);

        ImGui::PushID(file_info.filename.c_str());

        ImVec2 item_pos = ImGui::GetCursorScreenPos();

        bool is_currently_selected = directory_cache[original_index].is_selected;

        if ( is_currently_selected ) {
          ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.4f, 0.6f, 1.0f, 0.3f));
          ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.4f, 0.6f, 1.0f, 0.4f));
          ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.4f, 0.6f, 1.0f, 0.5f));
        }
        else {
          ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
          ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.3f, 0.2f));
          ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.5f, 0.5f, 0.5f, 0.3f));
        }

        bool item_clicked = image_button(
          "##" + file_info.filename,
          fan::graphics::is_image_valid(file_info.preview_image) ? file_info.preview_image
          : file_info.is_directory ? icon_directory : file_info.filename.ends_with(".json") ? icon_object : icon_file,
          ImVec2(thumbnail_size, thumbnail_size)
        );

        if ( item_clicked ) {
          ImGuiIO& io = ImGui::GetIO();
          if ( io.KeyCtrl ) {
            directory_cache[original_index].is_selected = !directory_cache[original_index].is_selected;
          }
          else {
            for ( auto& f : directory_cache ) f.is_selected = false;
            directory_cache[original_index].is_selected = true;
          }
        }

        if ( selection_state.is_selecting ) {
          ImVec2 rect_min = ImVec2(
            std::min(selection_state.selection_start.x, selection_state.selection_end.x),
            std::min(selection_state.selection_start.y, selection_state.selection_end.y)
          );
          ImVec2 rect_max = ImVec2(
            std::max(selection_state.selection_start.x, selection_state.selection_end.x),
            std::max(selection_state.selection_start.y, selection_state.selection_end.y)
          );

          ImVec2 item_min = item_pos;
          ImVec2 item_max = ImVec2(item_pos.x + thumbnail_size, item_pos.y + thumbnail_size);

          bool overlaps = !(rect_max.x < item_min.x || rect_min.x > item_max.x ||
            rect_max.y < item_min.y || rect_min.y > item_max.y);

          if ( overlaps ) {
            directory_cache[original_index].is_selected = true;
          }
          else if ( !selection_state.ctrl_held ) {
            directory_cache[original_index].is_selected = false;
          }
        }

        handle_right_click(file_info.filename);
        handle_item_interaction(file_info, original_index);

        ImGui::PopStyleColor(3);
        ImGui::TextWrapped("%s", file_info.filename.c_str());
        ImGui::NextColumn();
        ImGui::PopID();
      }
    }

    ImGui::Columns(1);
  }

  void content_browser_t::render_list_view() {
    if ( ImGui::BeginTable("##FileTable", 1,
      ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY |
      ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV |
      ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable |
      ImGuiTableFlags_Sortable) ) {

      ImGui::TableSetupColumn("##Filename", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableHeadersRow();

      int pressed_key = get_pressed_key();
      bool showing_search_results = !search_state.found_files.empty() && !search_buffer.empty();

      if ( showing_search_results ) {
        update_search_sorted_cache();
        auto& sorted_files = search_state.sorted_search_cache;

        for ( const auto& [file_info, original_index] : sorted_files ) {
          handle_keyboard_navigation(file_info.filename, pressed_key);

          ImGui::TableNextRow();
          ImGui::TableSetColumnIndex(0);

          ImVec2 row_pos = ImGui::GetCursorScreenPos();
          f32_t row_height = ImGui::GetTextLineHeightWithSpacing();

          fan::vec2 cursor_pos = fan::vec2(ImGui::GetWindowPos()) + fan::vec2(ImGui::GetCursorPos()) +
            fan::vec2(ImGui::GetScrollX(), -ImGui::GetScrollY());
          fan::vec2 image_size = ImVec2(thumbnail_size / 4, thumbnail_size / 4);

          std::string space;
          while ( ImGui::CalcTextSize(space.c_str()).x < image_size.x ) {
            space += " ";
          }
          auto str = space + file_info.filename;

          std::string unique_id = "search_" + std::to_string(original_index) + "_" + str;

          bool item_clicked = ImGui::Selectable(unique_id.c_str(), search_state.found_files[original_index].is_selected, ImGuiSelectableFlags_SpanAllColumns);

          if ( item_clicked ) {
            ImGuiIO& io = ImGui::GetIO();
            if ( io.KeyCtrl ) {
              search_state.found_files[original_index].is_selected = !search_state.found_files[original_index].is_selected;
            }
            else {
              for ( auto& f : search_state.found_files ) f.is_selected = false;
              search_state.found_files[original_index].is_selected = true;
            }
          }

          if ( selection_state.is_selecting ) {
            ImVec2 rect_min = ImVec2(
              std::min(selection_state.selection_start.x, selection_state.selection_end.x),
              std::min(selection_state.selection_start.y, selection_state.selection_end.y)
            );
            ImVec2 rect_max = ImVec2(
              std::max(selection_state.selection_start.x, selection_state.selection_end.x),
              std::max(selection_state.selection_start.y, selection_state.selection_end.y)
            );

            ImVec2 row_min = row_pos;
            ImVec2 row_max = ImVec2(row_pos.x + ImGui::GetContentRegionAvail().x, row_pos.y + row_height);

            bool overlaps = !(rect_max.x < row_min.x || rect_min.x > row_max.x ||
              rect_max.y < row_min.y || rect_min.y > row_max.y);

            if ( overlaps ) {
              search_state.found_files[original_index].is_selected = true;
            }
            else if ( !selection_state.ctrl_held ) {
              search_state.found_files[original_index].is_selected = false;
            }
          }

          handle_right_click(str);

          ImTextureID texture_id;
          if ( fan::graphics::is_image_valid(file_info.preview_image) ) {
            texture_id = (ImTextureID)fan::graphics::image_get_handle(file_info.preview_image);
          }
          else {
            texture_id = (ImTextureID)fan::graphics::image_get_handle(
              file_info.is_directory ? icon_directory : icon_file
            );
          }
          ImGui::GetWindowDrawList()->AddImage(texture_id, cursor_pos, cursor_pos + image_size);

          handle_item_interaction(file_info, original_index);
        }
      }
      else {
        update_sorted_cache();
        auto& sorted_files = search_state.sorted_cache;

        for ( const auto& [file_info, original_index] : sorted_files ) {
          handle_keyboard_navigation(file_info.filename, pressed_key);

          ImGui::TableNextRow();
          ImGui::TableSetColumnIndex(0);

          ImVec2 row_pos = ImGui::GetCursorScreenPos();
          f32_t row_height = ImGui::GetTextLineHeightWithSpacing();

          fan::vec2 cursor_pos = fan::vec2(ImGui::GetWindowPos()) + fan::vec2(ImGui::GetCursorPos()) +
            fan::vec2(ImGui::GetScrollX(), -ImGui::GetScrollY());
          fan::vec2 image_size = ImVec2(thumbnail_size / 4, thumbnail_size / 4);

          std::string space;
          while ( ImGui::CalcTextSize(space.c_str()).x < image_size.x ) {
            space += " ";
          }
          auto str = space + file_info.filename;

          bool item_clicked = ImGui::Selectable(str.c_str(), directory_cache[original_index].is_selected, ImGuiSelectableFlags_SpanAllColumns);

          if ( item_clicked ) {
            ImGuiIO& io = ImGui::GetIO();
            if ( io.KeyCtrl ) {
              directory_cache[original_index].is_selected = !directory_cache[original_index].is_selected;
            }
            else {
              for ( auto& f : directory_cache ) f.is_selected = false;
              directory_cache[original_index].is_selected = true;
            }
          }

          if ( selection_state.is_selecting ) {
            ImVec2 rect_min = ImVec2(
              std::min(selection_state.selection_start.x, selection_state.selection_end.x),
              std::min(selection_state.selection_start.y, selection_state.selection_end.y)
            );
            ImVec2 rect_max = ImVec2(
              std::max(selection_state.selection_start.x, selection_state.selection_end.x),
              std::max(selection_state.selection_start.y, selection_state.selection_end.y)
            );

            ImVec2 row_min = row_pos;
            ImVec2 row_max = ImVec2(row_pos.x + ImGui::GetContentRegionAvail().x, row_pos.y + row_height);

            bool overlaps = !(rect_max.x < row_min.x || rect_min.x > row_max.x ||
              rect_max.y < row_min.y || rect_min.y > row_max.y);

            if ( overlaps ) {
              directory_cache[original_index].is_selected = true;
            }
            else if ( !selection_state.ctrl_held ) {
              directory_cache[original_index].is_selected = false;
            }
          }

          handle_right_click(str);

          ImTextureID texture_id;
          if ( fan::graphics::is_image_valid(file_info.preview_image) ) {
            texture_id = (ImTextureID)fan::graphics::image_get_handle(file_info.preview_image);
          }
          else {
            texture_id = (ImTextureID)fan::graphics::image_get_handle(
              file_info.is_directory ? icon_directory : icon_file
            );
          }
          ImGui::GetWindowDrawList()->AddImage(texture_id, cursor_pos, cursor_pos + image_size);

          handle_item_interaction(file_info, original_index);
        }
      }

      ImGui::EndTable();
    }
  }

  void content_browser_t::handle_item_interaction(const file_info_t& file_info, size_t original_index) {
    if ( file_info.is_directory == false ) {
      if ( ImGui::BeginDragDropSource() ) {
        bool showing_search_results = !search_state.found_files.empty() && !search_buffer.empty();

        if ( showing_search_results ) {
          if ( !search_state.found_files[original_index].is_selected ) {
            for ( auto& f : search_state.found_files ) f.is_selected = false;
            search_state.found_files[original_index].is_selected = true;
          }
        }
        else {
          if ( !directory_cache[original_index].is_selected ) {
            for ( auto& f : directory_cache ) f.is_selected = false;
            directory_cache[original_index].is_selected = true;
          }
        }

        std::vector<std::wstring> selected_paths;

        if ( showing_search_results ) {
          for ( const auto& f : search_state.found_files ) {
            if ( f.is_selected && !f.is_directory ) {
              selected_paths.push_back(f.item_path);
            }
          }
        }
        else {
          for ( const auto& f : directory_cache ) {
            if ( f.is_selected && !f.is_directory ) {
              selected_paths.push_back(f.item_path);
            }
          }
        }

        if ( selected_paths.empty() ) {
          selected_paths.push_back(file_info.item_path);
        }

        std::wstring combined_paths;
        for ( size_t i = 0; i < selected_paths.size(); ++i ) {
          if ( i > 0 ) combined_paths += L";";
          combined_paths += selected_paths[i];
        }
        ImGui::SetDragDropPayload("CONTENT_BROWSER_ITEMS", combined_paths.data(), (combined_paths.size() + 1) * sizeof(wchar_t));

        if ( selected_paths.size() > 1 ) {
          ImGui::Text("%zu files selected", selected_paths.size());
        }
        else {
          ImGui::Text("%s", file_info.filename.c_str());
        }

        ImGui::EndDragDropSource();
      }
    }

    if ( ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) ) {
      if ( file_info.is_directory ) {
        current_directory = std::filesystem::path(asset_path) / file_info.item_path;
        update_directory_cache();
      }
    }
  }

  // [](const std::filesystem::path& path) {}
  void content_browser_t::receive_drag_drop_target(auto receive_func) {
    ImGui::Dummy(ImGui::GetContentRegionAvail());

    if ( ImGui::BeginDragDropTarget() ) {
      if ( const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("CONTENT_BROWSER_ITEMS") ) {
        const wchar_t* paths_data = (const wchar_t*)payload->Data;
        std::wstring combined_paths(paths_data);

        std::vector<std::filesystem::path> file_paths;
        std::wstring current_path;

        for ( wchar_t c : combined_paths ) {
          if ( c == L';' ) {
            if ( !current_path.empty() ) {
              file_paths.push_back(std::filesystem::path(asset_path) / current_path);
              current_path.clear();
            }
          }
          else {
            current_path += c;
          }
        }

        if ( !current_path.empty() ) {
          file_paths.push_back(std::filesystem::path(asset_path) / current_path);
        }

        for ( const auto& path : file_paths ) {
          receive_func(path);
        }
      }
      else if ( const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("CONTENT_BROWSER_ITEMS") ) {
        const wchar_t* path = (const wchar_t*)payload->Data;
        receive_func(std::filesystem::path(asset_path) / path);
      }
      ImGui::EndDragDropTarget();
    }
  }

  ImFont* get_font_impl(f32_t font_size, bool bold) {
    font_size /= 2;
    int best_index = 0;
    f32_t best_diff = std::abs(font_sizes[0] - font_size);

    for ( std::size_t i = 1; i < std::size(font_sizes); ++i ) {
      f32_t diff = std::abs(font_sizes[i] - font_size);
      if ( diff < best_diff ) {
        best_diff = diff;
        best_index = i;
      }
    }

    return !bold ? fonts[best_index] : fonts_bold[best_index];
  }

  font_t* get_font(f32_t font_size, bool bold) {
    return get_font_impl(font_size, bold);
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

  void fragment_shader_editor(uint16_t shape_type, std::string* fragment, bool* shader_compiled) {
    if ( fragment->empty() ) {
      *fragment = fan::graphics::shader_get_data(shape_type).sfragment;
    }
    if ( begin("shader editor", 0, window_flags_no_saved_settings) ) {
      if ( !*shader_compiled ) {
        gui::text("Failed to compile shader", fan::colors::red);
      }

      if ( gui::input_text_multiline("##Shader Code", fragment, gui::get_content_region_avail(), gui::input_text_flags_allow_tab_input) ) {
        *shader_compiled = fan::graphics::shader_update_fragment(shape_type, *fragment);
      }
      end();
    }
  }

  // called inside window begin end
  void animated_popup_window(const std::string& popup_id, const fan::vec2& popup_size, const fan::vec2& start_pos, const fan::vec2& target_pos, bool trigger_popup, auto content_cb, const f32_t anim_duration, const f32_t hide_delay) {
    ImGuiStorage* storage = ImGui::GetStateStorage();
    ImGuiID anim_time_id = ImGui::GetID((popup_id + "_anim_time").c_str());
    ImGuiID hide_timer_id = ImGui::GetID((popup_id + "_hide_timer").c_str());
    ImGuiID hovering_popup_id = ImGui::GetID((popup_id + "_hovering").c_str());
    ImGuiID popup_visible_id = ImGui::GetID((popup_id + "_visible").c_str());

    f32_t delta_time = ImGui::GetIO().DeltaTime;
    f32_t popup_anim_time = storage->GetFloat(anim_time_id, 0.0f);
    f32_t hide_timer = storage->GetFloat(hide_timer_id, 0.0f);
    bool hovering_popup = storage->GetBool(hovering_popup_id, false);
    bool popup_visible = storage->GetBool(popup_visible_id, false);

    // Check if mouse is in parent window area
    //bool mouse_in_parent = (current_mouse_pos.x >= parent_min.x &&
    //  current_mouse_pos.x <= parent_max.x &&
    //  current_mouse_pos.y >= parent_min.y &&
    //  current_mouse_pos.y <= parent_max.y);

    if ( trigger_popup || hovering_popup ) {
      popup_visible = true;
      hide_timer = 0.0f;
    }
    else {
      hide_timer += delta_time;
    }

    if ( popup_visible ) {
      popup_anim_time = hide_timer < hide_delay ? std::min(popup_anim_time + delta_time, anim_duration)
        : std::max(popup_anim_time - delta_time, 0.0f);

      if ( popup_anim_time == 0.0f ) {
        popup_visible = false;
      }

      if ( popup_anim_time > 0.0f ) {
        f32_t t = popup_anim_time / anim_duration;
        t = t * t * (3.0f - 2.0f * t); // smoothstep

        // Simple interpolation between start and target positions
        fan::vec2 popup_pos = start_pos + (target_pos - start_pos) * t;

        ImGui::SetNextWindowPos(popup_pos);
        ImGui::SetNextWindowSize(popup_size);

        if ( ImGui::Begin(popup_id.c_str(), nullptr,
          ImGuiWindowFlags_NoTitleBar |
          ImGuiWindowFlags_NoResize |
          ImGuiWindowFlags_NoMove |
          ImGuiWindowFlags_NoSavedSettings |
          ImGuiWindowFlags_NoCollapse |
          ImGuiWindowFlags_NoScrollbar |
          ImGuiWindowFlags_NoFocusOnAppearing) ) {

          hovering_popup = ImGui::IsWindowHovered(
            ImGuiHoveredFlags_ChildWindows |
            ImGuiHoveredFlags_NoPopupHierarchy |
            ImGuiHoveredFlags_AllowWhenBlockedByPopup |
            ImGuiHoveredFlags_AllowWhenBlockedByActiveItem
          );

          content_cb();
        }
        ImGui::End();
      }
    }

    storage->SetFloat(anim_time_id, popup_anim_time);
    storage->SetFloat(hide_timer_id, hide_timer);
    storage->SetBool(hovering_popup_id, hovering_popup);
    storage->SetBool(popup_visible_id, popup_visible);
  }

  void set_text_fade_time(f32_t seconds) {
    fan::graphics::g_render_context_handle.text_logger->set_text_fade_time(seconds);
  }

  void clear_static_text() {
    fan::graphics::g_render_context_handle.text_logger->clear_static_text();
  }

  bool sprite_animations_t::render_list_box(fan::graphics::animation_nr_t& shape_animation_id) {
    bool list_item_changed = false;

    gui::begin_child("animations_tool_bar", 0, 1);
    gui::set_cursor_pos_y(animation_names_padding);
    gui::indent(animation_names_padding);

    if ( gui::button("+") ) {
      fan::graphics::sprite_sheet_animation_t animation;
      animation.name = std::to_string((uint32_t)fan::graphics::all_animations_counter); // think this over
      shape_animation_id = fan::graphics::add_sprite_sheet_shape_animation(shape_animation_id, animation);
    }
    if ( !shape_animation_id ) {
      gui::unindent(animation_names_padding);
      gui::end_child();
      return false;
    }
    gui::push_item_width(gui::get_window_size().x * 0.8f);
    for ( auto [i, animation_nr] : fan::enumerate(fan::graphics::get_sprite_sheet_shape_animation(shape_animation_id)) ) {
      auto& animation = fan::graphics::get_sprite_sheet_animation(animation_nr);
      if ( animation.name == animation_list_name_to_edit ) {
        std::snprintf(animation_list_name_edit_buffer.data(), animation_list_name_edit_buffer.size() + 1, "%s", animation.name.c_str());
        gui::push_id(i);
        if ( set_focus ) {
          gui::set_keyboard_focus_here();
          set_focus = false;
        }

        if ( gui::input_text("##edit", &animation_list_name_edit_buffer, gui::input_text_flags_enter_returns_true) ) {
          if ( animation_list_name_edit_buffer != animation.name ) {
            fan::graphics::rename_sprite_sheet_shape_animation(shape_animation_id, animation.name, animation_list_name_edit_buffer);
            animation.name = animation_list_name_edit_buffer;
            animation_list_name_to_edit.clear();
            gui::pop_id();
            break;
          }
          else {
            animation_list_name_to_edit.clear();
            gui::pop_id();
            break;
          }
        }
        gui::pop_id();
      }
      else {
        gui::push_id(i);
        if ( gui::selectable(animation.name, current_animation_nr && current_animation_nr == animation_nr, gui::selectable_flags_allow_double_click, fan::vec2(gui::get_content_region_avail().x * 0.8f, 0)) ) {
          if ( gui::is_mouse_double_clicked() ) {
            animation_list_name_to_edit = animation.name;
            set_focus = true;
          }
          current_animation_shape_nr = shape_animation_id;
          current_animation_nr = animation_nr;
          list_item_changed = true;
        }
        gui::pop_id();
      }
    }
    gui::pop_item_width();
    gui::unindent(animation_names_padding);
    gui::end_child();
    return list_item_changed;
  }

  bool sprite_animations_t::render_selectable_frames(fan::graphics::sprite_sheet_animation_t& current_animation) {
    bool changed = false;
    if ( fan::window::is_mouse_released() ) {
      previous_hold_selected.clear();
    }
    int grid_index = 0;
    bool first_button = true;

    for ( int i = 0; i < current_animation.images.size(); ++i ) {
      auto current_image = current_animation.images[i];
      int hframes = current_image.hframes;
      int vframes = current_image.vframes;
      for ( int y = 0; y < vframes; ++y ) {
        for ( int x = 0; x < hframes; ++x ) {
          fan::vec2 tc_size = fan::vec2(1.0 / hframes, 1.0 / vframes);
          fan::vec2 uv_src = fan::vec2(
            fmod(tc_size.x * x, 1.0),
            y / tc_size.y
          );
          fan::vec2 uv_dst = uv_src + fan::vec2(1.0 / hframes, 1.0 / vframes);
          gui::push_id(grid_index);

          f32_t button_width = 128 + gui::get_style().ItemSpacing.x;
          f32_t window_width = gui::get_content_region_avail().x;
          f32_t current_line_width = gui::get_cursor_pos().x - gui::get_window_pos().x;

          if ( current_line_width + button_width > window_width && !first_button ) {
            gui::new_line();
          }

          fan::vec2 cursor_screen_pos = gui::get_cursor_screen_pos() + fan::vec2(7.f, 0);
          gui::image_button("", current_image.image, 128, uv_src, uv_dst);
          auto& sf = current_animation.selected_frames;
          auto it_found = std::find_if(sf.begin(), sf.end(), [a = grid_index](int b) {
            return a == b;
          });
          bool is_found = it_found != sf.end();
          auto previous_hold_it = std::find(previous_hold_selected.begin(), previous_hold_selected.end(), grid_index);
          bool was_added_by_hold_before = previous_hold_it != previous_hold_selected.end();
          if ( gui::is_item_held() && !was_added_by_hold_before ) {
            if ( is_found == false ) {
              sf.push_back(grid_index);
              previous_hold_selected.push_back(grid_index);
              changed = true;
            }
            else {
              changed = true;
              it_found = sf.erase(it_found);
              is_found = false;
              previous_hold_selected.push_back(grid_index);
            }
          }
          if ( is_found ) {
            gui::text_outlined_at(std::to_string(std::distance(sf.begin(), it_found)), cursor_screen_pos);
            gui::text_at(std::to_string(std::distance(sf.begin(), it_found)), cursor_screen_pos);
          }
          gui::pop_id();

          if ( !(y == vframes - 1 && x == hframes - 1 && i == current_animation.images.size() - 1) ) {
            gui::same_line();
          }

          first_button = false;
          ++grid_index;
        }
      }
    }
    return changed;
  }

  bool sprite_animations_t::render(const std::string& drag_drop_id, fan::graphics::animation_nr_t& shape_animation_id) {
    gui::push_style_var(gui::style_var_item_spacing, fan::vec2(12.f, 12.f));
    gui::columns(2, "animation_columns", false);
    gui::set_column_width(0, gui::get_window_size().x * 0.2f);

    bool list_changed = render_list_box(shape_animation_id);

    gui::next_column();

    gui::begin_child("animation_window_right", 0, 1, gui::window_flags_horizontal_scrollbar);

    // just drop image from directory

    gui::push_item_width(72);
    gui::indent(animation_names_padding);
    gui::set_cursor_pos_y(animation_names_padding);
    toggle_play_animation = false;
    if ( gui::image_button("play_button", fan::graphics::icons.play, 32) ) {
      play_animation = true;
      toggle_play_animation = true;
    }
    gui::same_line();
    if ( gui::image_button("pause_button", fan::graphics::icons.pause, 32) ) {
      play_animation = false;
      toggle_play_animation = true;
    }
    decltype(fan::graphics::all_animations)::iterator current_animation;
    if ( !current_animation_nr ) {
      goto g_end_frame;
    }
    current_animation = fan::graphics::all_animations.find(current_animation_nr);
    if ( current_animation == fan::graphics::all_animations.end() ) {
    g_end_frame:
      gui::columns(1);
      gui::end_child();
      gui::pop_style_var();
      return list_changed;
    }

    gui::same_line(0, 20.f);

    gui::slider_flags_t slider_flags = slider_flags_always_clamp | gui::slider_flags_no_speed_tweaks;
    list_changed |= gui::drag("fps", &current_animation->second.fps, 1, 0, 244, slider_flags);
    if ( gui::button("add sprite sheet") ) {
      adding_sprite_sheet = true;
    }
    if ( adding_sprite_sheet && gui::begin("add_animations_sprite_sheet") ) {
      gui::text_box("Drop sprite sheet here", fan::vec2(256, 64));
      gui::receive_drag_drop_target(drag_drop_id, [this, shape_animation_id](const std::string& file_path) {
        if ( fan::image::valid(file_path) ) {
          sprite_sheet_drag_drop_name = file_path;
        }
        else {
          fan::print("Warning: drop target not valid (requires image file)");
        }
      });

      gui::drag("Horizontal frames", &hframes, 1, 0, 1024, slider_flags);
      gui::drag("Vertical frames", &vframes, 1, 0, 1024, slider_flags);

      if ( !sprite_sheet_drag_drop_name.empty() ) {
        gui::separator();
        gui::text("Preview:");

        auto preview_image = fan::graphics::image_load(sprite_sheet_drag_drop_name);

        if ( fan::graphics::is_image_valid(preview_image) ) {
          f32_t content_width = hframes * (64 + gui::get_style().ItemSpacing.x);
          f32_t content_height = vframes * (64 + gui::get_style().ItemSpacing.y);

          if ( gui::begin_child("sprite_preview", fan::vec2(0, std::min(content_height + 20, 300.0f)), true, ImGuiWindowFlags_HorizontalScrollbar) ) {
            for ( int y = 0; y < vframes; ++y ) {
              for ( int x = 0; x < hframes; ++x ) {
                fan::vec2 tc_size = fan::vec2(1.0 / hframes, 1.0 / vframes);
                fan::vec2 uv_src = fan::vec2(
                  tc_size.x * x,
                  tc_size.y * y
                );
                fan::vec2 uv_dst = uv_src + tc_size;

                gui::push_id(y * hframes + x);
                gui::image_button("", preview_image, 64, uv_src, uv_dst);
                gui::pop_id();

                if ( x != hframes - 1 ) {
                  gui::same_line();
                }
              }
            }
          }
          gui::end_child();
        }
      }

      if ( gui::button("Add") ) {
        if ( auto it = fan::graphics::all_animations.find(current_animation_nr); it != fan::graphics::all_animations.end() ) {
          auto& anim = it->second;
          fan::graphics::sprite_sheet_animation_t::image_t new_image;
          new_image.image = fan::graphics::image_load(sprite_sheet_drag_drop_name);
          new_image.hframes = hframes;
          new_image.vframes = vframes;
          anim.images.push_back(new_image);
          sprite_sheet_drag_drop_name.clear();
        }
        adding_sprite_sheet = false;
        list_changed |= 1;
      }

      gui::end();
    }

    gui::separator();

    fan::vec2 cursor_pos = gui::get_cursor_pos();
    if ( drag_drop_id.size() ) {
      //fan::vec2 avail = gui::get_content_region_avail();
      fan::vec2 child_size = gui::get_window_size();
      ImGui::Dummy(child_size);
      gui::receive_drag_drop_target(drag_drop_id, [this, shape_animation_id](const std::string& file_paths) {
        for ( const std::string& file_path : fan::split(file_paths, ";") ) {
          if ( fan::image::valid(file_path) ) {
            if ( auto it = fan::graphics::all_animations.find(current_animation_nr); it != fan::graphics::all_animations.end() ) {
              auto& anim = it->second;
              //// unload previous image
              //if (fan::graphics::is_image_valid(anim.sprite_sheet)) {
              //  fan::graphics::image_unload(anim.sprite_sheet);
              //}
              fan::graphics::sprite_sheet_animation_t::image_t new_image;
              new_image.image = fan::graphics::image_load(file_path);
              anim.images.push_back(new_image);
            }
          }
          else {
            fan::print("Warning: drop target not valid (requires image file)");
          }
        }
      });
    }
    gui::set_cursor_pos(cursor_pos);

    //render_play_animation();

    list_changed |= render_selectable_frames(current_animation->second);

    gui::unindent(animation_names_padding);
    gui::end_child();
    gui::columns(1);
    gui::pop_style_var();
    return list_changed;
  }

  fan::graphics::shapes::particles_t::ri_t& particle_editor_t::get_ri() {
    return *(fan::graphics::shapes::particles_t::ri_t*)particle_shape.GetData(fan::graphics::g_shapes->shaper);
  }

  void particle_editor_t::handle_file_operations() {
    if ( open_file_dialog.is_finished() ) {
      if ( filename.size() != 0 ) {
        std::string data;
        fan::io::file::read(filename, &data);
        particle_shape = fan::json::parse(data);
      }
      open_file_dialog.finished = false;
    }

    if ( save_file_dialog.is_finished() ) {
      if ( filename.size() != 0 ) {
        fan::json json_data = particle_shape;
        fan::io::file::write(filename, json_data.dump(2), std::ios_base::binary);
      }
      save_file_dialog.finished = false;
    }
  }

  void particle_editor_t::render_menu() {
    if ( begin_main_menu_bar() ) {
      if ( begin_menu("File") ) {
        if ( menu_item("Open..", "Ctrl+O") ) {
          open_file_dialog.load("json;fmm", &filename);
        }
        if ( menu_item("Save as", "Ctrl+Shift+S") ) {
          save_file_dialog.save("json;fmm", &filename);
        }
        end_menu();
      }
      end_main_menu_bar();
    }
  }

  void particle_editor_t::render_settings() {
    begin("particle settings");
    color_edit4("background color", &bg_color);
    shape_properties(particle_shape);
    end();
  }

  void particle_editor_t::render() {
    render_menu();
    handle_file_operations();
    render_settings();
  }

  dialogue_box_t::render_type_t::~render_type_t() {}

  dialogue_box_t::text_delayed_t::~text_delayed_t() {
    dialogue_line_finished = true;
    character_advance_task = {};
  }

  void dialogue_box_t::text_delayed_t::render(dialogue_box_t* This, dialogue_box_t::drawable_nr_t nr, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing) {

    // initialize advance task but dont restart it after dialog finished
    if ( dialogue_line_finished == false && !character_advance_task.owner ) {
      character_advance_task = [This, nr]() -> fan::event::task_t {
        text_delayed_t* text_delayed = dynamic_cast<text_delayed_t*>(This->drawables[nr]);
        if ( text_delayed == nullptr ) {
          co_return;
        }

        // advance text rendering
        while ( text_delayed->render_pos < text_delayed->text.size() && !text_delayed->dialogue_line_finished && text_delayed->character_per_s ) {
          ++text_delayed->render_pos;
          co_await fan::co_sleep(1000 / text_delayed->character_per_s);
        }
      }();
    }

    //ImGui::BeginChild((fan::random::string(10) + "child").c_str(), fan::vec2(wrap_width, 0), 0, ImGuiWindowFlags_NoNavInputs | ImGuiWindowFlags_NoTitleBar |
    //  ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoBackground);
    //if (This->wait_user == false) {
    //  ImGui::SetScrollY(ImGui::GetScrollMaxY());
    //}
    text_partial_render(text, render_pos, wrap_width, line_spacing);
    //            ImGui::EndChild();

    if ( render_pos == text.size() ) {
      dialogue_line_finished = true;
    }

    if ( dialogue_line_finished && blink_timer.finished() ) {
      if ( render_cursor ) {
        text.push_back('|');
        render_pos = text.size();
      }
      else {
        if ( text.back() == '|' ) {
          text.pop_back();
          render_pos = text.size();
        }
      }
      render_cursor = !render_cursor;
      blink_timer.restart();
    }
  }

  void dialogue_box_t::text_t::render(dialogue_box_t* This, dialogue_box_t::drawable_nr_t nr, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing) {
    gui::text(text);
  }

  void dialogue_box_t::button_t::render(dialogue_box_t* This, dialogue_box_t::drawable_nr_t nr, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing) {
    if ( This->wait_user ) {
      fan::vec2 button_size = 0;
      fan::vec2 text_size = gui::calc_text_size(text);
      f32_t padding_x = ImGui::GetStyle().FramePadding.x;
      f32_t padding_y = ImGui::GetStyle().FramePadding.y;
      button_size = fan::vec2(text_size.x + padding_x * 2.0f, text_size.y + padding_y * 2.0f);

      ImVec2 cursor = (position * gui::get_window_size()) - (size == 0 ? button_size / 2 : size / 2);
      cursor.x += ImGui::GetStyle().WindowPadding.x;
      cursor.y += ImGui::GetStyle().WindowPadding.y;
      ImGui::SetCursorPos(cursor);

      if (gui::button(text, size == 0 ? button_size : size) ) {
        This->button_choice = nr;
        auto it = This->drawables.GetNodeFirst();
        while ( it != This->drawables.dst ) {
          This->drawables.StartSafeNext(it);
          if ( auto* button = dynamic_cast<button_t*>(This->drawables[it]) ) {
            delete This->drawables[it];
            This->drawables.unlrec(it);
          }
          it = This->drawables.EndSafeNext();
        }
        This->wait_user = false;
      }
    }
  }

  void dialogue_box_t::separator_t::render(dialogue_box_t* This, dialogue_box_t::drawable_nr_t nr, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing) {
    ImGui::Separator();
  }

  dialogue_box_t::dialogue_box_t() {
    fan::window::add_input_action(fan::mouse_left, "skip or continue dialog");
  }

  // 0-1
  void dialogue_box_t::set_cursor_position(const fan::vec2& pos) {
    this->cursor_position = pos;
  }

  void dialogue_box_t::set_indent(f32_t indent) {
    this->indent = indent;
  }

  fan::event::task_value_resume_t<dialogue_box_t::drawable_nr_t> dialogue_box_t::text_delayed(const std::string& character_name, const std::string& text) {
    return text_delayed(character_name, text, 20); // 20 characters per second
  }

  fan::event::task_value_resume_t<dialogue_box_t::drawable_nr_t> dialogue_box_t::text_delayed(const std::string& character_name, const std::string& text, int characters_per_second) {
    text_delayed_t td;
    td.character_per_s = characters_per_second;
    td.text = text;
    td.render_pos = 0;
    td.dialogue_line_finished = false;

    auto it = drawables.NewNodeLast();
    drawables[it] = new text_delayed_t(std::move(td));

    co_return it;
  }

  fan::event::task_value_resume_t<dialogue_box_t::drawable_nr_t> dialogue_box_t::text(const std::string& text) {
    text_t text_drawable;
    text_drawable.text = text;

    auto it = drawables.NewNodeLast();
    drawables[it] = new text_t(text_drawable);

    co_return it;
  }

  fan::event::task_value_resume_t<dialogue_box_t::drawable_nr_t> dialogue_box_t::button(const std::string& text, const fan::vec2& position, const fan::vec2& size) {
    button_choice.sic();
    button_t button;
    button.position = position;
    button.size = size;
    button.text = text;

    auto it = drawables.NewNodeLast();
    drawables[it] = new button_t(button);
    co_return it;
  }

  // default width 80% of the window
  fan::event::task_value_resume_t<dialogue_box_t::drawable_nr_t> dialogue_box_t::separator(f32_t width) {
    auto it = drawables.NewNodeLast();
    drawables[it] = new separator_t;

    co_return it;
  }

  int dialogue_box_t::get_button_choice() {
    int btn_choice = -1;

    uint64_t button_index = 0;
    auto it = drawables.GetNodeFirst();
    while ( it != drawables.dst ) {
      drawables.StartSafeNext(it);
      if ( auto* button = dynamic_cast<button_t*>(drawables[it]) ) {
        if ( button_choice == it ) {
          break;
        }
        ++button_index;
      }
      it = drawables.EndSafeNext();
    }
    return btn_choice;
  }

  fan::event::task_t dialogue_box_t::wait_user_input() {
    wait_user = true;
    while ( wait_user ) {
      co_await fan::co_sleep(10);
    }
  }

  void dialogue_box_t::render(const std::string& window_name, ImFont* font, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing, const auto& inside_window_cb) {
    ImGui::PushFont(font);

    fan::vec2 root_window_size = ImGui::GetWindowSize();
    fan::vec2 next_window_pos;
    next_window_pos.x = (root_window_size.x - window_size.x) / 2.0f;
    next_window_pos.y = (root_window_size.y - window_size.y) / 1.1;
    ImGui::SetNextWindowPos(next_window_pos);

    ImGui::SetNextWindowSize(window_size);
    ImGui::Begin(window_name.c_str(), 0,
      ImGuiWindowFlags_NoNavInputs | ImGuiWindowFlags_NoTitleBar |
      ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar
    );

    f32_t current_font_size = ImGui::GetFont()->FontSize;
    f32_t scale_factor = font_size / current_font_size;
    ImGui::SetWindowFontScale(scale_factor);

    //    ImGui::BeginChild((fan::random::string(10) + "child").c_str(), fan::vec2(wrap_width, 0), 0, ImGuiWindowFlags_NoNavInputs | ImGuiWindowFlags_NoTitleBar |
    //        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoBackground);

    inside_window_cb();

    render_content_cb(cursor_position == -1 ? fan::vec2(ImGui::GetStyle().WindowPadding) : cursor_position, indent);
    // render objects here

    auto it = drawables.GetNodeFirst();
    while ( it != drawables.dst ) {
      drawables.StartSafeNext(it);
      // co_await or task vector
      drawables[it]->render(this, it, window_size, wrap_width, line_spacing);
      it = drawables.EndSafeNext();
    }
    //     ImGui::EndChild();
    ImGui::SetWindowFontScale(1.0f);


    bool dialogue_line_finished = fan::window::is_input_action_active("skip or continue dialog") && ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows | ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);

    if ( dialogue_line_finished ) {
      wait_user = false;
      clear();
    }

    ImGui::End();
    ImGui::PopFont();
  }

  void dialogue_box_t::render(const std::string& window_name, ImFont* font, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing) {
    render(window_name, font, window_size, wrap_width, line_spacing, [] {});
  }

  void dialogue_box_t::clear() {
    auto it = drawables.GetNodeFirst();
    while ( it != drawables.dst ) {
      drawables.StartSafeNext(it);
      delete drawables[it];
      drawables.unlrec(it);
      it = drawables.EndSafeNext();
    }
  }

  void dialogue_box_t::default_render_content(const fan::vec2& cursor_pos, f32_t indent) {
    ImGui::SetCursorPos(cursor_pos);
    gui::indent(indent);
  }

  void text_partial_render(const std::string& text, size_t render_pos, f32_t wrap_width, f32_t line_spacing) {
    static auto find_next_word = [](const std::string& str, std::size_t offset) -> std::size_t {
      std::size_t found = str.find(' ', offset);
      if ( found == std::string::npos ) {
        found = str.size();
      }
      if ( found != std::string::npos ) {
      }
      return found;
    };
    static auto find_previous_word = [](const std::string& str, std::size_t offset) -> std::size_t {
      std::size_t found = str.rfind(' ', offset);
      if ( found == std::string::npos ) {
        found = str.size();
      }
      if ( found != std::string::npos ) {
      }
      return found;
    };

    std::vector<std::string> lines;
    std::size_t previous_word = 0;
    std::size_t previous_push = 0;
    bool found = false;
    for ( std::size_t i = 0; i < text.size(); ++i ) {
      std::size_t word_index = text.find(' ', i);
      if ( word_index == std::string::npos ) {
        word_index = text.size();
      }

      std::string str = text.substr(previous_push, word_index - previous_push);
      f32_t width = ImGui::CalcTextSize(str.c_str()).x;

      if ( width >= wrap_width ) {
        if ( previous_push == previous_word ) {
          lines.push_back(text.substr(previous_push, i - previous_push));
          previous_push = i;
        }
        else {
          lines.push_back(text.substr(previous_push, previous_word - previous_push));
          previous_push = previous_word + 1;
          i = previous_word;
        }
      }
      previous_word = word_index;
      i = word_index;
    }

    // Add remaining text as last line
    if ( previous_push < text.size() ) {
      lines.push_back(text.substr(previous_push));
    }

    std::size_t empty_lines = 0;
    std::size_t character_offset = 0;
    ImVec2 pos = ImGui::GetCursorScreenPos();
    for ( const auto& line : lines ) {
      if ( line.empty() ) {
        ++empty_lines;
        continue;
      }
      std::size_t empty = 0;
      if ( empty >= line.size() ) {
        break;
      }
      while ( line[empty] == ' ' ) {
        if ( empty >= line.size() ) {
          break;
        }
        ++empty;
      }
      if ( character_offset >= render_pos ) {
        break;
      }
      std::string render_text = line.substr(empty).c_str();
      ImGui::SetCursorScreenPos(pos);
      if ( character_offset + render_text.size() >= render_pos ) {
        ImGui::TextUnformatted(render_text.substr(0, render_pos - character_offset).c_str());
        break;
      }
      else {
        ImGui::TextUnformatted(render_text.c_str());
        if ( render_text.back() != ' ' ) {
          character_offset += 1;
        }
        character_offset += render_text.size();
        pos.y += ImGui::GetTextLineHeightWithSpacing() + line_spacing;
      }
    }
    if ( empty_lines ) {
      ImGui::TextColored(fan::colors::red, "warning empty lines:%zu", empty_lines);
    }
  }
}