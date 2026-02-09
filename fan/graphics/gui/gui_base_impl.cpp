module;

#if defined(FAN_GUI)
  #include <fan/imgui/imgui.h>
  #include <fan/imgui/imgui_impl_glfw.h>
  #include <fan/imgui/implot.h>
  #include <fan/imgui/imgui_internal.h>
  #include <fan/imgui/ImGuizmo.h>

  #include <fan/graphics/gui/imgui_themes.h>
  #include <fan/imgui/misc/freetype/imgui_freetype.h>
  #if defined(FAN_OPENGL)
    #include <fan/imgui/imgui_impl_opengl3.h>
  #endif
  #if defined(FAN_VULKAN)
    #include <vulkan/vulkan.h>
  #endif

  #include <string_view>
  #include <functional>
  #include <cstdint>
  #include <cstring>
  #include <limits>
  #include <string>

  #define GLFW_INCLUDE_NONE
  #include <GLFW/glfw3.h>
#endif

module fan.graphics.gui.base;

#if defined(FAN_GUI)

#if defined(FAN_AUDIO)
import fan.audio;
#endif

import fan.types.fstring;
import fan.graphics.gui.types;
import fan.types.vector;
import fan.types.color;
import fan.types.matrix;
import fan.utility;
import fan.math;
import fan.print;
import fan.graphics.common_context;

namespace fan::graphics::gui {
  namespace detail {

    struct input_text_callback_user_data_t {
      std::string* str;
      input_text_callback_t chain_callback;
      void* chain_callback_user_data;
    };

    inline int input_text_callback(ImGuiInputTextCallbackData* data) {
      auto* user_data = static_cast<input_text_callback_user_data_t*>(data->UserData);
      if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
        std::string* str = user_data->str;
        IM_ASSERT(data->Buf == str->c_str());
        str->resize(data->BufTextLen);
        data->Buf = const_cast<char*>(str->c_str());
      }
      else if (user_data->chain_callback) {
        data->UserData = user_data->chain_callback_user_data;
        return user_data->chain_callback(data);
      }
      return 0;
    }

    inline auto& want_io_ignore_list() {
      static std::unordered_map<std::string_view, bool> list;
      return list;
    }

    inline void receive_drag_drop_target_impl(const char* id, std::function<void(std::string)> receive_func) {
      if (ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(id)) {
          const wchar_t* wpath = static_cast<const wchar_t*>(payload->Data);
          std::string utf8 = fan::wstring_to_utf8(wpath);
          receive_func(std::move(utf8));
        }
        ImGui::EndDragDropTarget();
      }
    }

    inline bool image_button_img_impl(const char* str_id,
      fan::graphics::image_t img,
      const fan::vec2& size,
      const fan::vec2& uv0,
      const fan::vec2& uv1,
      int frame_padding,
      const fan::color& bg_col,
      const fan::color& tint_col) {
      texture_id_t tex = static_cast<texture_id_t>(fan::graphics::ctx()->image_get_handle(fan::graphics::ctx(), img));
      ImVec4 bg(bg_col.r, bg_col.g, bg_col.b, bg_col.a);
      ImVec4 tint(tint_col.r, tint_col.g, tint_col.b, tint_col.a);
      return ImGui::ImageButton(str_id, tex, ImVec2(size.x, size.y),
        ImVec2(uv0.x, uv0.y), ImVec2(uv1.x, uv1.y),
        bg, tint);
    }

    inline bool image_text_button_impl(fan::graphics::image_t img,
      const char* text,
      const fan::color& color,
      const fan::vec2& size,
      const fan::vec2& uv0,
      const fan::vec2& uv1,
      int frame_padding,
      const fan::color& bg_col,
      const fan::color& tint_col
    ) {
      texture_id_t tex = (texture_id_t)(uintptr_t)
        fan::graphics::ctx()->image_get_handle(fan::graphics::ctx(), img);

      char id_buf[64];
      std::snprintf(id_buf, sizeof(id_buf), "imgbtn_%p", tex);

      ImVec4 bg(bg_col.r, bg_col.g, bg_col.b, bg_col.a);
      ImVec4 tint(tint_col.r, tint_col.g, tint_col.b, tint_col.a);

      bool pressed = ImGui::ImageButton(
        id_buf,
        tex,
        ImVec2(size.x, size.y),
        ImVec2(uv0.x, uv0.y),
        ImVec2(uv1.x, uv1.y),
        bg,
        tint
      );

      if (text && *text) {
        ImGui::SameLine();
        ImVec4 col(color.r, color.g, color.b, color.a);
        ImGui::TextColored(col, "%s", text);
      }

      return pressed;
    }

    bool toggle_button_impl(const char* str, bool* v) {
      ImGui::Text("%s", str);
      ImGui::SameLine();

      ImVec2 p = ImGui::GetCursorScreenPos();
      ImDrawList* draw_list = ImGui::GetWindowDrawList();

      f32_t height = ImGui::GetFrameHeight();
      f32_t width = height * 1.55f;
      f32_t radius = height * 0.50f;

      char id_buf[256];
      std::snprintf(id_buf, sizeof(id_buf), "##%s", str);
      bool changed = ImGui::InvisibleButton(id_buf, ImVec2(width, height));
      if (changed)
        *v = !*v;
      ImU32 col_bg;
      if (ImGui::IsItemHovered())
        col_bg = *v ? IM_COL32(145 + 20, 211, 68 + 20, 255) : IM_COL32(218 - 20, 218 - 20, 218 - 20, 255);
      else
        col_bg = *v ? IM_COL32(145, 211, 68, 255) : IM_COL32(218, 218, 218, 255);

      draw_list->AddRectFilled(p, ImVec2(p.x + width, p.y + height), col_bg, height * 0.5f);
      draw_list->AddCircleFilled(ImVec2(*v ? (p.x + width - radius) : (p.x + radius), p.y + radius), radius - 1.5f, IM_COL32(255, 255, 255, 255));

      return changed;
    }

    bool audio_button_impl(const char* label, fan::audio::piece_t piece_hover, fan::audio::piece_t piece_click, const fan::vec2& size) {
      ImGui::PushID(label);
      ImGuiStorage* storage = ImGui::GetStateStorage();
      ImGuiID id = ImGui::GetID("audio_button_prev_hovered");
      bool previously_hovered = storage->GetBool(id);

      bool pressed = ImGui::Button(label, ImVec2(size.x, size.y));
      bool currently_hovered = ImGui::IsItemHovered();

      if (currently_hovered && !previously_hovered) {
        fan::audio::piece_t& piece = fan::audio::is_piece_valid(piece_hover) ? piece_hover : fan::audio::piece_hover;
        fan::audio::play(piece);
      }
      if (pressed) {
        fan::audio::piece_t& piece = fan::audio::is_piece_valid(piece_click) ? piece_click : fan::audio::piece_click;
        fan::audio::play(piece);
      }
      storage->SetBool(id, currently_hovered);

      ImGui::PopID();
      return pressed;
    }

    void text_outlined_at_impl(const char* text, const char* text_end, const fan::vec2& screen_pos, const fan::color& color, const fan::color& outline_color) {
      constexpr f32_t outline_thickness = 1.5f;
      constexpr fan::vec2 outline_offsets[] = {
        {0, -outline_thickness},
        {-outline_thickness, 0},
        {outline_thickness, 0},
        {0, outline_thickness}
      };

      ImDrawList* dl = ImGui::GetWindowDrawList();
      ImU32 outline_u32 = ImGui::ColorConvertFloat4ToU32(ImVec4(outline_color.r, outline_color.g, outline_color.b, outline_color.a));
      ImU32 color_u32 = ImGui::ColorConvertFloat4ToU32(ImVec4(color.r, color.g, color.b, color.a));

      for (const auto& off : outline_offsets) {
        dl->AddText(ImVec2(screen_pos.x + off.x, screen_pos.y + off.y), outline_u32, text, text_end);
      }
      dl->AddText(ImVec2(screen_pos.x, screen_pos.y), color_u32, text, text_end);
    }

    void tooltip_impl(const char* text, const char* text_end, bool show) {
      if (!show) return;
      gui::begin_tooltip();
      ImGui::TextUnformatted(text, text_end);
      gui::end_tooltip();
    }

    font_t* get_font_from_array(font_t* (&fonts)[std::size(fan::graphics::gui::font_sizes)], f32_t font_size) {
      font_size /= 2;
      std::size_t best_index = 0;
      f32_t best_diff = std::numeric_limits<f32_t>::max();
      for (std::size_t i = 0; i < std::size(font_sizes); ++i) {
        f32_t diff = fan::math::abs(font_sizes[i] - font_size);
        if (diff < best_diff) {
          best_diff = diff;
          best_index = i;
        }
      }
      return fonts[best_index];
    }

  } // namespace detail

  void topmost_window_data_t::register_window(std::string_view name) {
    if (std::find(windows.begin(), windows.end(), name) == windows.end()) {
      windows.push_back(std::string(name));
    }
  }

  void topmost_window_data_t::unregister_window(std::string_view name) {
    auto it = std::find(windows.begin(), windows.end(), name);
    if (it != windows.end()) {
      windows.erase(it);
    }
  }

  topmost_window_data_t& topmost_data() {
    static topmost_window_data_t d;
    return d;
  }
  void enforce_topmost() {
    ImGuiContext* g = ImGui::GetCurrentContext();
    if (!g) return;

    auto& data = topmost_data();

    for (const auto& name : data.windows) {
      ImGuiWindow* w = ImGui::FindWindowByName(name.c_str());
      if (!w || !w->WasActive) continue;

      ImGui::BringWindowToDisplayFront(w);
      ImGui::BringWindowToFocusFront(w);
    }
  }

  bool begin(label_t window_name, bool* p_open, window_flags_t flags) {
    bool is_topmost = flags & window_flags_topmost;
    flags &= ~window_flags_topmost;

    if (is_topmost) {
      topmost_data().register_window(window_name);

      flags |= ImGuiWindowFlags_NoBringToFrontOnFocus;
      flags |= ImGuiWindowFlags_NoFocusOnAppearing;
    }
    else {
      topmost_data().unregister_window(window_name);
    }

    if (flags & window_flags_no_title_bar) {
      ImGuiWindowClass window_class;
      window_class.DockNodeFlagsOverrideSet = ImGuiDockNodeFlags_NoTabBar;
      ImGui::SetNextWindowClass(&window_class);
    }
    if (flags & window_flags_override_input) {
      detail::want_io_ignore_list()[window_name] = true;
    }
    bool result = ImGui::Begin(fan::ct_string(window_name), p_open, flags);
    return result;
  }


  void end() {
    ImGui::End();
  }

  bool begin_child(label_t window_name, const fan::vec2& size, child_window_flags_t child_window_flags, window_flags_t window_flags) {
    return ImGui::BeginChild(fan::ct_string(window_name),
      ImVec2(size.x, size.y),
      child_window_flags,
      window_flags);
  }

  void end_child() {
    ImGui::EndChild();
  }

  bool begin_tab_item(label_t label, bool* p_open, window_flags_t window_flags) {
    return ImGui::BeginTabItem(label, p_open, window_flags);
  }

  void end_tab_item() {
    ImGui::EndTabItem();
  }

  bool begin_tab_bar(label_t tab_bar_name, window_flags_t window_flags) {
    return ImGui::BeginTabBar(fan::ct_string(tab_bar_name), window_flags);
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

  bool begin_menu(label_t label, bool enabled) {
    return ImGui::BeginMenu(label, enabled);
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

  void table_setup_column(label_t label, table_column_flags_t flags, f32_t init_width_or_weight, id_t user_id) {
    ImGui::TableSetupColumn(label, flags, init_width_or_weight, user_id);
  }

  void table_headers_row() {
    ImGui::TableHeadersRow();
  }

  bool table_set_column_index(int column_n) {
    return ImGui::TableSetColumnIndex(column_n);
  }

  f32_t table_get_column_offset(int column_n) {
    int current_col = ImGui::TableGetColumnIndex();
    ImGui::TableSetColumnIndex(column_n);
    f32_t x_pos = ImGui::GetCursorScreenPos().x;
    ImGui::TableSetColumnIndex(current_col);
    return x_pos;
  }

  f32_t table_get_cell_width(f32_t init_width) {
    return init_width + get_style().CellPadding.x * 2.f + get_style().FramePadding.x;
  }

  void push_clip_rect(const fan::vec2& min, const fan::vec2& max, bool intersect_with_current_clip_rect) {
    ImGui::PushClipRect(ImVec2(min.x, min.y), ImVec2(max.x, max.y), intersect_with_current_clip_rect);
  }

  void pop_clip_rect() {
    ImGui::PopClipRect();
  }

  bool button_behavior(const gui::rect_t& bb, gui::id_t id, bool* out_hovered, bool* out_held, int flags) {
    return ImGui::ButtonBehavior(bb, id, out_hovered, out_held, flags);
  }

  gui::table_data_t* get_current_table() {
    return ImGui::GetCurrentTable();
  }

  void show_debug_log_window(bool* p_open) {
    ImGui::ShowDebugLogWindow(p_open);
  }

  bool menu_item(label_t label, std::string_view shortcut, bool selected, bool enabled) {
    const char* sc = shortcut.empty() ? nullptr : shortcut.data();
    return ImGui::MenuItem(label, sc, selected, enabled);
  }

  bool begin_combo(label_t label, std::string_view preview_value, int flags) {
    return ImGui::BeginCombo(label, fan::ct_string(preview_value), flags);
  }

  void end_combo() {
    ImGui::EndCombo();
  }

  window_t::window_t(label_t window_name, bool* p_open, window_flags_t window_flags)
    : is_open(begin(window_name, p_open, window_flags)) {}

  window_t::~window_t() {
    end();
  }

  window_t::operator bool() const {
    return is_open;
  }

  child_window_t::child_window_t(label_t window_name, const fan::vec2& size, child_window_flags_t window_flags)
    : is_open(begin_child(window_name, size, window_flags)) {}

  child_window_t::~child_window_t() {
    end_child();
  }

  child_window_t::operator bool() const {
    return is_open;
  }

  table_t::table_t(label_t str_id, int columns, table_flags_t flags, const fan::vec2& outer_size, f32_t inner_width)
    : is_open(begin_table(str_id, columns, flags, outer_size, inner_width)) {}

  table_t::~table_t() {
    end_table();
  }

  table_t::operator bool() const {
    return is_open;
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

  viewport_rect_t get_viewport_rect() {
    viewport_rect_t vr;
    ImGuiViewport* vp = ImGui::GetMainViewport();
    vr.position = fan::vec2(vp->Pos.x, vp->Pos.y);
    vr.size = fan::vec2(vp->Size.x, vp->Size.y);
    return vr;
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
    ImVec2 p = ImGui::GetMousePos();
    return fan::vec2(p.x, p.y);
  }

  bool selectable(label_t label, bool selected, selectable_flag_t flags, const fan::vec2& size) {
    return ImGui::Selectable(label, selected, flags, ImVec2(size.x, size.y));
  }

  bool selectable(label_t label, bool* p_selected, selectable_flag_t flags, const fan::vec2& size) {
    return ImGui::Selectable(label, p_selected, flags, ImVec2(size.x, size.y));
  }

  bool is_mouse_double_clicked(int button) {
    return ImGui::IsMouseDoubleClicked(button);
  }

  fan::vec2 get_content_region_avail() {
    ImVec2 v = ImGui::GetContentRegionAvail();
    return fan::vec2(v.x, v.y);
  }

  fan::vec2 get_content_region_max() {
    ImVec2 v = ImGui::GetContentRegionMax();
    return fan::vec2(v.x, v.y);
  }

  fan::vec2 get_item_rect_min() {
    ImVec2 v = ImGui::GetItemRectMin();
    return fan::vec2(v.x, v.y);
  }

  fan::vec2 get_item_rect_max() {
    ImVec2 v = ImGui::GetItemRectMax();
    return fan::vec2(v.x, v.y);
  }

  void item_size(const fan::vec2& s) {
    ImGui::ItemSize(ImVec2(s.x, s.y));
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
    ImGui::SetCursorScreenPos(pos);
  }

  void push_id(label_t str_id) {
    ImGui::PushID(fan::ct_string(str_id));
  }

  void push_id(int int_id) {
    ImGui::PushID(int_id);
  }

  void push_id(const void* ptr_id) {
    ImGui::PushID(ptr_id);
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

  bool is_item_clicked(int mouse_button) {
    return ImGui::IsItemClicked(mouse_button);
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

  void set_tooltip(std::string_view tooltip) {
    ImGui::SetTooltip("%.*s", (int)tooltip.size(), tooltip.data());
  }

  bool begin_table(label_t str_id, int columns, table_flags_t flags, const fan::vec2& outer_size, f32_t inner_width) {
    return ImGui::BeginTable(fan::ct_string(str_id), columns, flags,
      outer_size, inner_width);
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

  void build_fonts() {
    auto& io = get_io();
    io.Fonts->Build();
  }

  void rebuild_fonts() {
    auto& io = get_io();
    io.Fonts->Clear();
    build_fonts();
  }

  void load_fonts(font_t* (&fonts)[std::size(fan::graphics::gui::font_sizes)], std::string_view name, font_config_t* cfg) {
    ImGuiIO& io = ImGui::GetIO();

    font_config_t internal_cfg;
    internal_cfg.FontBuilderFlags |= ImGuiFreeTypeBuilderFlags_LoadColor;
    if (cfg) {
      internal_cfg = *cfg;
    }

    for (std::size_t i = 0; i < std::size(fonts); ++i) {
      fonts[i] = io.Fonts->AddFontFromFileTTF(
        fan::ct_string(name),
        fan::graphics::gui::font_sizes[i] * 2,
        &internal_cfg
      );

      if (fonts[i] == nullptr) {
        fan::throw_error_impl((std::string("failed to load font:") + std::string(name)).c_str());
      }
    }
  }

  void load_fonts(font_t* (&fonts)[std::size(fan::graphics::gui::font_sizes)], const char* name, font_config_t* cfg) {
    ImGuiIO& io = ImGui::GetIO();

    font_config_t internal_cfg;
    internal_cfg.FontBuilderFlags |= ImGuiFreeTypeBuilderFlags_LoadColor;
    if (cfg) {
      internal_cfg = *cfg;
    }

    for (std::size_t i = 0; i < std::size(fonts); ++i) {
      fonts[i] = io.Fonts->AddFontFromFileTTF(
        name,
        fan::graphics::gui::font_sizes[i] * 2,
        &internal_cfg
      );

      if (fonts[i] == nullptr) {
        fan::throw_error_impl((std::string("failed to load font:") + name).c_str());
      }
    }
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
    ImGui::Unindent(indent_w);
  }

  fan::vec2 calc_text_size(std::string_view text, const char* text_end, bool hide_text_after_double_hash, f32_t wrap_width) {
    ImVec2 s = ImGui::CalcTextSize(text.data(),
      text_end ? text_end : text.data() + text.size(),
      hide_text_after_double_hash,
      wrap_width);
    return fan::vec2(s.x, s.y);
  }

  fan::vec2 get_text_size(std::string_view text, const char* text_end, bool hide_text_after_double_hash, f32_t wrap_width) {
    return calc_text_size(text, text_end, hide_text_after_double_hash, wrap_width);
  }

  fan::vec2 text_size(std::string_view text, const char* text_end, bool hide_text_after_double_hash, f32_t wrap_width) {
    return calc_text_size(text, text_end, hide_text_after_double_hash, wrap_width);
  }

  void set_cursor_pos_x(f32_t pos) {
    ImGui::SetCursorPosX(pos);
  }

  void set_cursor_pos_y(f32_t pos) {
    ImGui::SetCursorPosY(pos);
  }

  void set_cursor_pos(const fan::vec2& pos) {
    ImGui::SetCursorPos(ImVec2(pos.x, pos.y));
  }

  fan::vec2 get_cursor_pos() {
    ImVec2 p = ImGui::GetCursorPos();
    return fan::vec2(p.x, p.y);
  }

  f32_t get_cursor_pos_x() {
    return ImGui::GetCursorPosX();
  }

  f32_t get_cursor_pos_y() {
    return ImGui::GetCursorPosY();
  }

  fan::vec2 get_cursor_screen_pos() {
    ImVec2 p = ImGui::GetCursorScreenPos();
    return fan::vec2(p.x, p.y);
  }

  fan::vec2 get_cursor_start_pos() {
    ImVec2 p = ImGui::GetCursorStartPos();
    return fan::vec2(p.x, p.y);
  }

  bool is_window_hovered(hovered_flag_t hovered_flags) {
    return ImGui::IsWindowHovered(hovered_flags);
  }

  bool is_window_focused() {
    return ImGui::IsWindowFocused();
  }

  void set_next_window_focus() {
    ImGui::SetNextWindowFocus();
  }

  void set_window_focus(label_t name) {
    ImGui::SetWindowFocus(fan::ct_string(name));
  }

  int render_window_flags() {
    return window_flags_no_title_bar | window_flags_no_background | window_flags_override_input;
  }

  fan::vec2 get_window_content_region_min() {
    ImVec2 v = ImGui::GetWindowContentRegionMin();
    return fan::vec2(v.x, v.y);
  }

  fan::vec2 get_window_content_region_max() {
    ImVec2 v = ImGui::GetWindowContentRegionMax();
    return fan::vec2(v.x, v.y);
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

  bool& want_io() {
    static bool g_want_io = false;
    return g_want_io;
  }

  bool& force_want_io_for_frame() {
    static bool g_force_io = false;
    return g_force_io;
  }

  void set_want_io(bool flag, bool op_or) {
    if (force_want_io_for_frame()) {
      want_io() = true;
      return;
    }

    ImGuiContext* g = ImGui::GetCurrentContext();

    if (g->NavWindow) {
      std::string nav_window_name = g->NavWindow->Name;
      if (nav_window_name.rfind("WindowOverViewport_", 0) == 0) {
        want_io() = false;
        return;
      }
    }

    if (g->NavWindow && detail::want_io_ignore_list().find(g->NavWindow->Name) != detail::want_io_ignore_list().end()) {
      want_io() = false;
      return;
    }

    if (g->HoveredWindow) {
      std::string nav_window_name = g->HoveredWindow->Name;
      if (nav_window_name.rfind("WindowOverViewport_", 0) == 0) {
        want_io() = false;
        return;
      }
    }

    if (g->HoveredWindow && detail::want_io_ignore_list().find(g->HoveredWindow->Name) != detail::want_io_ignore_list().end()) {
      want_io() = false;
      return;
    }

    want_io() = op_or ? (want_io() | flag) : flag;
  }

  void set_keyboard_focus_here() {
    ImGui::SetKeyboardFocusHere();
  }

  fan::vec2 get_mouse_drag_delta(int button, f32_t lock_threshold) {
    ImVec2 d = ImGui::GetMouseDragDelta(button, lock_threshold);
    return fan::vec2(d.x, d.y);
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
    ImGui::PushStyleColor(index, ImVec4(col.r, col.g, col.b, col.a));
  }

  void pop_style_color(int n) {
    ImGui::PopStyleColor(n);
  }

  void push_style_var(style_var_t index, f32_t val) {
    ImGui::PushStyleVar(index, val);
  }

  void push_style_var(style_var_t index, const fan::vec2& val) {
    ImGui::PushStyleVar(index, ImVec2(val.x, val.y));
  }

  void pop_style_var(int n) {
    ImGui::PopStyleVar(n);
  }

  bool button(label_t label, const fan::vec2& size) {
    return ImGui::Button(label, ImVec2(size.x, size.y));
  }

  bool invisible_button(label_t label, const fan::vec2& size) {
    return ImGui::InvisibleButton(label, ImVec2(size.x, size.y));
  }

  bool arrow_button(label_t label, dir_t dir) {
    return ImGui::ArrowButton(label, dir);
  }

  void text_at(std::string_view text, const fan::vec2& position, const fan::color& color) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(color.r, color.g, color.b, color.a));
    dl->AddText(ImVec2(position.x, position.y), col, text.data(), text.data() + text.size());
  }

  void text_wrapped(std::string_view text, const fan::color& color) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(color.r, color.g, color.b, color.a));
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(text.data(), text.data() + text.size());
    ImGui::PopTextWrapPos();
    ImGui::PopStyleColor();
  }

  void text_unformatted(std::string_view text, const char* text_end) {
    ImGui::TextUnformatted(text.data(), text_end ? text_end : text.data() + text.size());
  }

  void text_disabled(std::string_view text) {
    ImGui::TextDisabled("%.*s", (int)text.size(), text.data());
  }

  void text_centered(std::string_view text, const fan::color& color) {
    fan::vec2 text_size_v = calc_text_size(text);
    fan::vec2 window_size = get_window_size();
    fan::vec2 current_pos = get_cursor_pos();
    current_pos.x = (window_size.x - text_size_v.x) * 0.5f;
    set_cursor_pos(current_pos);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(color.r, color.g, color.b, color.a));
    ImGui::TextUnformatted(text.data(), text.data() + text.size());
    ImGui::PopStyleColor();
  }

  void text_centered_at(std::string_view text, const fan::vec2& center_position, const fan::color& color) {
    fan::vec2 text_size_v = calc_text_size(text);
    fan::vec2 draw_position = center_position;
    draw_position.x -= text_size_v.x * 0.5f;
    draw_position.y -= text_size_v.y * 0.5f;
    text_at(text, draw_position, color);
  }

  void text_bottom_right(std::string_view text, const fan::color& color, const fan::vec2& offset) {
    fan::vec2 text_size_v = calc_text_size(text);
    fan::vec2 window_pos = get_window_pos();
    fan::vec2 window_size = get_window_size();
    fan::vec2 pos;
    pos.x = window_pos.x + window_size.x - text_size_v.x - get_style().WindowPadding.x;
    pos.y = window_pos.y + window_size.y - text_size_v.y - get_style().WindowPadding.y;
    text_at(text, pos + offset, color);
  }

  void text_outlined_at(std::string_view text, const fan::vec2& screen_pos, const fan::color& color, const fan::color& outline_color) {
    detail::text_outlined_at_impl(text.data(), text.data() + text.size(), screen_pos, color, outline_color);
  }

  void text_outlined(std::string_view text, const fan::color& color, const fan::color& outline_color) {
    fan::vec2 cursor_pos = get_cursor_pos();
    text_outlined_at(text, cursor_pos, color, outline_color);
    fan::vec2 size = calc_text_size(text);
    ImGui::Dummy(ImVec2(size.x, size.y));
  }

  void text_centered_outlined_at(std::string_view text, const fan::vec2& center_position, const fan::color& color, const fan::color& outline_color) {
    fan::vec2 text_size_v = calc_text_size(text);
    fan::vec2 draw_position = center_position;
    draw_position.x -= text_size_v.x * 0.5f;
    draw_position.y -= text_size_v.y * 0.5f;
    text_outlined_at(text, draw_position, color, outline_color);
  }

  void text_centered_outlined(std::string_view text, const fan::color& color, const fan::color& outline_color) {
    fan::vec2 text_size_v = calc_text_size(text);
    fan::vec2 window_size = get_window_size();
    fan::vec2 current_pos = get_cursor_pos();
    current_pos.x = (window_size.x - text_size_v.x) * 0.5f;
    set_cursor_pos(current_pos);
    text_outlined(text, color, outline_color);
  }

  void text_box(std::string_view text,
    const fan::vec2& size,
    const fan::color& text_color,
    const fan::color& bg_color) {
    text_box_at(text, get_cursor_screen_pos(), size, text_color, bg_color);
  }

  void text_box_at(std::string_view text,
    const fan::vec2& pos,
    const fan::vec2& size,
    const fan::color& text_color,
    const fan::color& bg_color) {
    ImVec2 text_size_v = ImGui::CalcTextSize(text.data(), text.data() + text.size());
    ImVec2 padding = ImGui::GetStyle().FramePadding;
    ImVec2 box_size(size.x, size.y);
    if (box_size.x <= 0) {
      box_size.x = text_size_v.x + padding.x * 2;
    }
    if (box_size.y <= 0) {
      box_size.y = text_size_v.y + padding.y * 2;
    }

    ImDrawList* dl = ImGui::GetWindowDrawList();
    fan::color actual_bg = bg_color;
    if (bg_color.r == 0 && bg_color.g == 0 && bg_color.b == 0 && bg_color.a == 0) {
      ImVec4 def_bg = ImGui::GetStyleColorVec4(ImGuiCol_Button);
      actual_bg = fan::color(def_bg.x, def_bg.y, def_bg.z, def_bg.w);
    }

    ImU32 bg_u32 = ImGui::ColorConvertFloat4ToU32(ImVec4(actual_bg.r, actual_bg.g, actual_bg.b, actual_bg.a));
    ImU32 border_u32 = ImGui::GetColorU32(ImGuiCol_Border);
    f32_t rounding = ImGui::GetStyle().FrameRounding;

    ImVec2 p0(pos.x, pos.y);
    ImVec2 p1(pos.x + box_size.x, pos.y + box_size.y);
    dl->AddRectFilled(p0, p1, bg_u32, rounding);
    dl->AddRect(p0, p1, border_u32, rounding);

    ImVec2 text_pos(
      pos.x + (box_size.x - text_size_v.x) * 0.5f,
      pos.y + (box_size.y - text_size_v.y) * 0.5f
    );

    ImU32 text_u32 = ImGui::ColorConvertFloat4ToU32(ImVec4(text_color.r, text_color.g, text_color.b, text_color.a));
    dl->AddText(text_pos, text_u32, text.data(), text.data() + text.size());
    ImGui::Dummy(box_size);
  }

  f32_t calc_item_width() {
    return ImGui::CalcItemWidth();
  }

  f32_t get_item_width() {
    return calc_item_width();
  }

  bool input_text(label_t label, std::string* buf, input_text_flags_t flags, input_text_callback_t callback, void* user_data) {
    IM_ASSERT((flags & ImGuiInputTextFlags_CallbackResize) == 0);
    flags |= ImGuiInputTextFlags_CallbackResize;
    detail::input_text_callback_user_data_t cb_user_data;
    cb_user_data.str = buf;
    cb_user_data.chain_callback = callback;
    cb_user_data.chain_callback_user_data = user_data;
    return ImGui::InputText(label,
      buf->data(),
      buf->capacity() + 1,
      flags,
      detail::input_text_callback,
      &cb_user_data);
  }

  bool input_text_multiline(label_t label, std::string* buf, const fan::vec2& size, input_text_flags_t flags, input_text_callback_t callback, void* user_data) {
    IM_ASSERT((flags & ImGuiInputTextFlags_CallbackResize) == 0);
    flags |= ImGuiInputTextFlags_CallbackResize;
    detail::input_text_callback_user_data_t cb_user_data;
    cb_user_data.str = buf;
    cb_user_data.chain_callback = callback;
    cb_user_data.chain_callback_user_data = user_data;
    return ImGui::InputTextMultiline(label,
      buf->data(),
      buf->capacity() + 1,
      size,
      flags,
      detail::input_text_callback,
      &cb_user_data);
  }

  bool input_float(label_t label, f32_t* v, f32_t step, f32_t step_fast, const char* format, input_text_flags_t flags) {
    return ImGui::InputFloat(label, v, step, step_fast, format, flags);
  }

  bool input_float(label_t label, fan::vec2* v, const char* format, input_text_flags_t flags) {
    return ImGui::InputFloat2(label, v->data(), format, flags);
  }

  bool input_float(label_t label, fan::vec3* v, const char* format, input_text_flags_t flags) {
    return ImGui::InputFloat3(label, v->data(), format, flags);
  }

  bool input_float(label_t label, fan::vec4* v, const char* format, input_text_flags_t flags) {
    return ImGui::InputFloat4(label, v->data(), format, flags);
  }

  bool input_int(label_t label, int* v, int step, int step_fast, input_text_flags_t flags) {
    return ImGui::InputInt(label, v, step, step_fast, flags);
  }

  bool input_int(label_t label, fan::vec2i* v, input_text_flags_t flags) {
    return ImGui::InputInt2(label, v->data(), flags);
  }

  bool input_int(label_t label, fan::vec3i* v, input_text_flags_t flags) {
    return ImGui::InputInt3(label, v->data(), flags);
  }

  bool input_int(label_t label, fan::vec4i* v, input_text_flags_t flags) {
    return ImGui::InputInt4(label, v->data(), flags);
  }

  bool color_edit3(label_t label, fan::color* color, color_edit_flags_t flags) {
    return ImGui::ColorEdit3(label, color->data(), flags);
  }

  bool color_edit3(label_t label, fan::vec3* color, color_edit_flags_t flags) {
    return ImGui::ColorEdit3(label, color->data(), flags);
  }

  bool color_edit4(label_t label, fan::color* color, color_edit_flags_t flags) {
    return ImGui::ColorEdit4(label, color->data(), flags);
  }

  fan::vec2 get_window_pos() {
    ImVec2 p = ImGui::GetWindowPos();
    return fan::vec2(p.x, p.y);
  }

  fan::vec2 get_window_size() {
    ImVec2 s = ImGui::GetWindowSize();
    return fan::vec2(s.x, s.y);
  }

  void set_next_window_pos(const fan::vec2& position, cond_t cond, const fan::vec2& pivot) {
    ImGui::SetNextWindowPos(ImVec2(position.x, position.y), cond, ImVec2(pivot.x, pivot.y));
  }

  void set_next_window_size(const fan::vec2& size, cond_t cond) {
    ImGui::SetNextWindowSize(ImVec2(size.x, size.y), cond);
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
    ImVec4 c = ImGui::GetStyle().Colors[idx];
    return fan::color(c.x, c.y, c.z, c.w);
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

  bool tree_node_ex(label_t label, tree_node_flags_t flags) {
    return ImGui::TreeNodeEx(label, flags);
  }

  void tree_pop() {
    ImGui::TreePop();
  }

  bool tree_node(label_t label) {
    return ImGui::TreeNode(label);
  }

  bool is_item_toggled_open() {
    return ImGui::IsItemToggledOpen();
  }

  void dummy(const fan::vec2& size) {
    ImGui::Dummy(ImVec2(size.x, size.y));
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

  bool combo(label_t label, int* current_item, const char* const items[], int items_count, int popup_max_height_in_items) {
    return ImGui::Combo(label, current_item, items, items_count, popup_max_height_in_items);
  }

  bool combo(label_t label, int* current_item, const char* items_separated_by_zeros, int popup_max_height_in_items) {
    return ImGui::Combo(label, current_item, items_separated_by_zeros, popup_max_height_in_items);
  }

  bool combo(label_t label, int* current_item, const char* (*getter)(void* user_data, int idx), void* user_data, int items_count, int popup_max_height_in_items) {
    return ImGui::Combo(label, current_item, getter, user_data, items_count, popup_max_height_in_items);
  }

  bool checkbox(label_t label, bool* v) {
    return ImGui::Checkbox(label, v);
  }

  bool list_box(label_t label, int* current_item, bool (*old_callback)(void* user_data, int idx, const char** out_text), void* user_data, int items_count, int height_in_items) {
    return ImGui::ListBox(label, current_item, old_callback, user_data, items_count, height_in_items);
  }

  bool list_box(label_t label, int* current_item, const char* const items[], int items_count, int height_in_items) {
    return ImGui::ListBox(label, current_item, items, items_count, height_in_items);
  }

  bool list_box(label_t label, int* current_item, const char* (*getter)(void* user_data, int idx), void* user_data, int items_count, int height_in_items) {
    return ImGui::ListBox(label, current_item, getter, user_data, items_count, height_in_items);
  }

  bool toggle_button(label_t str, bool* v) {
    return detail::toggle_button_impl(fan::ct_string(str), v);
  }

  void text_bottom_right(std::string_view text, uint32_t reverse_yoffset) {
    fan::vec2 pos = get_position_bottom_corner(text, reverse_yoffset);
    text_at(text, pos, fan::colors::white);
  }

  fan::vec2 get_position_bottom_corner(std::string_view text, uint32_t reverse_yoffset) {
    fan::vec2 text_size_v = calc_text_size(text);
    fan::vec2 window_pos = get_window_pos();
    fan::vec2 window_size = get_window_size();
    fan::vec2 pos;
    pos.x = window_pos.x + window_size.x - text_size_v.x - get_style().WindowPadding.x;
    pos.y = window_pos.y + window_size.y - text_size_v.y - get_style().WindowPadding.y - reverse_yoffset;
    return pos;
  }

  void send_drag_drop_item(label_t id, const std::wstring& path, label_t popup) {
    if (ImGui::BeginDragDropSource()) {
      ImGui::SetDragDropPayload(fan::ct_string(id), path.c_str(), (path.size() + 1) * sizeof(wchar_t));
      if (!popup.empty()) {
        ImGui::TextUnformatted(popup.data(), popup.data() + popup.size());
      }
      ImGui::EndDragDropSource();
    }
  }

  void receive_drag_drop_target(label_t id, std::function<void(std::string)> receive_func) {
    detail::receive_drag_drop_target_impl(fan::ct_string(id), std::move(receive_func));
  }

  bool slider_scalar(label_t label, data_type_t data_type, void* p_data, const void* p_min, const void* p_max, const char* format, slider_flags_t flags) {
    return ImGui::SliderScalar(label, data_type, p_data, p_min, p_max, format, flags);
  }

  bool slider_scalar_n(label_t label, data_type_t data_type, void* p_data, int components, const void* p_min, const void* p_max, const char* format, slider_flags_t flags) {
    return ImGui::SliderScalarN(label, data_type, p_data, components, p_min, p_max, format, flags);
  }

  bool drag_scalar(label_t label, data_type_t data_type, void* p_data, f32_t v_speed, const void* p_min, const void* p_max, const char* format, slider_flags_t flags) {
    return ImGui::DragScalar(label, data_type, p_data, v_speed, p_min, p_max, format, flags);
  }

  bool drag_scalar_n(label_t label, data_type_t data_type, void* p_data, int components, f32_t v_speed, const void* p_min, const void* p_max, const char* format, slider_flags_t flags) {
    return ImGui::DragScalarN(label, data_type, p_data, components, v_speed, p_min, p_max, format, flags);
  }

  font_t* get_font_impl(f32_t font_size, bool bold) {
    return detail::get_font_from_array(bold ? get_font_bold() : get_font_main(), font_size);
  }

  font_t* get_font(font_t* (&fonts)[std::size(fan::graphics::gui::font_sizes)], f32_t font_size) {
    return detail::get_font_from_array(fonts, font_size);
  }

  font_t* get_font(f32_t font_size, bool bold) {
    return get_font_impl(font_size, bold);
  }

  void image(texture_id_t texture,
    const fan::vec2& size,
    const fan::vec2& uv0,
    const fan::vec2& uv1,
    const fan::color& tint_col,
    const fan::color& border_col) {
    ImGui::Image(texture,
      ImVec2(size.x, size.y),
      ImVec2(uv0.x, uv0.y),
      ImVec2(uv1.x, uv1.y),
      ImVec4(tint_col.r, tint_col.g, tint_col.b, tint_col.a),
      ImVec4(border_col.r, border_col.g, border_col.b, border_col.a));
  }

  void image(fan::graphics::image_t img,
    const fan::vec2& size,
    const fan::vec2& uv0,
    const fan::vec2& uv1,
    const fan::color& tint_col,
    const fan::color& border_col) {
    texture_id_t tex = static_cast<texture_id_t>(fan::graphics::ctx()->image_get_handle(fan::graphics::ctx(), img));
    image(tex, size, uv0, uv1, tint_col, border_col);
  }

  bool image_button(label_t str_id,
    fan::graphics::image_t img,
    const fan::vec2& size,
    const fan::vec2& uv0,
    const fan::vec2& uv1,
    int frame_padding,
    const fan::color& bg_col,
    const fan::color& tint_col) {
    return detail::image_button_img_impl(fan::ct_string(str_id), img, size, uv0, uv1, frame_padding, bg_col, tint_col);
  }

  bool image_text_button(fan::graphics::image_t img,
    std::string_view text,
    const fan::color& color,
    const fan::vec2& size,
    const fan::vec2& uv0,
    const fan::vec2& uv1,
    int frame_padding,
    const fan::color& bg_col,
    const fan::color& tint_col) {
    return detail::image_text_button_impl(img, fan::ct_string(text), color, size, uv0, uv1, frame_padding, bg_col, tint_col);
  }

  bool image_button(label_t str_id,
    texture_id_t texture,
    const fan::vec2& size,
    const fan::vec2& uv0,
    const fan::vec2& uv1,
    const fan::color& bg_col,
    const fan::color& tint_col) {
    ImVec4 bg(bg_col.r, bg_col.g, bg_col.b, bg_col.a);
    ImVec4 tint(tint_col.r, tint_col.g, tint_col.b, tint_col.a);
    return ImGui::ImageButton(fan::ct_string(str_id),
      texture,
      ImVec2(size.x, size.y),
      ImVec2(uv0.x, uv0.y),
      ImVec2(uv1.x, uv1.y),
      bg,
      tint);
  }

  bool item_add(const rect_t& bb, id_t id, const rect_t* nav_bb, item_flags_t extra_flags) {
    return ImGui::ItemAdd(bb, id, nav_bb, extra_flags);
  }

  int is_key_clicked(key_t key, bool repeat) {
    return ImGui::IsKeyPressed(key, repeat) ? 1 : 0;
  }

  int get_pressed_key() {
    for (int k = ImGuiKey_NamedKey_BEGIN; k < ImGuiKey_NamedKey_END; ++k) {
      if (ImGui::IsKeyPressed(static_cast<ImGuiKey>(k))) {
        return k;
      }
    }
    return -1;
  }
  void set_next_window_class(const class_t* c) {
    ImGui::SetNextWindowClass(static_cast<const ImGuiWindowClass*>(c));
  }

  bool begin_drag_drop_source() {
    return ImGui::BeginDragDropSource();
  }

  bool set_drag_drop_payload(label_t type, const void* data, size_t sz, cond_t cond) {
    return ImGui::SetDragDropPayload(fan::ct_string(type), data, sz, cond);
  }

  void end_drag_drop_source() {
    ImGui::EndDragDropSource();
  }

  bool begin_drag_drop_target() {
    return ImGui::BeginDragDropTarget();
  }

  const payload_t* accept_drag_drop_payload(label_t type) {
    return ImGui::AcceptDragDropPayload(fan::ct_string(type));
  }

  void end_drag_drop_target() {
    ImGui::EndDragDropTarget();
  }

  const payload_t* get_drag_drop_payload() {
    return ImGui::GetDragDropPayload();
  }

  bool begin_popup(label_t id, window_flags_t flags) {
    return ImGui::BeginPopup(fan::ct_string(id), flags);
  }

  bool begin_popup_modal(label_t id, window_flags_t flags) {
    return ImGui::BeginPopupModal(fan::ct_string(id), nullptr, flags);
  }

  void end_popup() {
    ImGui::EndPopup();
  }

  void open_popup(label_t id) {
    ImGui::OpenPopup(fan::ct_string(id));
  }

  void close_current_popup() {
    ImGui::CloseCurrentPopup();
  }

  bool is_popup_open(label_t id) {
    return ImGui::IsPopupOpen(fan::ct_string(id));
  }

  id_t get_id(label_t str_id) {
    return ImGui::GetID(fan::ct_string(str_id));
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

  void dock_space_over_viewport(id_t dockspace_id, const gui::viewport_t* viewport, int flags, const void* window_class) {
    ImGui::DockSpaceOverViewport(dockspace_id, viewport, flags, static_cast<const ImGuiWindowClass*>(window_class));
  }

  context_t* get_context() {
    return ImGui::GetCurrentContext();
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
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
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
    #if defined(FAN_VULKAN)
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

    is_gui_initialized() = true;
  }

  void init_graphics_context(
    GLFWwindow* window,
    int renderer,
    int opengl_renderer_definition,
    int vulkan_renderer_definition
  #if defined(FAN_VULKAN)
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

  #if defined(FAN_VULKAN)
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

  void init_fonts() {
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->FontBuilderIO = ImGuiFreeType::GetBuilderForFreeType();
    io.Fonts->FontBuilderFlags |= ImGuiFreeTypeBuilderFlags_LoadColor;

    load_fonts(fan::graphics::gui::get_font_main(), "fonts/SourceCodePro-Regular.ttf");
    load_fonts(fan::graphics::gui::get_font_bold(), "fonts/SourceCodePro-Bold.ttf");

    build_fonts();

    io.FontDefault = fan::graphics::gui::get_font_main()[default_font_size_index];
  }

  void load_emojis() {
    ImFontConfig emoji_cfg;
    emoji_cfg.FontBuilderFlags |= ImGuiFreeTypeBuilderFlags_LoadColor | ImGuiFreeTypeBuilderFlags_Bitmap;

    // TODO: expand ranges if needed
    static const ImWchar emoji_ranges[] = {
      0x2600, 0x26FF,
      0x2B00, 0x2BFF,
      0x1F600, 0x1F64F,
      0
    };

    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->Clear();
    io.Fonts->FontBuilderIO = ImGuiFreeType::GetBuilderForFreeType();
    io.Fonts->FontBuilderFlags |= ImGuiFreeTypeBuilderFlags_LoadColor;

    for (std::size_t i = 0; i < std::size(fan::graphics::gui::get_font_main()); ++i) {
      f32_t font_size = fan::graphics::gui::font_sizes[i] * 2;

      ImFontConfig main_cfg;
      fan::graphics::gui::get_font_main()[i] = io.Fonts->AddFontFromFileTTF(
        "fonts/SourceCodePro-Regular.ttf", font_size, &main_cfg
      );

      ImFontConfig emoji_cfg2;
      emoji_cfg2.MergeMode = true;
      emoji_cfg2.FontBuilderFlags |= ImGuiFreeTypeBuilderFlags_LoadColor;
      emoji_cfg2.SizePixels = 0;
      emoji_cfg2.RasterizerDensity = 1.0f;
      emoji_cfg2.GlyphMinAdvanceX = font_size;

      io.Fonts->AddFontFromFileTTF(
        "fonts/seguiemj.ttf", font_size, &emoji_cfg2, emoji_ranges
      );
    }

    load_fonts(fan::graphics::gui::get_font_bold(), "fonts/SourceCodePro-Bold.ttf");
    build_fonts();
    io.FontDefault = fan::graphics::gui::get_font_main()[9];
  }

  void shutdown_graphics_context(
    int renderer,
    int opengl_renderer_definition,
    int vulkan_renderer_definition
  #if defined(FAN_VULKAN)
    , VkDevice device
  #endif
  ) {
    if (renderer == opengl_renderer_definition) {
      ImGui_ImplOpenGL3_Shutdown();
    }
  #if defined(FAN_VULKAN)
    if (renderer == vulkan_renderer_definition) {
      ImGui_ImplVulkan_Shutdown();
    }
  #endif
    ImGui_ImplGlfw_Shutdown();
  }

  void shutdown_window_context() {
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
  }

  void destroy() {
    is_gui_initialized() = false;
  }

  void new_frame(
    int renderer,
    int opengl_renderer_definition,
    int vulkan_renderer_definition
  ) {
    ImGui_ImplGlfw_NewFrame();
    if (renderer == opengl_renderer_definition) {
      ImGui_ImplOpenGL3_NewFrame();
    }
  #if defined(FAN_VULKAN)
    if (renderer == vulkan_renderer_definition) {
      ImGui_ImplVulkan_NewFrame();
    }
  #endif
    ImGui::NewFrame();
    force_want_io_for_frame() = false;
    detail::want_io_ignore_list().clear();
  }

#if defined(FAN_VULKAN)
  void render(
    int renderer,
    int opengl_renderer_definition,
    int vulkan_renderer_definition,
    bool render_shapes_top,
    void* context,
    const fan::color& clear_color,
    VkResult& image_error,
    VkCommandBuffer& cmd_buffer,
    ImGuiFrameRenderFunc render_func
  ) {
#else
  void render(
    int renderer,
    int opengl_renderer_definition,
    int vulkan_renderer_definition,
    bool render_shapes_top
  ) {
#endif
    ImGui::Render();

    if (renderer == opengl_renderer_definition) {
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

  #if defined(FAN_VULKAN)
    if (renderer == vulkan_renderer_definition) {
      ImDrawData* draw_data = ImGui::GetDrawData();
      ImGui_ImplVulkan_RenderDrawData(draw_data, cmd_buffer);
      render_func(context, image_error, clear_color);
    }
  #endif

    if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
      ImGui::UpdatePlatformWindows();
      ImGui::RenderPlatformWindowsDefault();
    }
  }

  void profile_heap(void* (*dynamic_malloc)(size_t, void*), void (*dynamic_free)(void*, void*)) {
    ImGui::SetAllocatorFunctions(dynamic_malloc, dynamic_free);
  }

  window::window(label_t title, window_flags_t flags)
    : wnd(title, nullptr, flags) {}

  window::window(label_t title, bool* p_open, window_flags_t flags)
    : wnd(title, p_open, flags) {}

  window::operator bool() const {
    return static_cast<bool>(wnd);
  }

  hud::hud(label_t name, bool* p_open)
    : wnd(
      (
        gui::set_next_window_pos(gui::get_viewport_rect().position),
        gui::set_next_window_size(gui::get_viewport_rect().size),
        name
      ),
      p_open,
      gui::window_flags_no_background |
      gui::window_flags_no_nav |
      gui::window_flags_no_inputs |
      gui::window_flags_no_title_bar |
      gui::window_flags_no_resize |
      gui::window_flags_no_move |
      gui::window_flags_no_saved_settings |
      gui::window_flags_override_input
    ) {}

  hud::operator bool() const {
    return static_cast<bool>(wnd);
  }

  table::table(label_t str_id,
    int columns,
    table_flags_t flags,
    const fan::vec2& outer_size,
    f32_t inner_width)
    : tbl(str_id, columns, flags, outer_size, inner_width) {}

  table::operator bool() const {
    return static_cast<bool>(tbl);
  }

} // namespace fan::graphics::gui

namespace fan::graphics::gui::plot {

  bool begin_plot(label_t title, const fan::vec2& size, flags_t flags) {
    return ImPlot::BeginPlot(fan::ct_string(title), ImVec2(size.x, size.y), flags);
  }

  void end_plot() {
    ImPlot::EndPlot();
  }

  void setup_axes(label_t x_label, label_t y_label, axis_flags_t x_flags, axis_flags_t y_flags) {
    ImPlot::SetupAxes(fan::ct_string(x_label), fan::ct_string(y_label), x_flags, y_flags);
  }

  void setup_axis(axis_t axis, label_t label, axis_flags_t flags) {
    ImPlot::SetupAxis(axis, label, flags);
  }

  void setup_axis_limits(axis_t axis, double v_min, double v_max, cond_t cond) {
    ImPlot::SetupAxisLimits(axis, v_min, v_max, cond);
  }

  void setup_axes_limits(double x_min, double x_max, double y_min, double y_max, cond_t cond) {
    ImPlot::SetupAxesLimits(x_min, x_max, y_min, y_max, cond);
  }

  void setup_axis_format(axis_t idx, std::string_view format) {
    ImPlot::SetupAxisFormat(idx, fan::ct_string(format));
  }

  void setup_axis_links(axis_t idx, double* min_lnk, double* max_lnk) {
    ImPlot::SetupAxisLinks(idx, min_lnk, max_lnk);
  }

  void setup_axis_format(axis_t idx, formatter_t formatter, void* data) {
    ImPlot::SetupAxisFormat(idx, formatter, data);
  }

  void setup_legend(location_t location, int flags) {
    ImPlot::SetupLegend(location, flags);
  }

  void setup_finish() {
    ImPlot::SetupFinish();
  }

  void plot_line(label_t label, const std::vector<f32_t>& values, double x_scale, double x_start, line_flags_t flags) {
    ImPlot::PlotLine(label, values.data(), (int)values.size(), x_scale, x_start, flags);
  }

  void plot_line(label_t label, const std::vector<f32_t>& xs, const std::vector<f32_t>& ys, line_flags_t flags) {
    ImPlot::PlotLine(label, xs.data(), ys.data(), (int)xs.size(), flags);
  }

  void plot_line(label_t label, const f32_t* xs, const f32_t* ys, int count, line_flags_t flags) {
    ImPlot::PlotLine(label, xs, ys, count, flags);
  }

  void plot_scatter(label_t label, const std::vector<f32_t>& values, double x_scale, double x_start, scatter_flags_t flags) {
    ImPlot::PlotScatter(label, values.data(), (int)values.size(), x_scale, x_start, flags);
  }

  void plot_scatter(label_t label, const std::vector<f32_t>& xs, const std::vector<f32_t>& ys, scatter_flags_t flags) {
    ImPlot::PlotScatter(label, xs.data(), ys.data(), (int)xs.size(), flags);
  }

  void plot_scatter(label_t label, const f32_t* xs, const f32_t* ys, int count, scatter_flags_t flags) {
    ImPlot::PlotScatter(label, xs, ys, count, flags);
  }

  void plot_bars(label_t label, const std::vector<f32_t>& values, double bar_size, double shift, bars_flags_t flags) {
    ImPlot::PlotBars(label, values.data(), (int)values.size(), bar_size, shift, flags);
  }

  void plot_bars(label_t label, const std::vector<f32_t>& xs, const std::vector<f32_t>& ys, double bar_size, bars_flags_t flags) {
    ImPlot::PlotBars(label, xs.data(), ys.data(), (int)xs.size(), bar_size, flags);
  }

  void plot_shaded(label_t label, const std::vector<f32_t>& xs, const std::vector<f32_t>& ys, double y_ref, int flags) {
    ImPlot::PlotShaded(label, xs.data(), ys.data(), (int)xs.size(), y_ref, flags);
  }

  void push_style_color(col_t idx, const fan::color& color) {
    ImPlot::PushStyleColor(idx, ImVec4(color.r, color.g, color.b, color.a));
  }

  void pop_style_color(int count) {
    ImPlot::PopStyleColor(count);
  }

  void push_style_var(int idx, f32_t val) {
    ImPlot::PushStyleVar(idx, val);
  }

  void push_style_var(int idx, const fan::vec2& val) {
    ImPlot::PushStyleVar(idx, ImVec2(val.x, val.y));
  }

  void pop_style_var(int count) {
    ImPlot::PopStyleVar(count);
  }

  void set_next_line_style(const fan::color& col, f32_t weight) {
    ImPlot::SetNextLineStyle(ImVec4(col.r, col.g, col.b, col.a), weight);
  }

  void set_next_fill_style(const fan::color& col, f32_t alpha_mod) {
    ImPlot::SetNextFillStyle(ImVec4(col.r, col.g, col.b, col.a), alpha_mod);
  }

  void set_next_marker_style(marker_t marker, f32_t size, const fan::color& fill, f32_t weight, const fan::color& outline) {
    ImPlot::SetNextMarkerStyle(marker,
      size,
      ImVec4(fill.r, fill.g, fill.b, fill.a),
      weight,
      ImVec4(outline.r, outline.g, outline.b, outline.a));
  }

  fan::vec2 get_plot_pos() {
    ImVec2 p = ImPlot::GetPlotPos();
    return fan::vec2(p.x, p.y);
  }

  fan::vec2 get_plot_size() {
    ImVec2 s = ImPlot::GetPlotSize();
    return fan::vec2(s.x, s.y);
  }

  bool is_plot_hovered() {
    return ImPlot::IsPlotHovered();
  }

  bool is_axis_hovered(axis_t axis) {
    return ImPlot::IsAxisHovered(axis);
  }

  fan::vec2 pixels_to_plot(const fan::vec2& pix, axis_t x_axis, axis_t y_axis) {
    ImPlotPoint p = ImPlot::PixelsToPlot(ImVec2(pix.x, pix.y), x_axis, y_axis);
    return fan::vec2(p.x, p.y);
  }

  fan::vec2 plot_to_pixels(double x, double y, axis_t x_axis, axis_t y_axis) {
    ImVec2 p = ImPlot::PlotToPixels(x, y, x_axis, y_axis);
    return fan::vec2(p.x, p.y);
  }

  fan::vec2 get_plot_mouse_pos(axis_t x_axis, axis_t y_axis) {
    ImPlotPoint p = ImPlot::GetPlotMousePos(x_axis, y_axis);
    return fan::vec2(p.x, p.y);
  }

  void annotation(double x, double y, const fan::color& col, const fan::vec2& pix_offset, bool clamp, std::string_view text) {
    ImPlot::Annotation(x, y, ImVec4(col.r, col.g, col.b, col.a),
      ImVec2(pix_offset.x, pix_offset.y),
      clamp,
      "%.*s",
      (int)text.size(),
      text.data());
  }

  void tag_x(double x, const fan::color& col, std::string_view text) {
    ImPlot::TagX(x, ImVec4(col.r, col.g, col.b, col.a), "%.*s", (int)text.size(), text.data());
  }

  void tag_y(double y, const fan::color& col, std::string_view text) {
    ImPlot::TagY(y, ImVec4(col.r, col.g, col.b, col.a), "%.*s", (int)text.size(), text.data());
  }

  void plot_text(std::string_view text, double x, double y, const fan::vec2& pix_offset, int flags) {
    ImPlot::PlotText(fan::ct_string(text), x, y, ImVec2(pix_offset.x, pix_offset.y), flags);
  }

  void plot_dummy(label_t label_id, int flags) {
    ImPlot::PlotDummy(fan::ct_string(label_id), flags);
  }

  fan::color next_colormap_color() {
    ImVec4 c = ImPlot::NextColormapColor();
    return fan::color(c.x, c.y, c.z, c.w);
  }

  fan::color get_last_item_color() {
    ImVec4 c = ImPlot::GetLastItemColor();
    return fan::color(c.x, c.y, c.z, c.w);
  }

  void setup_axis_ticks(axis_t axis, const double* values, int n_ticks, const char* const labels[], bool keep_default) {
    ImPlot::SetupAxisTicks(axis, values, n_ticks, labels, keep_default);
  }

  void setup_axis_ticks(axis_t axis, double v_min, double v_max, int n_ticks, const char* const labels[], bool keep_default) {
    ImPlot::SetupAxisTicks(axis, v_min, v_max, n_ticks, labels, keep_default);
  }

  void push_plot_clip_rect(f32_t expand) {
    ImPlot::PushPlotClipRect(expand);
  }

  void pop_plot_clip_rect() {
    ImPlot::PopPlotClipRect();
  }

} // namespace fan::graphics::gui::plot

#if defined(FAN_AUDIO)
namespace fan::graphics::gui {
  bool audio_button(label_t label, fan::audio::piece_t piece_hover, fan::audio::piece_t piece_click, const fan::vec2& size) {
    return detail::audio_button_impl(label, piece_hover, piece_click, size);
  }
}
#endif

namespace fan::graphics::gui::gizmo {

  void begin_frame() {
    ImGuizmo::BeginFrame();
  }

  void set_orthographic(bool ortho) {
    ImGuizmo::SetOrthographic(ortho);
  }

  void set_drawlist() {
    ImGuizmo::SetDrawlist();
  }

  void set_rect(const fan::vec2& pos, const fan::vec2& size) {
    ImGuizmo::SetRect(pos.x, pos.y, size.x, size.y);
  }

  bool manipulate(const fan::mat4& view,
    const fan::mat4& projection,
    int op,
    int m,
    fan::mat4& transform,
    const fan::mat4* delta,
    const fan::mat4* snap,
    const fan::mat4* bounds,
    const fan::mat4* bounds_snap) {
    return ImGuizmo::Manipulate(
      &view[0][0],
      &projection[0][0],
      static_cast<ImGuizmo::OPERATION>(op),
      static_cast<ImGuizmo::MODE>(m),
      &transform[0][0],
      delta ? &(*delta)[0][0] : nullptr,
      snap ? &(*snap)[0][0] : nullptr,
      bounds ? &(*bounds)[0][0] : nullptr,
      bounds_snap ? &(*bounds_snap)[0][0] : nullptr
    );
  }

  bool is_using() {
    return ImGuizmo::IsUsing();
  }

  bool is_over() {
    return ImGuizmo::IsOver();
  }

  bool is_using_any() {
    return ImGuizmo::IsUsingAny();
  }

  void draw_grid(const fan::mat4& view,
    const fan::mat4& projection,
    const fan::mat4& matrix,
    float size) {
    ImGuizmo::DrawGrid(&view[0][0], &projection[0][0], &matrix[0][0], size);
  }

} // namespace fan::graphics::gui::gizmo

namespace fan::graphics::gui::slot {

  void background(gui::draw_list_t* dl,
    const fan::vec2& p_min,
    const fan::vec2& p_max,
    const fan::color& color,
    f32_t rounding) {
    ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(color.r, color.g, color.b, color.a));
    dl->AddRectFilled(ImVec2(p_min.x, p_min.y), ImVec2(p_max.x, p_max.y), col, rounding);
  }

  void border(gui::draw_list_t* dl,
    const fan::vec2& p_min,
    const fan::vec2& p_max,
    const fan::color& color,
    f32_t rounding,
    f32_t thickness) {
    ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(color.r, color.g, color.b, color.a));
    dl->AddRect(ImVec2(p_min.x, p_min.y), ImVec2(p_max.x, p_max.y), col, rounding, 0, thickness);
  }

  void selected_border(gui::draw_list_t* dl,
    const fan::vec2& p_min,
    const fan::vec2& p_max,
    const fan::color& color,
    f32_t thickness,
    f32_t expand) {
    fan::vec2 min = p_min - fan::vec2(expand, expand);
    fan::vec2 max = p_max + fan::vec2(expand, expand);
    ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(color.r, color.g, color.b, color.a));
    dl->AddRect(ImVec2(min.x, min.y), ImVec2(max.x, max.y), col, 0.f, 0, thickness);
  }

  void icon(const fan::graphics::image_t& img,
    const fan::vec2& p_min,
    const fan::vec2& p_max,
    const fan::vec2& padding) {
    texture_id_t tex = static_cast<texture_id_t>(fan::graphics::ctx()->image_get_handle(fan::graphics::ctx(), img));
    fan::vec2 size = p_max - p_min - padding * 2.f;
    fan::vec2 pos = p_min + padding;
    ImGui::GetWindowDrawList()->AddImage(tex,
      ImVec2(pos.x, pos.y),
      ImVec2(pos.x + size.x, pos.y + size.y));
  }

  void stack_count(uint32_t count,
    const fan::vec2& p_min,
    const fan::vec2& p_max) {
    if (count <= 1) return;
    std::string s = std::to_string(count);
    fan::vec2 ts = gui::calc_text_size(s);
    gui::set_cursor_screen_pos(fan::vec2(p_max.x - ts.x - 4, p_max.y - ts.y - 4));
    gui::text(s.c_str());
  }

  void tooltip(std::string_view text, bool show) {
    detail::tooltip_impl(text.data(), text.data() + text.size(), show);
  }

} // namespace fan::graphics::gui::slot

#endif