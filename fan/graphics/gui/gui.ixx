module;

#include <fan/types/types.h>
#include <fan/time/timer.h>
#include <fan/event/types.h>

#if defined(fan_gui)
  #include <fan/imgui/imgui.h>
  #include <fan/imgui/imgui_internal.h>
  #include <fan/imgui/imgui_impl_glfw.h>
  #include <fan/imgui/imgui_neo_sequencer.h>
  #include <fan/imgui/implot.h>
#endif

#include <string>
#include <array>
#include <filesystem>
#include <coroutine>

export module fan:graphics.gui;

import :print;

#if defined(fan_gui)
  import :types.fstring;
  import :event;

  import :graphics;
  import :io.file;
  import :io.directory;

#endif


#if defined(fan_gui)

export namespace fan {
  namespace graphics {
    #if defined(fan_gui)
      namespace gui {

        using window_flags_t = int;
        enum {
          window_flags_none = ImGuiWindowFlags_None,
          window_flags_no_title_bar = ImGuiWindowFlags_NoTitleBar,   
          window_flags_no_resize = ImGuiWindowFlags_NoResize,   
          window_flags_no_move = ImGuiWindowFlags_NoMove,   
          window_flags_no_scrollbar = ImGuiWindowFlags_NoScrollbar,   
          window_flags_no_scroll_with_mouse = ImGuiWindowFlags_NoScrollWithMouse,   
          window_flags_no_collapse = ImGuiWindowFlags_NoCollapse,   
          window_flags_always_auto_resize = ImGuiWindowFlags_AlwaysAutoResize,   
          window_flags_no_background = ImGuiWindowFlags_NoBackground,   
          window_flags_no_saved_settings = ImGuiWindowFlags_NoSavedSettings,   
          window_flags_no_mouse_inputs = ImGuiWindowFlags_NoMouseInputs,   
          window_flags_menu_bar = ImGuiWindowFlags_MenuBar,  
          window_flags_horizontal_scrollbar = ImGuiWindowFlags_HorizontalScrollbar,  
          window_flags_no_focus_on_appearing = ImGuiWindowFlags_NoFocusOnAppearing,  
          window_flags_no_bring_to_front_on_focus = ImGuiWindowFlags_NoBringToFrontOnFocus,  
          window_flags_always_vertical_scrollbar = ImGuiWindowFlags_AlwaysVerticalScrollbar,  
          window_flags_always_horizontal_scrollbar = ImGuiWindowFlags_AlwaysHorizontalScrollbar,  
          window_flags_no_nav_inputs = ImGuiWindowFlags_NoNavInputs,  
          window_flags_no_nav_focus = ImGuiWindowFlags_NoNavFocus,  
          window_flags_unsaved_document = ImGuiWindowFlags_UnsavedDocument,  
          window_flags_no_docking = ImGuiWindowFlags_NoDocking,  
          window_flags_no_nav = ImGuiWindowFlags_NoNav,  
          window_flags_no_decoration = ImGuiWindowFlags_NoDecoration,  
          window_flags_no_inputs = ImGuiWindowFlags_NoInputs,
        };
        using child_window_flags_t = int;
        enum {
          child_flags_none = ImGuiChildFlags_None,
          child_flags_borders = ImGuiChildFlags_Borders,
          child_flags_always_use_window_padding = ImGuiChildFlags_AlwaysUseWindowPadding,
          child_flags_resize_x = ImGuiChildFlags_ResizeX,
          child_flags_resize_y = ImGuiChildFlags_ResizeY,
          child_flags_auto_resize_x = ImGuiChildFlags_AutoResizeX,
          child_flags_auto_resize_y = ImGuiChildFlags_AutoResizeY,
          child_flags_always_auto_resize = ImGuiChildFlags_AlwaysAutoResize,
          child_flags_frame_style = ImGuiChildFlags_FrameStyle,
          child_flags_nav_flattened = ImGuiChildFlags_NavFlattened,
        };

        bool begin(const std::string& window_name, bool* p_open = 0, window_flags_t window_flags = 0) {
          return ImGui::Begin(window_name.c_str(), p_open, window_flags);
        }
        void end() {
          ImGui::End();
        }
        bool begin_child(const std::string& window_name, const fan::vec2& size = fan::vec2(0, 0), child_window_flags_t window_flags = 0) {
          return ImGui::BeginChild(window_name.c_str(), size, window_flags);
        }
        void end_child() {
          ImGui::EndChild();
        }

        bool begin_tab_item(const std::string& label, bool* p_open = 0, window_flags_t window_flags = 0) {
          return ImGui::BeginTabItem(label.c_str(), p_open, window_flags);
        }
        void end_tab_item() {
          ImGui::EndTabItem();
        }
        bool begin_tab_bar(const std::string& tab_bar_name, window_flags_t window_flags = 0) {
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

        bool begin_menu(const std::string& label) {
          return ImGui::BeginMenu(label.c_str());
        }
        void end_menu() {
          ImGui::EndMenu();
        }
        bool menu_item(const std::string& label, const std::string& shortcut = "", bool selected = false, bool enabled = true) {
          return ImGui::MenuItem(label.c_str(), shortcut.empty() ? nullptr : shortcut.c_str(), selected, enabled);
        }

        void same_line(f32_t offset_from_start_x = 0.f, f32_t spacing_w = -1.f) {
          ImGui::SameLine(offset_from_start_x, spacing_w);
        }
        void new_line() {
          ImGui::NewLine();
        }

        ImGuiViewport* get_main_viewport() {
          return ImGui::GetMainViewport();
        }

        f32_t get_text_line_height_with_spacing() {
          return ImGui::GetTextLineHeightWithSpacing();
        }

        fan::vec2 get_mouse_pos() {
          ImVec2 pos = ImGui::GetMousePos();
          return fan::vec2(pos.x, pos.y);
        }


        fan::vec2 get_content_region_avail () {
          return ImGui::GetContentRegionAvail();
        }
        fan::vec2 get_content_region_max() {
          return ImGui::GetContentRegionMax();
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

        using hovered_flag_t = int;
        enum {
          hovered_flags_none = ImGuiHoveredFlags_None,                          // Return true if directly over the item/window, not obstructed by another window, not obstructed by an active popup or modal blocking inputs under them.
          hovered_flags_child_windows = ImGuiHoveredFlags_ChildWindows,                 // IsWindowHovered() only: Return true if any children of the window is hovered
          hovered_flags_root_window = ImGuiHoveredFlags_RootWindow,                   // IsWindowHovered() only: Test from root window (top most parent of the current hierarchy)
          hovered_flags_any_window = ImGuiHoveredFlags_AnyWindow,                    // IsWindowHovered() only: Return true if any window is hovered
          hovered_flags_no_popup_hierarchy = ImGuiHoveredFlags_NoPopupHierarchy,            // IsWindowHovered() only: Do not consider popup hierarchy (do not treat popup emitter as parent of popup) (when used with _ChildWindows or _RootWindow)
          hovered_flags_dock_hierarchy = ImGuiHoveredFlags_DockHierarchy,                // IsWindowHovered() only: Consider docking hierarchy (treat dockspace host as parent of docked window) (when used with _ChildWindows or _RootWindow)
          hovered_flags_allow_when_blocked_by_popup = ImGuiHoveredFlags_AllowWhenBlockedByPopup,      // Return true even if a popup window is normally blocking access to this item/window
          //hovered_flags_allow_when_blocked_by_modal = ImGuiHoveredFlags_AllowWhenBlockedByModal,     // Return true even if a modal popup window is normally blocking access to this item/window. FIXME-TODO: Unavailable yet.
          hovered_flags_allow_when_blocked_by_active_item = ImGuiHoveredFlags_AllowWhenBlockedByActiveItem,  // Return true even if an active item is blocking access to this item/window. Useful for Drag and Drop patterns.
          hovered_flags_allow_when_overlapped_by_item = ImGuiHoveredFlags_AllowWhenOverlappedByItem, // IsItemHovered() only: Return true even if the item uses AllowOverlap mode and is overlapped by another hoverable item.
          hovered_flags_allow_when_overlapped_by_window = ImGuiHoveredFlags_AllowWhenOverlappedByWindow, // IsItemHovered() only: Return true even if the position is obstructed or overlapped by another window.
          hovered_flags_allow_when_disabled = ImGuiHoveredFlags_AllowWhenDisabled,             // IsItemHovered() only: Return true even if the item is disabled
          hovered_flags_no_nav_override = ImGuiHoveredFlags_NoNavOverride,                 // IsItemHovered() only: Disable using keyboard/gamepad navigation state when active, always query mouse
          hovered_flags_allow_when_overlapped = ImGuiHoveredFlags_AllowWhenOverlapped,           // Combined flag
          hovered_flags_rect_only = ImGuiHoveredFlags_RectOnly,                      // Combined flag
          hovered_flags_root_and_child_windows = ImGuiHoveredFlags_RootAndChildWindows,           // Combined flag

          // Tooltips mode
          hovered_flags_for_tooltip = ImGuiHoveredFlags_ForTooltip,                    // Shortcut for standard flags when using IsItemHovered() + SetTooltip() sequence.

          // (Advanced) Mouse Hovering delays.
          hovered_flags_stationary = ImGuiHoveredFlags_Stationary,                    // Require mouse to be stationary for style.HoverStationaryDelay (~0.15 sec) _at least one time_. After this, can move on same item/window. Using the stationary test tends to reduces the need for a long delay.
          hovered_flags_delay_none = ImGuiHoveredFlags_DelayNone,                     // IsItemHovered() only: Return true immediately (default). As this is the default you generally ignore this.
          hovered_flags_delay_short = ImGuiHoveredFlags_DelayShort,                    // IsItemHovered() only: Return true after style.HoverDelayShort elapsed (~0.15 sec) (shared between items) + requires mouse to be stationary for style.HoverStationaryDelay (once per item).
          hovered_flags_delay_normal = ImGuiHoveredFlags_DelayNormal,                   // IsItemHovered() only: Return true after style.HoverDelayNormal elapsed (~0.40 sec) (shared between items) + requires mouse to be stationary for style.HoverStationaryDelay (once per item).
          hovered_flags_no_shared_delay = ImGuiHoveredFlags_NoSharedDelay,                 // IsItemHovered() only: Disable shared delay system where moving from one item to the next keeps the previous timer for a short time (standard for tooltips with long delays)
        };



        bool is_item_hovered(hovered_flag_t flags = 0) {
          return ImGui::IsItemHovered(flags);
        }
        bool is_any_item_hovered() {
          return ImGui::IsAnyItemHovered();
        }
        bool is_any_item_active() {
          return ImGui::IsAnyItemActive();
        }

        using selectable_flag_t = int;
        enum {
          selectable_flags_none = ImGuiSelectableFlags_None,
          selectable_flags_no_auto_close_popups = ImGuiSelectableFlags_NoAutoClosePopups,  // Clicking this doesn't close parent popup window (overrides ImGuiItemFlags_AutoClosePopups).
          selectable_flags_span_all_columns = ImGuiSelectableFlags_SpanAllColumns,     // Frame will span all columns of its container table (text will still fit in current column).
          selectable_flags_allow_double_click = ImGuiSelectableFlags_AllowDoubleClick,   // Generate press events on double clicks too.
          selectable_flags_disabled = ImGuiSelectableFlags_Disabled,           // Cannot be selected, display grayed-out text.
          selectable_flags_allow_overlap = ImGuiSelectableFlags_AllowOverlap,       // (WIP) Hit testing to allow subsequent widgets to overlap this one.
          selectable_flags_highlight = ImGuiSelectableFlags_Highlight,          // Make the item be displayed as if it is hovered.
        };

        bool selectable(const std::string& label, bool selected = false, selectable_flag_t flags = 0, const fan::vec2& size = fan::vec2(0, 0)) {
          return ImGui::Selectable(label.c_str(), selected, flags, size);
        }
        bool selectable(const std::string& label, bool* p_selected, selectable_flag_t flags = 0, const fan::vec2& size = fan::vec2(0, 0)) {
          return ImGui::Selectable(label.c_str(), p_selected, flags, size);
        }

        using table_flags_t = int;
        enum {
          table_flags_none = ImGuiTableFlags_None,
          table_flags_resizable = ImGuiTableFlags_Resizable,                  // Enable resizing columns.
          table_flags_reorderable = ImGuiTableFlags_Reorderable,                // Enable reordering columns in header row (need calling TableSetupColumn() + TableHeadersRow() to display headers).
          table_flags_hideable = ImGuiTableFlags_Hideable,                   // Enable hiding/disabling columns in context menu.
          table_flags_sortable = ImGuiTableFlags_Sortable,                   // Enable sorting. Call TableGetSortSpecs() to obtain sort specs. Also see ImGuiTableFlags_SortMulti and ImGuiTableFlags_SortTristate.
          table_flags_no_saved_settings = ImGuiTableFlags_NoSavedSettings,            // Disable persisting columns order, width, and sort settings in the .ini file.
          table_flags_context_menu_in_body = ImGuiTableFlags_ContextMenuInBody,          // Right-click on columns body/contents will display table context menu. By default, it is available in TableHeadersRow().

          // Decorations
          table_flags_row_bg = ImGuiTableFlags_RowBg,                      // Set each RowBg color with ImGuiCol_TableRowBg or ImGuiCol_TableRowBgAlt (equivalent of calling TableSetBgColor with ImGuiTableBgFlags_RowBg0 on each row manually).
          table_flags_borders_inner_h = ImGuiTableFlags_BordersInnerH,              // Draw horizontal borders between rows.
          table_flags_borders_outer_h = ImGuiTableFlags_BordersOuterH,              // Draw horizontal borders at the top and bottom.
          table_flags_borders_inner_v = ImGuiTableFlags_BordersInnerV,              // Draw vertical borders between columns.
          table_flags_borders_outer_v = ImGuiTableFlags_BordersOuterV,              // Draw vertical borders on the left and right sides.
          table_flags_borders_h = ImGuiTableFlags_BordersH,                   // Draw horizontal borders.
          table_flags_borders_v = ImGuiTableFlags_BordersV,                   // Draw vertical borders.
          table_flags_borders_inner = ImGuiTableFlags_BordersInner,               // Draw inner borders.
          table_flags_borders_outer = ImGuiTableFlags_BordersOuter,               // Draw outer borders.
          table_flags_borders = ImGuiTableFlags_Borders,                    // Draw all borders.
          table_flags_no_borders_in_body = ImGuiTableFlags_NoBordersInBody,            // [ALPHA] Disable vertical borders in columns body (borders will always appear in headers). -> May move to style.
          table_flags_no_borders_in_body_until_resize = ImGuiTableFlags_NoBordersInBodyUntilResize,  // [ALPHA] Disable vertical borders in columns body until hovered for resize (borders will always appear in headers). -> May move to style.

          // Sizing Policy
          table_flags_sizing_fixed_fit = ImGuiTableFlags_SizingFixedFit,             // Columns default to _WidthFixed or _WidthAuto (if resizable or not resizable), matching contents width.
          table_flags_sizing_fixed_same = ImGuiTableFlags_SizingFixedSame,            // Columns default to _WidthFixed or _WidthAuto (if resizable or not resizable), matching the maximum contents width of all columns. Implicitly enable ImGuiTableFlags_NoKeepColumnsVisible.
          table_flags_sizing_stretch_prop = ImGuiTableFlags_SizingStretchProp,          // Columns default to _WidthStretch with default weights proportional to each column's contents widths.
          table_flags_sizing_stretch_same = ImGuiTableFlags_SizingStretchSame,          // Columns default to _WidthStretch with default weights all equal, unless overridden by TableSetupColumn().

          // Sizing Extra Options
          table_flags_no_host_extend_x = ImGuiTableFlags_NoHostExtendX,              // Make outer width auto-fit to columns, overriding outer_size.x value. Only available when ScrollX/ScrollY are disabled and stretch columns are not used.
          table_flags_no_host_extend_y = ImGuiTableFlags_NoHostExtendY,              // Make outer height stop exactly at outer_size.y (prevent auto-extending table past the limit). Only available when ScrollX/ScrollY are disabled. Data below the limit will be clipped and not visible.
          table_flags_no_keep_columns_visible = ImGuiTableFlags_NoKeepColumnsVisible,       // Disable keeping columns always minimally visible when ScrollX is off, and the table gets too small. Not recommended if columns are resizable.
          table_flags_precise_widths = ImGuiTableFlags_PreciseWidths,              // Disable distributing remainder width to stretched columns (width allocation on a 100-wide table with 3 columns: Without this flag: 33,33,34. With this flag: 33,33,33). With larger number of columns, resizing will appear less smooth.

          // Clipping
          table_flags_no_clip = ImGuiTableFlags_NoClip,                     // Disable clipping rectangle for every individual column (reduce draw command count, items will be able to overflow into other columns). Generally incompatible with TableSetupScrollFreeze().

          // Padding
          table_flags_pad_outer_x = ImGuiTableFlags_PadOuterX,                  // Default if BordersOuterV is on. Enable outermost padding. Generally desirable if you have headers.
          table_flags_no_pad_outer_x = ImGuiTableFlags_NoPadOuterX,                // Default if BordersOuterV is off. Disable outermost padding.
          table_flags_no_pad_inner_x = ImGuiTableFlags_NoPadInnerX,                // Disable inner padding between columns (double inner padding if BordersOuterV is on, single inner padding if BordersOuterV is off).

          // Scrolling
          table_flags_scroll_x = ImGuiTableFlags_ScrollX,                    // Enable horizontal scrolling. Require 'outer_size' parameter of BeginTable() to specify the container size. Changes default sizing policy. Because this creates a child window, ScrollY is currently generally recommended when using ScrollX.
          table_flags_scroll_y = ImGuiTableFlags_ScrollY,                    // Enable vertical scrolling. Require 'outer_size' parameter of BeginTable() to specify the container size.

          // Sorting
          table_flags_sort_multi = ImGuiTableFlags_SortMulti,                  // Hold shift when clicking headers to sort on multiple columns. TableGetSortSpecs() may return specs where (SpecsCount > 1).
          table_flags_sort_tristate = ImGuiTableFlags_SortTristate,               // Allow no sorting, disable default sorting. TableGetSortSpecs() may return specs where (SpecsCount == 0).

          // Miscellaneous
          table_flags_highlight_hovered_column = ImGuiTableFlags_HighlightHoveredColumn,     // Highlight column headers when hovered (may evolve into a fuller highlight).

        };

        using table_row_flags_t = int;
        // Flags for ImGui::TableNextRow()
        enum {
          table_row_flags_none = ImGuiTableRowFlags_None,
          table_row_flags_headers = ImGuiTableRowFlags_Headers,   // Identify header row (set default background color + width of its contents accounted differently for auto column width)
        };

        using table_column_flags_t = int;
        enum {
          table_column_flags_none = ImGuiTableColumnFlags_None,
          table_column_flags_disabled = ImGuiTableColumnFlags_Disabled,              // Overriding/master disable flag: hide column, won't show in context menu (unlike calling TableSetColumnEnabled() which manipulates the user accessible state).
          table_column_flags_default_hide = ImGuiTableColumnFlags_DefaultHide,           // Default as a hidden/disabled column.
          table_column_flags_default_sort = ImGuiTableColumnFlags_DefaultSort,           // Default as a sorting column.
          table_column_flags_width_stretch = ImGuiTableColumnFlags_WidthStretch,          // Column will stretch. Preferable with horizontal scrolling disabled (default if table sizing policy is _SizingStretchSame or _SizingStretchProp).
          table_column_flags_width_fixed = ImGuiTableColumnFlags_WidthFixed,            // Column will not stretch. Preferable with horizontal scrolling enabled (default if table sizing policy is _SizingFixedFit and table is resizable).
          table_column_flags_no_resize = ImGuiTableColumnFlags_NoResize,              // Disable manual resizing.
          table_column_flags_no_reorder = ImGuiTableColumnFlags_NoReorder,             // Disable manual reordering this column, this will also prevent other columns from crossing over this column.
          table_column_flags_no_hide = ImGuiTableColumnFlags_NoHide,                // Disable ability to hide/disable this column.
          table_column_flags_no_clip = ImGuiTableColumnFlags_NoClip,                // Disable clipping for this column (all NoClip columns will render in the same draw command).
          table_column_flags_no_sort = ImGuiTableColumnFlags_NoSort,                // Disable ability to sort on this field (even if ImGuiTableFlags_Sortable is set on the table).
          table_column_flags_no_sort_ascending = ImGuiTableColumnFlags_NoSortAscending,       // Disable ability to sort in the ascending direction.
          table_column_flags_no_sort_descending = ImGuiTableColumnFlags_NoSortDescending,      // Disable ability to sort in the descending direction.
          table_column_flags_no_header_label = ImGuiTableColumnFlags_NoHeaderLabel,         // TableHeadersRow() will submit an empty label for this column. Convenient for some small columns. Name will still appear in the context menu or in angled headers. You may append into this cell by calling TableSetColumnIndex() right after the TableHeadersRow() call.
          table_column_flags_no_header_width = ImGuiTableColumnFlags_NoHeaderWidth,         // Disable header text width contribution to automatic column width.
          table_column_flags_prefer_sort_ascending = ImGuiTableColumnFlags_PreferSortAscending,   // Make the initial sort direction Ascending when first sorting on this column (default).
          table_column_flags_prefer_sort_descending = ImGuiTableColumnFlags_PreferSortDescending,  // Make the initial sort direction Descending when first sorting on this column.
          table_column_flags_indent_enable = ImGuiTableColumnFlags_IndentEnable,          // Use current Indent value when entering cell (default for column 0).
          table_column_flags_indent_disable = ImGuiTableColumnFlags_IndentDisable,         // Ignore current Indent value when entering cell (default for columns > 0). Indentation changes _within_ the cell will still be honored.
          table_column_flags_angled_header = ImGuiTableColumnFlags_AngledHeader,          // TableHeadersRow() will submit an angled header row for this column. Note this will add an extra row.

          // Output status flags, read-only via TableGetColumnFlags()
          table_column_flags_is_enabled = ImGuiTableColumnFlags_IsEnabled,             // Status: is enabled == not hidden by user/api (referred to as "Hide" in _DefaultHide and _NoHide) flags.
          table_column_flags_is_visible = ImGuiTableColumnFlags_IsVisible,             // Status: is visible == is enabled AND not clipped by scrolling.
          table_column_flags_is_sorted = ImGuiTableColumnFlags_IsSorted,              // Status: is currently part of the sort specs.
          table_column_flags_is_hovered = ImGuiTableColumnFlags_IsHovered,             // Status: is hovered by mouse.

          // [Internal] Combinations and masks
          table_column_flags_width_mask = ImGuiTableColumnFlags_WidthMask_,            // WidthStretch | WidthFixed combination.
          table_column_flags_indent_mask = ImGuiTableColumnFlags_IndentMask_,           // IndentEnable | IndentDisable combination.
          table_column_flags_status_mask = ImGuiTableColumnFlags_StatusMask_,           // IsEnabled | IsVisible | IsSorted | IsHovered combination.

        };

        bool begin_table(const std::string& str_id, int columns, table_flags_t flags = 0, const fan::vec2& outer_size = fan::vec2(0.0f, 0.0f), f32_t inner_width = 0.0f) {
          return ImGui::BeginTable(str_id.c_str(), columns, flags, outer_size, inner_width);
        }

        void end_table() {
          ImGui::EndTable();
        }

        void table_next_row(table_row_flags_t row_flags = 0, f32_t min_row_height = 0.0f) {
          ImGui::TableNextRow(row_flags, min_row_height);
        }
        bool table_next_column() {
          return ImGui::TableNextColumn();
        }

        void columns(int count = 1, const char* id = nullptr, bool borders = true) {
          ImGui::Columns(count, id, borders);
        }

        void next_column() {
          ImGui::NextColumn();
        }
        using font_t = ImFont;

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
        void indent(f32_t indent_w = 0.0f) {
          ImGui::Indent(indent_w);
        }
        void unindent(f32_t indent_w = 0.0f) {
          ImGui::Indent(indent_w);
        }



        fan::vec2 calc_text_size(const std::string& text, const char* text_end = NULL, bool hide_text_after_double_hash = false, float wrap_width = -1.0f){
          return ImGui::CalcTextSize(text.c_str(), text_end, hide_text_after_double_hash, wrap_width);
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

        bool is_window_hovered() {
          return ImGui::IsWindowHovered();
        }


        fan::vec2 get_window_content_region_min() {
          ImVec2 min = ImGui::GetWindowContentRegionMin();
          return fan::vec2(min.x, min.y);
        }


        f32_t get_column_width(int index = -1) {
          return ImGui::GetColumnWidth(index);
        }

        bool is_item_active() {
          return ImGui::IsItemActive();
        }


        fan::vec2 get_mouse_drag_delta(int button = 0, float lock_threshold = -1.0f) {
          ImVec2 delta = ImGui::GetMouseDragDelta(button, lock_threshold);
          return fan::vec2(delta.x, delta.y);
        }

        void reset_mouse_drag_delta(int button = 0) {
          ImGui::ResetMouseDragDelta(button);
        }

        void set_scroll_x(float scroll_x) {
          ImGui::SetScrollX(scroll_x);
        }

        void set_scroll_y(float scroll_y) {
          ImGui::SetScrollY(scroll_y);
        }
        f32_t get_scroll_x() {
          return ImGui::GetScrollX();
        }
        float get_scroll_y() {
          return ImGui::GetScrollY();
        }


        /// <summary>
        /// RAII containers for gui windows.
        /// </summary>
        struct window_t {
          window_t(const std::string& window_name, bool* p_open = 0, window_flags_t window_flags = 0)
            : is_open(fan::graphics::gui::begin(window_name.c_str(), p_open, window_flags)) {}
          ~window_t() {
            fan::graphics::gui::end();
          }
          explicit operator bool() const {
            return is_open;
          }

        private:
          bool is_open;
        };
        /// <summary>
        /// RAII containers for gui child windows.
        /// </summary>
        struct child_window_t {
          child_window_t(const std::string& window_name, const fan::vec2& size = fan::vec2(0, 0), child_window_flags_t window_flags = 0)
            : is_open(ImGui::BeginChild(window_name.c_str(), size, window_flags)) {}
          ~child_window_t() {
            ImGui::EndChild();
          }
          explicit operator bool() const {
            return is_open;
          }

        private:
          bool is_open;
        };

        /// <summary>
        /// RAII containers for gui tables.
        /// </summary>
        struct table_t {
          table_t(const std::string& str_id, int columns, table_flags_t flags = 0, const fan::vec2& outer_size = fan::vec2(0.0f, 0.0f), f32_t inner_width = 0.0f)
            : is_open(ImGui::BeginTable(str_id.c_str(), columns, flags, outer_size, inner_width)) {}
          ~table_t() {
            ImGui::EndTable();
          }
          explicit operator bool() const {
            return is_open;
          }

        private:
          bool is_open;
        };

        bool button(const std::string& label, const fan::vec2& size = fan::vec2(0, 0)){
          return ImGui::Button(label.c_str(), size);
        }

        /// <summary>
        /// Draws the specified text, with its position influenced by other GUI elements.
        /// </summary>
        /// <param name="text">The text to draw.</param>
        /// <param name="color">The color of the text (defaults to white).</param>
        void text(const std::string& text, const fan::color& color = fan::colors::white) {
          ImGui::PushStyleColor(ImGuiCol_Text, color);
          ImGui::Text("%s", text.c_str());
          ImGui::PopStyleColor();
        }

        /// <summary>
        /// Draws the specified text at a given position on the screen.
        /// </summary>
        /// <param name="text">The text to draw.</param>
        /// <param name="position">The position of the text.</param>
        /// <param name="color">The color of the text (defaults to white).</param>
        void text_at(const std::string& text, const fan::vec2& position = 0, const fan::color& color = fan::colors::white) {
          ImGui::SetCursorPos(position);
          ImGui::PushStyleColor(ImGuiCol_Text, color);
          ImGui::Text("%s", text.c_str());
          ImGui::PopStyleColor();
        }

        void text_wrapped(const std::string& text, const fan::color& color = fan::colors::white) {
          ImGui::PushStyleColor(ImGuiCol_Text, color);
          ImGui::TextWrapped("%s", text.c_str());
          ImGui::PopStyleColor();
        }
        void text_unformatted(const std::string& text, const char* text_end = NULL) {
          ImGui::TextUnformatted(text.c_str(), text_end);
        }

        /// <summary>
        /// Draws text to bottom right.
        /// </summary>
        /// <param name="text">The text to draw.</param>
        /// <param name="color">The color of the text (defaults to white).</param>
        /// <param name="offset">Offset from the bottom-right corner.</param>
        void text_bottom_right(const std::string& text, const fan::color& color = fan::colors::white, const fan::vec2& offset = 0) {
          fan::vec2 text_pos;
          fan::vec2 text_size = ImGui::CalcTextSize(text.c_str());
          fan::vec2 window_pos = ImGui::GetWindowPos();
          fan::vec2 window_size = ImGui::GetWindowSize();

          text_pos.x = window_pos.x + window_size.x - text_size.x - ImGui::GetStyle().WindowPadding.x;
          text_pos.y = window_pos.y + window_size.y - text_size.y - ImGui::GetStyle().WindowPadding.y;
          fan::graphics::gui::text_at(text, text_pos + offset, color);
        }


        using slider_flags_t = int;
        enum {
          slider_flags_none = ImGuiSliderFlags_None,
          slider_flags_logarithmic = ImGuiSliderFlags_Logarithmic,       // Make the widget logarithmic (linear otherwise). Consider using ImGuiSliderFlags_NoRoundToFormat with this if using a format-string with small amount of digits.
          slider_flags_no_round_to_format = ImGuiSliderFlags_NoRoundToFormat,    // Disable rounding underlying value to match precision of the display format string (e.g. %.3f values are rounded to those 3 digits).
          slider_flags_no_input = ImGuiSliderFlags_NoInput,            // Disable CTRL+Click or Enter key allowing to input text directly into the widget.
          slider_flags_wrap_around = ImGuiSliderFlags_WrapAround,         // Enable wrapping around from max to min and from min to max. Only supported by DragXXX() functions for now.
          slider_flags_clamp_on_input = ImGuiSliderFlags_ClampOnInput,       // Clamp value to min/max bounds when input manually with CTRL+Click. By default CTRL+Click allows going out of bounds.
          slider_flags_clamp_zero_range = ImGuiSliderFlags_ClampZeroRange,     // Clamp even if min==max==0.0f. Otherwise due to legacy reason DragXXX functions don't clamp with those values. When your clamping limits are dynamic you almost always want to use it.
          slider_flags_no_speed_tweaks = ImGuiSliderFlags_NoSpeedTweaks,      // Disable keyboard modifiers altering tweak speed. Useful if you want to alter tweak speed yourself based on your own logic.
          slider_flags_always_clamp = ImGuiSliderFlags_AlwaysClamp,        // ClampOnInput | ClampZeroRange combination.
        };

        // todo make drag(), slider()

        bool drag_float(const std::string& label, f32_t* v, f32_t v_speed = 1.0f, f32_t v_min = 0.0f, f32_t v_max = 0.0f, const std::string& format = "%.3f", slider_flags_t flags = 0) {
          return ImGui::DragFloat(label.c_str(), v, v_speed, v_min, v_max, format.c_str(), flags);
        }
        bool drag_float(const std::string& label, fan::vec2* v, f32_t v_speed = 1.0f, f32_t v_min = 0.0f, f32_t v_max = 0.0f, const std::string& format = "%.3f", slider_flags_t flags = 0) {
          return ImGui::DragFloat2(label.c_str(), v->data(), v_speed, v_min, v_max, format.c_str(), flags);
        }
        bool drag_float(const std::string& label, fan::vec3* v, f32_t v_speed = 1.0f, f32_t v_min = 0.0f, f32_t v_max = 0.0f, const std::string& format = "%.3f", slider_flags_t flags = 0) {
          return ImGui::DragFloat3(label.c_str(), v->data(), v_speed, v_min, v_max, format.c_str(), flags);
        }
        bool drag_float(const std::string& label, fan::vec4* v, f32_t v_speed = 1.0f, f32_t v_min = 0.0f, f32_t v_max = 0.0f, const std::string& format = "%.3f", slider_flags_t flags = 0) {
          return ImGui::DragFloat4(label.c_str(), v->data(), v_speed, v_min, v_max, format.c_str(), flags);
        }
        bool drag_float(const std::string& label, fan::quat* q, f32_t v_speed = 1.0f, f32_t v_min = 0.0f, f32_t v_max = 0.0f, const std::string& format = "%.3f", slider_flags_t flags = 0) {
          return ImGui::DragFloat4(label.c_str(), q->data(), v_speed, v_min, v_max, format.c_str(), flags);
        }
        bool drag_float(const std::string& label, fan::color* c, f32_t v_speed = 1.0f, f32_t v_min = 0.0f, f32_t v_max = 0.0f, const std::string& format = "%.3f", slider_flags_t flags = 0) {
          return ImGui::DragFloat4(label.c_str(), c->data(), v_speed, v_min, v_max, format.c_str(), flags);
        }

        bool drag_int(const std::string& label, int* v, f32_t v_speed = 1.0f, int v_min = 0, int v_max = 0, const std::string& format = "%d", slider_flags_t flags = 0) {
          return ImGui::DragInt(label.c_str(), v, v_speed, v_min, v_max, format.c_str(), flags);
        }
        bool drag_int(const std::string& label, fan::vec2i* v, f32_t v_speed = 1.0f, int v_min = 0, int v_max = 0, const std::string& format = "%d", slider_flags_t flags = 0) {
          return ImGui::DragInt2(label.c_str(), v->data(), v_speed, v_min, v_max, format.c_str(), flags);
        }
        bool drag_int(const std::string& label, fan::vec3i* v, f32_t v_speed = 1.0f, int v_min = 0, int v_max = 0, const std::string& format = "%d", slider_flags_t flags = 0) {
          return ImGui::DragInt3(label.c_str(), v->data(), v_speed, v_min, v_max, format.c_str(), flags);
        }
        bool drag_int(const std::string& label, fan::vec4i* v, f32_t v_speed = 1.0f, int v_min = 0, int v_max = 0, const std::string& format = "%d", slider_flags_t flags = 0) {
          return ImGui::DragInt4(label.c_str(), v->data(), v_speed, v_min, v_max, format.c_str(), flags);
        }

        bool slider_int(const std::string& label, int* v, int v_min, int v_max, const std::string& format = "%d", slider_flags_t flags = 0) {
          return ImGui::SliderInt(label.c_str(), v, v_min, v_max, format.c_str(), flags);
        }

        bool slider_int(const std::string& label, fan::vec2i* v, int v_min, int v_max, const std::string& format = "%d", slider_flags_t flags = 0) {
          return ImGui::SliderInt2(label.c_str(), v->data(), v_min, v_max, format.c_str(), flags);
        }

        bool slider_int(const std::string& label, fan::vec3i* v, int v_min, int v_max, const std::string& format = "%d", slider_flags_t flags = 0) {
          return ImGui::SliderInt3(label.c_str(), v->data(), v_min, v_max, format.c_str(), flags);
        }

        bool slider_int(const std::string& label, fan::vec4i* v, int v_min, int v_max, const std::string& format = "%d", slider_flags_t flags = 0) {
          return ImGui::SliderInt4(label.c_str(), v->data(), v_min, v_max, format.c_str(), flags);
        }

        bool slider_float(const std::string& label, float* v, float v_min, float v_max, const std::string& format = "%.3f", slider_flags_t flags = 0) {
          return ImGui::SliderFloat(label.c_str(), v, v_min, v_max, format.c_str(), flags);
        }

        bool slider_float(const std::string& label, fan::vec2f* v, float v_min, float v_max, const std::string& format = "%.3f", slider_flags_t flags = 0) {
          return ImGui::SliderFloat2(label.c_str(), v->data(), v_min, v_max, format.c_str(), flags);
        }

        bool slider_float(const std::string& label, fan::vec3f* v, float v_min, float v_max, const std::string& format = "%.3f", slider_flags_t flags = 0) {
          return ImGui::SliderFloat3(label.c_str(), v->data(), v_min, v_max, format.c_str(), flags);
        }

        bool slider_float(const std::string& label, fan::vec4f* v, float v_min, float v_max, const std::string& format = "%.3f", slider_flags_t flags = 0) {
          return ImGui::SliderFloat4(label.c_str(), v->data(), v_min, v_max, format.c_str(), flags);
        }

        f32_t calc_item_width() {
          return ImGui::CalcItemWidth();
        }

        f32_t get_item_width() {
          return calc_item_width();
        }

        using input_text_flags_t = int;
        enum {
          input_text_flags_none = ImGuiInputTextFlags_None,
          input_text_flags_chars_decimal = ImGuiInputTextFlags_CharsDecimal,        // Allow 0123456789.+-*/
          input_text_flags_chars_hexadecimal = ImGuiInputTextFlags_CharsHexadecimal,    // Allow 0123456789ABCDEFabcdef
          input_text_flags_chars_scientific = ImGuiInputTextFlags_CharsScientific,     // Allow 0123456789.+-*/eE (Scientific notation input)
          input_text_flags_chars_uppercase = ImGuiInputTextFlags_CharsUppercase,      // Turn a..z into A..Z
          input_text_flags_chars_no_blank = ImGuiInputTextFlags_CharsNoBlank,        // Filter out spaces, tabs

          // Inputs
          input_text_flags_allow_tab_input = ImGuiInputTextFlags_AllowTabInput,       // Pressing TAB inputs a '\t' character into the text field
          input_text_flags_enter_returns_true = ImGuiInputTextFlags_EnterReturnsTrue,    // Return 'true' when Enter is pressed (as opposed to every time the value was modified). Consider using IsItemDeactivatedAfterEdit() instead!
          input_text_flags_escape_clears_all = ImGuiInputTextFlags_EscapeClearsAll,     // Escape key clears content if not empty, and deactivate otherwise (contrast to default behavior of Escape to revert)
          input_text_flags_ctrl_enter_for_new_line = ImGuiInputTextFlags_CtrlEnterForNewLine, // In multi-line mode, validate with Enter, add new line with Ctrl+Enter (default is opposite: validate with Ctrl+Enter, add line with Enter).

          // Other options
          input_text_flags_read_only = ImGuiInputTextFlags_ReadOnly,            // Read-only mode
          input_text_flags_password = ImGuiInputTextFlags_Password,            // Password mode, display all characters as '*', disable copy
          input_text_flags_always_overwrite = ImGuiInputTextFlags_AlwaysOverwrite,     // Overwrite mode
          input_text_flags_auto_select_all = ImGuiInputTextFlags_AutoSelectAll,       // Select entire text when first taking mouse focus
          input_text_flags_parse_empty_ref_val = ImGuiInputTextFlags_ParseEmptyRefVal,    // InputFloat(), InputInt(), InputScalar() etc. only: parse empty string as zero value.
          input_text_flags_display_empty_ref_val = ImGuiInputTextFlags_DisplayEmptyRefVal,  // InputFloat(), InputInt(), InputScalar() etc. only: when value is zero, do not display it. Generally used with ImGuiInputTextFlags_ParseEmptyRefVal.
          input_text_flags_no_horizontal_scroll = ImGuiInputTextFlags_NoHorizontalScroll,  // Disable following the cursor horizontally
          input_text_flags_no_undo_redo = ImGuiInputTextFlags_NoUndoRedo,          // Disable undo/redo. Note that input text owns the text data while active, if you want to provide your own undo/redo stack you need e.g. to call ClearActiveID().

          // Elide display / Alignment
          input_text_flags_elide_left = ImGuiInputTextFlags_ElideLeft,            // When text doesn't fit, elide left side to ensure right side stays visible. Useful for path/filenames. Single-line only!

          // Callback features
          input_text_flags_callback_completion = ImGuiInputTextFlags_CallbackCompletion,  // Callback on pressing TAB (for completion handling)
          input_text_flags_callback_history = ImGuiInputTextFlags_CallbackHistory,     // Callback on pressing Up/Down arrows (for history handling)
          input_text_flags_callback_always = ImGuiInputTextFlags_CallbackAlways,      // Callback on each iteration. User code may query cursor position, modify text buffer.
          input_text_flags_callback_char_filter = ImGuiInputTextFlags_CallbackCharFilter,  // Callback on character inputs to replace or discard them. Modify 'EventChar' to replace or discard, or return 1 in callback to discard.
          input_text_flags_callback_resize = ImGuiInputTextFlags_CallbackResize,      // Callback on buffer capacity changes request (beyond 'buf_size' parameter value), allowing the string to grow. Notify when the string wants to be resized (for string types which hold a cache of their Size). You will be provided a new BufSize in the callback and NEED to honor it. (see misc/cpp/imgui_stdlib.h for an example of using this)
          input_text_flags_callback_edit = ImGuiInputTextFlags_CallbackEdit,        // Callback on any edit. Note that InputText() already returns true on edit + you can always use IsItemEdited(). The callback is useful to manipulate the underlying buffer while focus is active.
        };

        using input_flags_t = int;
        enum {
          input_flags_none = ImGuiInputFlags_None,
          input_flags_repeat = ImGuiInputFlags_Repeat,                  // Enable repeat. Return true on successive repeats. Default for legacy IsKeyPressed(). NOT Default for legacy IsMouseClicked(). MUST BE == 1.

          // Flags for Shortcut(), SetNextItemShortcut()
          // - Routing policies: RouteGlobal+OverActive >> RouteActive or RouteFocused (if owner is active item) >> RouteGlobal+OverFocused >> RouteFocused (if in focused window stack) >> RouteGlobal.
          // - Default policy is RouteFocused. Can select only 1 policy among all available.
          input_flags_route_active = ImGuiInputFlags_RouteActive,             // Route to active item only.
          input_flags_route_focused = ImGuiInputFlags_RouteFocused,            // Route to windows in the focus stack (DEFAULT). Deep-most focused window takes inputs. Active item takes inputs over deep-most focused window.
          input_flags_route_global = ImGuiInputFlags_RouteGlobal,             // Global route (unless a focused window or active item registered the route).
          input_flags_route_always = ImGuiInputFlags_RouteAlways,             // Do not register route, poll keys directly.
          // - Routing options
          input_flags_route_over_focused = ImGuiInputFlags_RouteOverFocused,        // Option: global route: higher priority than focused route (unless active item in focused route).
          input_flags_route_over_active = ImGuiInputFlags_RouteOverActive,         // Option: global route: higher priority than active item. Unlikely you need to use that: will interfere with every active items, e.g. CTRL+A registered by InputText will be overridden by this. May not be fully honored as user/internal code is likely to always assume they can access keys when active.
          input_flags_route_unless_bg_focused = ImGuiInputFlags_RouteUnlessBgFocused,    // Option: global route: will not be applied if underlying background/void is focused (== no Dear ImGui windows are focused). Useful for overlay applications.
          input_flags_route_from_root_window = ImGuiInputFlags_RouteFromRootWindow,     // Option: route evaluated from the point of view of root window rather than current window.

          // Flags for SetNextItemShortcut()
          input_flags_tooltip = ImGuiInputFlags_Tooltip,                 // Automatically display a tooltip when hovering item [BETA] Unsure of right api (opt-in/opt-out)
        };

        using input_text_callback_t = ImGuiInputTextCallback;

        struct InputTextCallback_UserData {
          std::string* Str;
          ImGuiInputTextCallback  ChainCallback;
          void* ChainCallbackUserData;
        };

        //imgui_stdlib.cpp:
        inline int InputTextCallback(ImGuiInputTextCallbackData* data) {
          InputTextCallback_UserData* user_data = (InputTextCallback_UserData*)data->UserData;
          if (data->EventFlag == ImGuiInputTextFlags_CallbackResize)
          {
            // Resize string callback
            // If for some reason we refuse the new length (BufTextLen) and/or capacity (BufSize) we need to set them back to what we want.
            std::string* str = user_data->Str;
            IM_ASSERT(data->Buf == str->c_str());
            str->resize(data->BufTextLen);
            data->Buf = (char*)str->c_str();
          }
          else if (user_data->ChainCallback)
          {
            // Forward to user callback, if any
            data->UserData = user_data->ChainCallbackUserData;
            return user_data->ChainCallback(data);
          }
          return 0;
        }


        bool input_text(const std::string& label, std::string* buf, input_text_flags_t flags = 0, input_text_callback_t callback = nullptr, void* user_data = nullptr) {
          IM_ASSERT((flags & ImGuiInputTextFlags_CallbackResize) == 0);
          flags |= ImGuiInputTextFlags_CallbackResize;

          InputTextCallback_UserData cb_user_data;
          cb_user_data.Str = buf;
          cb_user_data.ChainCallback = callback;
          cb_user_data.ChainCallbackUserData = user_data;
          return ImGui::InputText(label.c_str(), (char*)buf->c_str(), buf->capacity() + 1, flags, InputTextCallback, &cb_user_data);
        }
        bool input_text_multiline(const std::string& label, std::string* buf, const ImVec2& size = ImVec2(0, 0), input_text_flags_t flags = 0, input_text_callback_t callback = nullptr, void* user_data = nullptr) {
          IM_ASSERT((flags & ImGuiInputTextFlags_CallbackResize) == 0);
          flags |= ImGuiInputTextFlags_CallbackResize;

          InputTextCallback_UserData cb_user_data;
          cb_user_data.Str = buf;
          cb_user_data.ChainCallback = callback;
          cb_user_data.ChainCallbackUserData = user_data;
          return ImGui::InputTextMultiline(label.c_str(), (char*)buf->c_str(), buf->capacity() + 1, size, flags, InputTextCallback, &cb_user_data);
        }

        bool input_float(const std::string& label, f32_t* v, f32_t step = 0.0f, f32_t step_fast = 0.0f, const char* format = "%.3f", input_text_flags_t flags = 0) {
          return ImGui::InputFloat(label.c_str(), v, step, step_fast, format, flags);
        }
        bool input_float(const std::string& label, fan::vec2* v, const char* format = "%.3f", input_text_flags_t flags = 0) {
          return ImGui::InputFloat2(label.c_str(), v->data(), format, flags);
        }
        bool input_float(const std::string& label, fan::vec3* v, const char* format = "%.3f", input_text_flags_t flags = 0) {
          return ImGui::InputFloat3(label.c_str(), v->data(), format, flags);
        }
        bool input_float(const std::string& label, fan::vec4* v, const char* format = "%.3f", input_text_flags_t flags = 0) {
          return ImGui::InputFloat4(label.c_str(), v->data(), format, flags);
        }
        bool input_int(const std::string& label, int* v, int step = 1, int step_fast = 100, input_text_flags_t flags = 0) {
          return ImGui::InputInt(label.c_str(), v, step, step_fast, flags);
        }
        bool input_int(const std::string& label, fan::vec2i* v, input_text_flags_t flags = 0) {
          return ImGui::InputInt2(label.c_str(), v->data(), flags);
        }
        bool input_int(const std::string& label, fan::vec3i* v, input_text_flags_t flags = 0) {
          return ImGui::InputInt3(label.c_str(), v->data(), flags);
        }
        bool input_int(const std::string& label, fan::vec4i* v, input_text_flags_t flags = 0) {
          return ImGui::InputInt4(label.c_str(), v->data(), flags);
        }

        using color_edit_flags_t = int;
        enum {
          color_edit_flags_none = ImGuiColorEditFlags_None,
          color_edit_flags_no_alpha = ImGuiColorEditFlags_NoAlpha,         // ColorEdit, ColorPicker, ColorButton: ignore Alpha component (will only read 3 components from the input pointer).
          color_edit_flags_no_picker = ImGuiColorEditFlags_NoPicker,        // ColorEdit: disable picker when clicking on color square.
          color_edit_flags_no_options = ImGuiColorEditFlags_NoOptions,       // ColorEdit: disable toggling options menu when right-clicking on inputs/small preview.
          color_edit_flags_no_small_preview = ImGuiColorEditFlags_NoSmallPreview,  // ColorEdit, ColorPicker: disable color square preview next to the inputs. (e.g. to show only the inputs)
          color_edit_flags_no_inputs = ImGuiColorEditFlags_NoInputs,        // ColorEdit, ColorPicker: disable inputs sliders/text widgets (e.g. to show only the small preview color square).
          color_edit_flags_no_tooltip = ImGuiColorEditFlags_NoTooltip,       // ColorEdit, ColorPicker, ColorButton: disable tooltip when hovering the preview.
          color_edit_flags_no_label = ImGuiColorEditFlags_NoLabel,         // ColorEdit, ColorPicker: disable display of inline text label (the label is still forwarded to the tooltip and picker).
          color_edit_flags_no_side_preview = ImGuiColorEditFlags_NoSidePreview,   // ColorPicker: disable bigger color preview on right side of the picker, use small color square preview instead.
          color_edit_flags_no_drag_drop = ImGuiColorEditFlags_NoDragDrop,      // ColorEdit: disable drag and drop target. ColorButton: disable drag and drop source.
          color_edit_flags_no_border = ImGuiColorEditFlags_NoBorder,        // ColorButton: disable border (which is enforced by default)

          // User Options (right-click on widget to change some of them).
          color_edit_flags_alpha_bar = ImGuiColorEditFlags_AlphaBar,        // ColorEdit, ColorPicker: show vertical alpha bar/gradient in picker.
          color_edit_flags_alpha_preview = ImGuiColorEditFlags_AlphaPreview,    // ColorEdit, ColorPicker, ColorButton: display preview as a transparent color over a checkerboard, instead of opaque.
          color_edit_flags_alpha_preview_half = ImGuiColorEditFlags_AlphaPreviewHalf, // ColorEdit, ColorPicker, ColorButton: display half opaque / half checkerboard, instead of opaque.
          color_edit_flags_hdr = ImGuiColorEditFlags_HDR,             // (WIP) ColorEdit: Currently only disable 0.0f..1.0f limits in RGBA edition (note: you probably want to use ImGuiColorEditFlags_Float flag as well).
          color_edit_flags_display_rgb = ImGuiColorEditFlags_DisplayRGB,      // [Display] ColorEdit: override _display_ type among RGB/HSV/Hex. ColorPicker: select any combination using one or more of RGB/HSV/Hex.
          color_edit_flags_display_hsv = ImGuiColorEditFlags_DisplayHSV,      // [Display] "
          color_edit_flags_display_hex = ImGuiColorEditFlags_DisplayHex,      // [Display] "
          color_edit_flags_uint8 = ImGuiColorEditFlags_Uint8,           // [DataType] ColorEdit, ColorPicker, ColorButton: _display_ values formatted as 0..255.
          color_edit_flags_float = ImGuiColorEditFlags_Float,           // [DataType] ColorEdit, ColorPicker, ColorButton: _display_ values formatted as 0.0f..1.0f floats instead of 0..255 integers. No round-trip of value via integers.
          color_edit_flags_picker_hue_bar = ImGuiColorEditFlags_PickerHueBar,    // [Picker] ColorPicker: bar for Hue, rectangle for Sat/Value.
          color_edit_flags_picker_hue_wheel = ImGuiColorEditFlags_PickerHueWheel,  // [Picker] ColorPicker: wheel for Hue, triangle for Sat/Value.
          color_edit_flags_input_rgb = ImGuiColorEditFlags_InputRGB,        // [Input] ColorEdit, ColorPicker: input and output data in RGB format.
          color_edit_flags_input_hsv = ImGuiColorEditFlags_InputHSV,        // [Input] ColorEdit, ColorPicker: input and output data in HSV format.

          // Defaults Options. You can set application defaults using SetColorEditOptions(). The intent is that you probably don't want to
          // override them in most of your calls. Let the user choose via the option menu and/or call SetColorEditOptions() once during startup.
          color_edit_flags_default_options = ImGuiColorEditFlags_DefaultOptions_, // Uint8 | DisplayRGB | InputRGB | PickerHueBar combination.
        };

        bool color_edit3(const std::string& label, fan::color* color, color_edit_flags_t flags = 0) {
          return ImGui::ColorEdit3(label.c_str(), color->data(), flags);
        }

        bool color_edit3(const std::string& label, fan::vec3* color, color_edit_flags_t flags= 0) {
          return ImGui::ColorEdit3(label.c_str(), color->data(), flags);
        }

        bool color_edit4(const std::string& label, fan::color* color, color_edit_flags_t flags = 0) {
          return ImGui::ColorEdit4(label.c_str(), color->data(), flags);
        }

        fan::vec2 get_window_pos() {
          return ImGui::GetWindowPos();
        }
        fan::vec2 get_window_size() {
          return ImGui::GetWindowSize();
        }

        void set_next_window_pos(const fan::vec2& position) {
          ImGui::SetNextWindowPos(position);
        }

        void set_next_window_size(const fan::vec2& size) {
          ImGui::SetNextWindowSize(size);
        }

        void set_window_font_scale(f32_t scale) {
          ImGui::SetWindowFontScale(scale);
        }

        bool is_mouse_dragging(int button = 0, float threshold = -1.0f) {
          return ImGui::IsMouseDragging(button, threshold);
        }


        using col_t = int;
        enum {
          col_text = ImGuiCol_Text,
          col_text_disabled = ImGuiCol_TextDisabled,
          col_window_bg = ImGuiCol_WindowBg,              // Background of normal windows
          col_child_bg = ImGuiCol_ChildBg,               // Background of child windows
          col_popup_bg = ImGuiCol_PopupBg,               // Background of popups, menus, tooltips windows
          col_border = ImGuiCol_Border,
          col_border_shadow = ImGuiCol_BorderShadow,
          col_frame_bg = ImGuiCol_FrameBg,               // Background of checkbox, radio button, plot, slider, text input
          col_frame_bg_hovered = ImGuiCol_FrameBgHovered,
          col_frame_bg_active = ImGuiCol_FrameBgActive,
          col_title_bg = ImGuiCol_TitleBg,               // Title bar
          col_title_bg_active = ImGuiCol_TitleBgActive,         // Title bar when focused
          col_title_bg_collapsed = ImGuiCol_TitleBgCollapsed,      // Title bar when collapsed
          col_menu_bar_bg = ImGuiCol_MenuBarBg,
          col_scrollbar_bg = ImGuiCol_ScrollbarBg,
          col_scrollbar_grab = ImGuiCol_ScrollbarGrab,
          col_scrollbar_grab_hovered = ImGuiCol_ScrollbarGrabHovered,
          col_scrollbar_grab_active = ImGuiCol_ScrollbarGrabActive,
          col_check_mark = ImGuiCol_CheckMark,             // Checkbox tick and RadioButton circle
          col_slider_grab = ImGuiCol_SliderGrab,
          col_slider_grab_active = ImGuiCol_SliderGrabActive,
          col_button = ImGuiCol_Button,
          col_button_hovered = ImGuiCol_ButtonHovered,
          col_button_active = ImGuiCol_ButtonActive,
          col_header = ImGuiCol_Header,                // Header* colors are used for CollapsingHeader, TreeNode, Selectable, MenuItem
          col_header_hovered = ImGuiCol_HeaderHovered,
          col_header_active = ImGuiCol_HeaderActive,
          col_separator = ImGuiCol_Separator,
          col_separator_hovered = ImGuiCol_SeparatorHovered,
          col_separator_active = ImGuiCol_SeparatorActive,
          col_resize_grip = ImGuiCol_ResizeGrip,            // Resize grip in lower-right and lower-left corners of windows.
          col_resize_grip_hovered = ImGuiCol_ResizeGripHovered,
          col_resize_grip_active = ImGuiCol_ResizeGripActive,
          col_tab_hovered = ImGuiCol_TabHovered,            // Tab background, when hovered
          col_tab = ImGuiCol_Tab,                   // Tab background, when tab-bar is focused & tab is unselected
          col_tab_selected = ImGuiCol_TabSelected,           // Tab background, when tab-bar is focused & tab is selected
          col_tab_selected_overline = ImGuiCol_TabSelectedOverline,   // Tab horizontal overline, when tab-bar is focused & tab is selected
          col_tab_dimmed = ImGuiCol_TabDimmed,             // Tab background, when tab-bar is unfocused & tab is unselected
          col_tab_dimmed_selected = ImGuiCol_TabDimmedSelected,     // Tab background, when tab-bar is unfocused & tab is selected
          col_tab_dimmed_selected_overline = ImGuiCol_TabDimmedSelectedOverline, //..horizontal overline, when tab-bar is unfocused & tab is selected
          col_docking_preview = ImGuiCol_DockingPreview,        // Preview overlay color when about to docking something
          col_docking_empty_bg = ImGuiCol_DockingEmptyBg,        // Background color for empty node (e.g. CentralNode with no window docked into it)
          col_plot_lines = ImGuiCol_PlotLines,
          col_plot_lines_hovered = ImGuiCol_PlotLinesHovered,
          col_plot_histogram = ImGuiCol_PlotHistogram,
          col_plot_histogram_hovered = ImGuiCol_PlotHistogramHovered,
          col_table_header_bg = ImGuiCol_TableHeaderBg,         // Table header background
          col_table_border_strong = ImGuiCol_TableBorderStrong,     // Table outer and header borders (prefer using Alpha=1.0 here)
          col_table_border_light = ImGuiCol_TableBorderLight,      // Table inner borders (prefer using Alpha=1.0 here)
          col_table_row_bg = ImGuiCol_TableRowBg,            // Table row background (even rows)
          col_table_row_bg_alt = ImGuiCol_TableRowBgAlt,         // Table row background (odd rows)
          col_text_link = ImGuiCol_TextLink,              // Hyperlink color
          col_text_selected_bg = ImGuiCol_TextSelectedBg,
          col_drag_drop_target = ImGuiCol_DragDropTarget,        // Rectangle highlighting a drop target
          col_nav_cursor = ImGuiCol_NavCursor,             // Color of keyboard/gamepad navigation cursor/rectangle, when visible
          col_nav_windowing_highlight = ImGuiCol_NavWindowingHighlight, // Highlight window when using CTRL+TAB
          col_nav_windowing_dim_bg = ImGuiCol_NavWindowingDimBg,     // Darken/colorize entire screen behind the CTRL+TAB window list, when active
          col_modal_window_dim_bg = ImGuiCol_ModalWindowDimBg,      // Darken/colorize entire screen behind a modal window, when one is active
          col_count = ImGuiCol_COUNT,
        };

        using style_t = ImGuiStyle;
        style_t& get_style() {
          return ImGui::GetStyle();
        }


        using io_t = ImGuiIO;
        io_t& get_io() {
          return ImGui::GetIO();
        }

        using style_var_t = int;
        enum {
          style_var_alpha = ImGuiStyleVar_Alpha,                    // float     Alpha
          style_var_disabled_alpha = ImGuiStyleVar_DisabledAlpha,            // float     DisabledAlpha
          style_var_window_padding = ImGuiStyleVar_WindowPadding,            // ImVec2    WindowPadding
          style_var_window_rounding = ImGuiStyleVar_WindowRounding,           // float     WindowRounding
          style_var_window_border_size = ImGuiStyleVar_WindowBorderSize,         // float     WindowBorderSize
          style_var_window_min_size = ImGuiStyleVar_WindowMinSize,            // ImVec2    WindowMinSize
          style_var_window_title_align = ImGuiStyleVar_WindowTitleAlign,         // ImVec2    WindowTitleAlign
          style_var_child_rounding = ImGuiStyleVar_ChildRounding,            // float     ChildRounding
          style_var_child_border_size = ImGuiStyleVar_ChildBorderSize,          // float     ChildBorderSize
          style_var_popup_rounding = ImGuiStyleVar_PopupRounding,            // float     PopupRounding
          style_var_popup_border_size = ImGuiStyleVar_PopupBorderSize,          // float     PopupBorderSize
          style_var_frame_padding = ImGuiStyleVar_FramePadding,             // ImVec2    FramePadding
          style_var_frame_rounding = ImGuiStyleVar_FrameRounding,            // float     FrameRounding
          style_var_frame_border_size = ImGuiStyleVar_FrameBorderSize,          // float     FrameBorderSize
          style_var_item_spacing = ImGuiStyleVar_ItemSpacing,              // ImVec2    ItemSpacing
          style_var_item_inner_spacing = ImGuiStyleVar_ItemInnerSpacing,         // ImVec2    ItemInnerSpacing
          style_var_indent_spacing = ImGuiStyleVar_IndentSpacing,            // float     IndentSpacing
          style_var_cell_padding = ImGuiStyleVar_CellPadding,              // ImVec2    CellPadding
          style_var_scrollbar_size = ImGuiStyleVar_ScrollbarSize,            // float     ScrollbarSize
          style_var_scrollbar_rounding = ImGuiStyleVar_ScrollbarRounding,        // float     ScrollbarRounding
          style_var_grab_min_size = ImGuiStyleVar_GrabMinSize,              // float     GrabMinSize
          style_var_grab_rounding = ImGuiStyleVar_GrabRounding,             // float     GrabRounding
          style_var_tab_rounding = ImGuiStyleVar_TabRounding,              // float     TabRounding
          style_var_tab_border_size = ImGuiStyleVar_TabBorderSize,          // float     TabBorderSize
          style_var_tab_bar_border_size = ImGuiStyleVar_TabBarBorderSize,         // float     TabBarBorderSize
          style_var_tab_bar_overline_size = ImGuiStyleVar_TabBarOverlineSize,       // float     TabBarOverlineSize
          style_var_table_angled_headers_angle = ImGuiStyleVar_TableAngledHeadersAngle,  // float     TableAngledHeadersAngle
          style_var_table_angled_headers_text_align = ImGuiStyleVar_TableAngledHeadersTextAlign,// ImVec2  TableAngledHeadersTextAlign
          style_var_button_text_align = ImGuiStyleVar_ButtonTextAlign,          // ImVec2    ButtonTextAlign
          style_var_selectable_text_align = ImGuiStyleVar_SelectableTextAlign,      // ImVec2    SelectableTextAlign
          style_var_separator_text_border_size = ImGuiStyleVar_SeparatorTextBorderSize,  // float     SeparatorTextBorderSize
          style_var_separator_text_align = ImGuiStyleVar_SeparatorTextAlign,       // ImVec2    SeparatorTextAlign
          style_var_separator_text_padding = ImGuiStyleVar_SeparatorTextPadding,     // ImVec2    SeparatorTextPadding
          style_var_docking_separator_size = ImGuiStyleVar_DockingSeparatorSize,     // float     DockingSeparatorSize
          style_var_count = ImGuiStyleVar_COUNT
        };

        void push_style_color(col_t index, const fan::color& col) {
          ImGui::PushStyleColor(index, col);
        }

        void pop_style_color(int n = 1) {
          ImGui::PopStyleColor(n);
        }

        void push_style_var(style_var_t index, f32_t val) {
          ImGui::PushStyleVar(index, val);
        }

        void push_style_var(style_var_t index, const fan::vec2& val) {
          ImGui::PushStyleVar(index, val);
        }

        void pop_style_var(int n = 1) {
          ImGui::PopStyleVar(n);
        }

        using draw_list_t = ImDrawList;
        draw_list_t* get_window_draw_list() {
          return ImGui::GetWindowDrawList();
        }
        draw_list_t* get_background_draw_list() {
          return ImGui::GetBackgroundDrawList();
        }

        #if !defined(__INTELLISENSE__)
        #define fan_imgui_dragfloat_named(name, variable, speed, m_min, m_max) \
          fan::graphics::gui::drag_float(name, &variable, speed, m_min, m_max)
        #endif

        #define fan_imgui_dragfloat(variable, speed, m_min, m_max) \
            fan_imgui_dragfloat_named(STRINGIFY(variable), variable, speed, m_min, m_max)


        #define fan_imgui_dragfloat1(variable, speed) \
            fan_imgui_dragfloat_named(STRINGIFY(variable), variable, speed, 0, 0)

        using gui_draw_cb_nr_t = loco_t::gui_draw_cb_NodeReference_t;

        struct imgui_element_nr_t : gui_draw_cb_nr_t {
          using base_t = gui_draw_cb_nr_t;

          imgui_element_nr_t() = default;

          imgui_element_nr_t(const imgui_element_nr_t& nr) : imgui_element_nr_t() {
            if (nr.is_invalid()) {
              return;
            }
            init();
          }

          imgui_element_nr_t(imgui_element_nr_t&& nr) {
            NRI = nr.NRI;
            nr.invalidate_soft();
          }

          ~imgui_element_nr_t() {
            invalidate();
          }

          fan::graphics::gui::imgui_element_nr_t& operator=(const imgui_element_nr_t& id) {
            if (!is_invalid()) {
              invalidate();
            }
            if (id.is_invalid()) {
              return *this;
            }

            if (this != &id) {
              init();
            }
            return *this;
          }

          fan::graphics::gui::imgui_element_nr_t& operator=(imgui_element_nr_t&& id) {
            if (!is_invalid()) {
              invalidate();
            }
            if (id.is_invalid()) {
              return *this;
            }

            if (this != &id) {
              if (!is_invalid()) {
                invalidate();
              }
              NRI = id.NRI;

              id.invalidate_soft();
            }
            return *this;
          }
          void init() {
            *(base_t*)this = gloco->gui_draw_cb.NewNodeLast();
          }

          bool is_invalid() const {
            return loco_t::gui_draw_cb_inric(*this);
          }

          void invalidate_soft() {
            *(base_t*)this = gloco->gui_draw_cb.gnric();
          }

          void invalidate() {
            if (is_invalid()) {
              return;
            }
            gloco->gui_draw_cb.unlrec(*this);
            *(base_t*)this = gloco->gui_draw_cb.gnric();
          }

          void set(const auto& lambda) {
            gloco->gui_draw_cb[*this] = lambda;
          }
        };

        struct imgui_element_t : imgui_element_nr_t {
          imgui_element_t() = default;
          imgui_element_t(const auto& lambda) {
            imgui_element_nr_t::init();
            imgui_element_nr_t::set(lambda);
          }
        };


        inline const char* item_getter1(const std::vector<std::string>& items, int index) {
          if (index >= 0 && index < (int)items.size()) {
            return items[index].c_str();
          }
          return "N/A";
        }

        void set_viewport(fan::graphics::viewport_t viewport) {
          ImVec2 mainViewportPos = ImGui::GetMainViewport()->Pos;

          ImVec2 windowPos = ImGui::GetWindowPos();

          fan::vec2 windowPosRelativeToMainViewport;
          windowPosRelativeToMainViewport.x = windowPos.x - mainViewportPos.x;
          windowPosRelativeToMainViewport.y = windowPos.y - mainViewportPos.y;

          fan::vec2 window_size = gloco->window.get_size();
          fan::vec2 viewport_size = ImGui::GetContentRegionAvail();

          ImVec2 padding = ImGui::GetStyle().WindowPadding;
          viewport_size.x += padding.x * 2;
          viewport_size.y += padding.y * 2;

          fan::vec2 viewport_pos = fan::vec2(windowPosRelativeToMainViewport);
          gloco->viewport_set(
            viewport,
            viewport_pos,
            viewport_size,
            window_size
          );
        }

        using window_handle_t = ImGuiWindow;
        window_handle_t* get_current_window() {
          return ImGui::GetCurrentWindow();
        }

        enum {
          config_flags_none = ImGuiConfigFlags_None,
          config_flags_nav_enable_keyboard = ImGuiConfigFlags_NavEnableKeyboard,
          config_flags_nav_enable_gamepad = ImGuiConfigFlags_NavEnableGamepad,
          config_flags_no_mouse = ImGuiConfigFlags_NoMouse,
          config_flags_no_mouse_cursor_change = ImGuiConfigFlags_NoMouseCursorChange,
          config_flags_no_keyboard = ImGuiConfigFlags_NoKeyboard,
          config_flags_docking_enable = ImGuiConfigFlags_DockingEnable,
          config_flags_viewports_enable = ImGuiConfigFlags_ViewportsEnable,
          config_flags_dpi_enable_scale_viewports = ImGuiConfigFlags_DpiEnableScaleViewports,
          config_flags_dpi_enable_scale_fonts = ImGuiConfigFlags_DpiEnableScaleFonts,
          config_flags_is_srgb = ImGuiConfigFlags_IsSRGB,
          config_flags_is_touch_screen = ImGuiConfigFlags_IsTouchScreen,

        };

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
          fan::audio::piece_t piece_hover = {0}, 
          fan::audio::piece_t piece_click = {0}, 
          const fan::vec2& size = fan::vec2(0, 0)
        ) {
          ImGui::PushID(label.c_str());
          ImGuiStorage* storage = ImGui::GetStateStorage();
          ImGuiID id = ImGui::GetID("audio_button_prev_hovered");
          bool previously_hovered = storage->GetBool(id);

          bool pressed = ImGui::Button(label.c_str(), size);
          bool currently_hovered = ImGui::IsItemHovered();

          if (currently_hovered && !previously_hovered) {
            fan::audio::piece_t& piece = fan::audio::is_piece_valid(piece_hover) ? piece_hover : gloco->piece_hover;
            fan::audio::play(piece);
          }
          if (pressed) {
            fan::audio::piece_t& piece = fan::audio::is_piece_valid(piece_click) ? piece_click : gloco->piece_click;
            fan::audio::play(piece);
          }
          storage->SetBool(id, currently_hovered);

          ImGui::PopID();
          return pressed;
        }

        bool combo(const std::string& label, int* current_item, const char* const items[], int items_count, int popup_max_height_in_items = -1) {
          return ImGui::Combo(label.c_str(), current_item, items, items_count, popup_max_height_in_items);
        }
        // Separate items with \0 within a string, end item-list with \0\0. e.g. "One\0Two\0Three\0"
        bool combo(const std::string& label, int* current_item, const char* items_separated_by_zeros, int popup_max_height_in_items = -1){
          return ImGui::Combo(label.c_str(), current_item, items_separated_by_zeros, popup_max_height_in_items);
        }
        bool combo(const std::string& label, int* current_item, const char* (*getter)(void* user_data, int idx), void* user_data, int items_count, int popup_max_height_in_items = -1) {
          return ImGui::Combo(label.c_str(), current_item, getter, user_data, items_count, popup_max_height_in_items);
        }
        bool checkbox(const std::string& label, bool* v) {
          return ImGui::Checkbox(label.c_str(), v);
        }
        bool list_box(const std::string &label, int* current_item, bool (*old_callback)(void* user_data, int idx, const char** out_text), void* user_data, int items_count, int height_in_items = -1) {
          return ImGui::ListBox(label.c_str(), current_item, old_callback, user_data, items_count, height_in_items);
        }
        bool list_box(const std::string& label, int* current_item, const char* const items[], int items_count, int height_in_items = -1) {
          return ImGui::ListBox(label.c_str(), current_item, items, items_count, height_in_items);
        }
        bool list_box(const std::string& label, int* current_item, const char* (*getter)(void* user_data, int idx), void* user_data, int items_count, int height_in_items = -1) {
          return ImGui::ListBox(label.c_str(), current_item, getter, user_data, items_count, height_in_items);
        }

        void image(loco_t::image_t img, const fan::vec2& size, const fan::vec2& uv0 = fan::vec2(0, 0), const fan::vec2& uv1 = fan::vec2(1, 1), const fan::color& tint_col = fan::color(1, 1, 1, 1), const fan::color& border_col = fan::color(0, 0, 0, 0)) {
          ImGui::Image((ImTextureID)gloco->image_get_handle(img), size, uv0, uv1, tint_col, border_col);
        }
        bool image_button(const std::string& str_id, loco_t::image_t img, const fan::vec2& size, const fan::vec2& uv0 = fan::vec2(0, 0), const fan::vec2& uv1 = fan::vec2(1, 1), int frame_padding = -1, const fan::color& bg_col = fan::color(0, 0, 0, 0), const fan::color& tint_col = fan::color(1, 1, 1, 1)) {
          return ImGui::ImageButton(str_id.c_str(), (ImTextureID)gloco->image_get_handle(img), size, uv0, uv1, bg_col, tint_col);
        }
        bool image_text_button(
          loco_t::image_t img,
          const std::string& text,
          const fan::color& color,
          const fan::vec2& size,
          const fan::vec2& uv0 = fan::vec2(0, 0),
          const fan::vec2& uv1 = fan::vec2(1, 1),
          int frame_padding = -1,
          const fan::color& bg_col = fan::color(0, 0, 0, 0),
          const fan::color& tint_col = fan::color(1, 1, 1, 1)
        ) {
          bool ret = ImGui::ImageButton(text.c_str(), (ImTextureID)gloco->image_get_handle(img), size, uv0, uv1, bg_col, tint_col);
          ImVec2 text_size = ImGui::CalcTextSize(text.c_str());
          ImVec2 pos = ImGui::GetItemRectMin();
          pos.x += (size.x - text_size.x) * 0.5f;
          pos.y += (size.y - text_size.y) * 0.5f;
          ImGui::GetWindowDrawList()->AddText(pos, ImGui::GetColorU32(color), text.c_str());
          return ret;
        }

        bool toggle_button(const std::string& str, bool* v) {
          ImGui::Text("%s", str.c_str());
          ImGui::SameLine();

          ImVec2 p = ImGui::GetCursorScreenPos();
          ImDrawList* draw_list = ImGui::GetWindowDrawList();

          float height = ImGui::GetFrameHeight();
          float width = height * 1.55f;
          float radius = height * 0.50f;

          bool changed = ImGui::InvisibleButton(("##" + str).c_str(), ImVec2(width, height));
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
        bool toggle_image_button(const std::string& char_id, loco_t::image_t image, const fan::vec2& size, bool* toggle) {
          bool clicked = false;

          ImVec4 tintColor = ImVec4(1, 1, 1, 1);
          if (*toggle) {
            tintColor = ImVec4(0.3f, 0.3f, 0.3f, 1.0f);
          }

          if (fan::graphics::gui::image_button(char_id, image, size, ImVec2(0, 0), ImVec2(1, 1), -1, ImVec4(0, 0, 0, 0), tintColor)) {
            *toggle = !(*toggle);
            clicked = true;
          }

          return clicked;
        }

        void text_bottom_right(const char* text, uint32_t reverse_yoffset = 0) {
          ImDrawList* draw_list = ImGui::GetWindowDrawList();

          ImVec2 window_pos = ImGui::GetWindowPos();
          ImVec2 window_size = ImGui::GetWindowSize();

          ImVec2 text_size = ImGui::CalcTextSize(text);

          ImVec2 text_pos;
          text_pos.x = window_pos.x + window_size.x - text_size.x - ImGui::GetStyle().WindowPadding.x;
          text_pos.y = window_pos.y + window_size.y - text_size.y - ImGui::GetStyle().WindowPadding.y;

          text_pos.y -= reverse_yoffset * ImGui::GetTextLineHeightWithSpacing();

          draw_list->AddText(text_pos, IM_COL32(255, 255, 255, 255), text);
        }


        template <std::size_t N>
        bool toggle_image_button(const std::array<loco_t::image_t, N>& images, const fan::vec2& size, int* selectedIndex)
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


        fan::vec2 get_position_bottom_corner(const std::string& text = "", uint32_t reverse_yoffset = 0) {
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
        void image_rotated(
          loco_t::image_t image,
          const fan::vec2& size,
          int angle,
          const fan::vec2& uv0 = fan::vec2(0, 0),
          const fan::vec2& uv1 = fan::vec2(1, 1),
          const fan::color& tint_col = fan::color(1, 1, 1, 1),
          const fan::color& border_col = fan::color(0, 0, 0, 0)
        ) {
          IM_ASSERT(angle % 90 == 0);
          fan::vec2 _uv0, _uv1, _uv2, _uv3;

          switch (angle % 360) {
          case 0:
            fan::graphics::gui::image(image, size, uv0, uv1, tint_col, border_col);
            return;
          case 180:
            fan::graphics::gui::image(image, size, uv1, uv0, tint_col, border_col);
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
          if (window->SkipItems)
            return;

          fan::vec2 _size(size.y, size.x); // swapped for rotation
          fan::vec2 cursor_pos = *(fan::vec2*)&window->DC.CursorPos;
          fan::vec2 bb_max = cursor_pos + _size;
          if (border_col.a > 0.0f) {
            bb_max += fan::vec2(2, 2);
          }

          ImRect bb(*(ImVec2*)&cursor_pos, *(ImVec2*)&bb_max);
          ImGui::ItemSize(bb);
          if (!ImGui::ItemAdd(bb, 0))
            return;

          if (border_col.a > 0.0f) {
            window->DrawList->AddRect(*(ImVec2*)&bb.Min, *(ImVec2*)&bb.Max, ImGui::GetColorU32(border_col), 0.0f);
            fan::vec2 x0 = cursor_pos + fan::vec2(1, 1);
            fan::vec2 x2 = bb_max - fan::vec2(1, 1);
            fan::vec2 x1 = fan::vec2(x2.x, x0.y);
            fan::vec2 x3 = fan::vec2(x0.x, x2.y);

            window->DrawList->AddImageQuad(
              (ImTextureID)gloco->image_get_handle(image),
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
              (ImTextureID)gloco->image_get_handle(image),
              *(ImVec2*)&x0, *(ImVec2*)&x1, *(ImVec2*)&x2, *(ImVec2*)&x3,
              *(ImVec2*)&_uv0, *(ImVec2*)&_uv1, *(ImVec2*)&_uv2, *(ImVec2*)&_uv3,
              ImGui::GetColorU32(tint_col)
            );
          }
        }
        void send_drag_drop_item(const std::string& id, const std::wstring& path, const std::string& popup = "") {
          std::string popup_ = popup;
          if (popup.empty()) {
            popup_ = { path.begin(), path.end() };
          }
          if (ImGui::BeginDragDropSource()) {
            ImGui::SetDragDropPayload(id.c_str(), path.data(), (path.size() + 1) * sizeof(wchar_t));
            ImGui::Text("%s", popup_.c_str());
            ImGui::EndDragDropSource();
          }
        }
        void receive_drag_drop_target(const std::string& id, auto receive_func, bool use_absolute_path = true) {
          if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(id.c_str())) {
              const wchar_t* path = (const wchar_t*)payload->Data;
              std::wstring wstr = path;
              receive_func(use_absolute_path ? std::filesystem::absolute(path).string() : std::string(wstr.begin(), wstr.end()));
            }
            ImGui::EndDragDropTarget();
          }
        }

      }// loco gui

  #endif//loco_gui
  }
}

export namespace fan {
  namespace graphics {
    namespace gui {
      struct imgui_fs_var_t {
        fan::graphics::gui::imgui_element_t ie;

        imgui_fs_var_t() = default;

        template <typename T>
        imgui_fs_var_t(
          loco_t::shader_t shader_nr,
          const std::string& var_name,
          T initial_ = 0,
          f32_t speed = 1,
          f32_t min = -100000,
          f32_t max = 100000
        ) {
          //fan::vec_wrap_t < sizeof(T) / fan::conditional_value_t < std::is_class_v<T>, sizeof(T{} [0] ), sizeof(T) > , f32_t > initial = initial_;
          fan::vec_wrap_t<fan::conditional_value_t<std::is_arithmetic_v<T>, 1, sizeof(T) / sizeof(f32_t)>::value, f32_t>
            initial;
          if constexpr (std::is_arithmetic_v<T>) {
            initial = (f32_t)initial_;
          }
          else {
            initial = initial_;
          }
          fan::opengl::context_t::shader_t shader = gloco->shader_get(shader_nr).gl;
          if (gloco->window.renderer == loco_t::renderer_t::vulkan) {
            fan::throw_error("");
          }
          auto found = gloco->shader_list[shader_nr].uniform_type_table.find(var_name);
          if (found == gloco->shader_list[shader_nr].uniform_type_table.end()) {
            //fan::print("failed to set uniform value");
            return;
            //fan::throw_error("failed to set uniform value");
          }
          ie = [str = found->second, shader_nr, var_name, speed, min, max, data = initial]() mutable {
            bool modify = false;
            switch (fan::get_hash(str)) {
            case fan::get_hash(std::string_view("float")): {
              modify = ImGui::DragFloat(std::string(std::move(var_name)).c_str(), &data[0], (f32_t)speed, (f32_t)min, (f32_t)max);
              break;
            }
            case fan::get_hash(std::string_view("vec2")): {
              modify = ImGui::DragFloat2(std::string(std::move(var_name)).c_str(), ((fan::vec2*)&data)->data(), (f32_t)speed, (f32_t)min, (f32_t)max);
              break;
            }
            case fan::get_hash(std::string_view("vec3")): {
              modify = ImGui::DragFloat3(std::string(std::move(var_name)).c_str(), ((fan::vec3*)&data)->data(), (f32_t)speed, (f32_t)min, (f32_t)max);
              break;
            }
            case fan::get_hash(std::string_view("vec4")): {
              modify = ImGui::DragFloat4(std::string(std::move(var_name)).c_str(), ((fan::vec4*)&data)->data(), (f32_t)speed, (f32_t)min, (f32_t)max);
              break;
            }
            }
            if (modify) {
              gloco->shader_set_value(shader_nr, var_name, data);
            }
            };
          gloco->shader_set_value(shader_nr, var_name, initial);
        }
      };
    }
  }
}

export namespace fan {
  namespace graphics {
    namespace gui {
      struct content_browser_t {
        struct file_info_t {
          std::string filename;
          std::filesystem::path some_path; //?
          std::wstring item_path;
          bool is_directory;
          loco_t::image_t preview_image;
          //std::string 
        };

        std::vector<file_info_t> directory_cache;

        loco_t::image_t icon_arrow_left = gloco->image_load("images_content_browser/arrow_left.webp");
        loco_t::image_t icon_arrow_right = gloco->image_load("images_content_browser/arrow_right.webp");

        loco_t::image_t icon_file = gloco->image_load("images_content_browser/file.webp");
        loco_t::image_t icon_directory = gloco->image_load("images_content_browser/folder.webp");

        loco_t::image_t icon_files_list = gloco->image_load("images_content_browser/files_list.webp");
        loco_t::image_t icon_files_big_thumbnail = gloco->image_load("images_content_browser/files_big_thumbnail.webp");

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
        float thumbnail_size = 128.0f;
        f32_t padding = 16.0f;
        std::string search_buffer;

        // lambda [this] capture
        content_browser_t(const content_browser_t&) = delete;
        content_browser_t(content_browser_t&&) = delete;

        content_browser_t() {
          search_buffer.resize(32);
          asset_path = std::filesystem::absolute(std::filesystem::path(asset_path)).wstring();
          current_directory = std::filesystem::path(asset_path);
          update_directory_cache();
        }
        content_browser_t(bool no_init) {

        }
        content_browser_t(const std::wstring& path) {
          init(path);
        }
        void init(const std::wstring& path) {
          search_buffer.resize(32);
          asset_path = std::filesystem::absolute(std::filesystem::path(asset_path)).wstring();
          current_directory = asset_path / std::filesystem::path(path);
          update_directory_cache();
        }
        void update_directory_cache() {
          for (auto& img : directory_cache) {
            if (img.preview_image.iic() == false) {
              gloco->image_unload(img.preview_image);
            }
          }
          directory_cache.clear();

          if (!directory_iterator.callback) {
            directory_iterator.sort_alphabetically = true;
            directory_iterator.callback = [this](const std::filesystem::directory_entry& entry) -> fan::event::task_t {
              file_info_t file_info;
              std::filesystem::path relative_path;
              try {
                // SLOW
                relative_path = std::filesystem::relative(entry, asset_path);
              }
              catch (const std::exception& e) {
                fan::print("exception came", e.what());
              }
              
              file_info.filename = relative_path.filename().string();
              file_info.item_path = relative_path.wstring();
              file_info.is_directory = entry.is_directory();
              file_info.some_path = entry.path().filename();//?
              //fan::print(get_file_extension(path.path().string()));
              if (fan::image::valid(entry.path().string())) {
                file_info.preview_image = gloco->image_load(std::filesystem::absolute(entry.path()).string());
              }
              directory_cache.push_back(file_info);
              co_return;
            };
          }
          
          fan::io::async_directory_iterate(
            &directory_iterator,
            current_directory.string()
          );
        }
        void render() {
          item_right_clicked = false;
          item_right_clicked_name.clear();
          ImGuiStyle& style = ImGui::GetStyle();
          ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 16.0f));
          ImGuiWindowClass window_class;
          //window_class.DockNodeFlagsOverrideSet = ImGuiDockNodeFlags_NoTabBar; TODO ?
          ImGui::SetNextWindowClass(&window_class);
          if (ImGui::Begin("Content Browser", 0, ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoTitleBar)) {
            if (ImGui::BeginMenuBar()) {
              ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.f, 0.f, 0.f, 0.f));
              ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.f, 0.f, 0.f, 0.f));
              ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.3f, 0.3f));

              if (fan::graphics::gui::image_button("##icon_arrow_left", icon_arrow_left, fan::vec2(32))) {
                if (std::filesystem::equivalent(current_directory, asset_path) == false) {
                  current_directory = current_directory.parent_path();
                }
                update_directory_cache();
              }
              ImGui::SameLine();
              fan::graphics::gui::image_button("##icon_arrow_right", icon_arrow_right, fan::vec2(32));
              ImGui::SameLine();
              ImGui::PopStyleColor(3);

              ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.f, 0.f, 0.f, 0.f));
              ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.f, 0.f, 0.f, 0.f));
              ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.3f, 0.3f));

              auto image_list = std::to_array({ icon_files_list, icon_files_big_thumbnail });

              fan::vec2 bc = fan::graphics::gui::get_position_bottom_corner();

              bc.x -= ImGui::GetWindowPos().x;
              ImGui::SetCursorPosX(bc.x / 2);

              fan::vec2 button_sizes = 32;

              ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - (button_sizes.x * 2 + style.ItemSpacing.x) * image_list.size());

              ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 20.0f);
              ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 7.0f));
              f32_t y_pos = ImGui::GetCursorPosY() + ImGui::GetStyle().WindowPadding.y;
              ImGui::SetCursorPosY(y_pos);


              if (ImGui::InputText("##content_browser_search", search_buffer.data(), search_buffer.size())) {

              }
              ImGui::PopStyleVar(2);

              fan::graphics::gui::toggle_image_button(image_list, button_sizes, (int*)&current_view_mode);

              ImGui::PopStyleColor(3);

              ///ImGui::InputText("Search", search_buffer.data(), search_buffer.size());

              ImGui::EndMenuBar();
            }
            switch (current_view_mode) {
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
        void render_large_thumbnails_view() {
          float thumbnail_size = 128.0f;
          float panel_width = ImGui::GetContentRegionAvail().x;
          int column_count = std::max((int)(panel_width / (thumbnail_size + padding)), 1);

          ImGui::Columns(column_count, 0, false);

          int pressed_key = -1;

          auto& style = ImGui::GetStyle();
          // basically bad way to check if gui is disabled. I couldn't find other way
          if (style.DisabledAlpha != style.Alpha) {
            if (ImGui::IsWindowFocused()) {
              for (int i = ImGuiKey_A; i != ImGuiKey_Z + 1; ++i) {
                if (ImGui::IsKeyPressed((ImGuiKey)i, false)) {
                  pressed_key = (i - ImGuiKey_A) + 'A';
                  break;
                }
              }
            }
          }

          for (std::size_t i = 0; i < directory_cache.size(); ++i) {
            // reference somehow corrupts
            auto file_info = directory_cache[i];
            if (search_buffer.size() && strstr(file_info.filename.c_str(), search_buffer.c_str()) == nullptr) {
              continue;
            }

            if (pressed_key != -1 && ImGui::IsWindowFocused()) {
              if (!file_info.filename.empty() && std::tolower(file_info.filename[0]) == std::tolower(pressed_key)) {
                ImGui::SetScrollHereY();
              }
            }

            ImGui::PushID(file_info.filename.c_str());
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
            fan::graphics::gui::image_button("##" + file_info.filename, file_info.preview_image.iic() == false ? file_info.preview_image : file_info.is_directory ? icon_directory : icon_file, ImVec2(thumbnail_size, thumbnail_size));

            bool item_hovered = ImGui::IsItemHovered();
            if (item_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
              item_right_clicked = true;
              item_right_clicked_name = file_info.filename;
              item_right_clicked_name.erase(std::remove_if(item_right_clicked_name.begin(), item_right_clicked_name.end(), isspace), item_right_clicked_name.end());
            }

            // Handle drag and drop, double click, etc.
            handle_item_interaction(file_info);

            ImGui::PopStyleColor();
            ImGui::TextWrapped("%s", file_info.filename.c_str());
            ImGui::NextColumn();
            ImGui::PopID();
          }

          ImGui::Columns(1);
        }
        void render_list_view() {
          if (ImGui::BeginTable("##FileTable", 1, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY
            | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV
            | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_Sortable)) {
            ImGui::TableSetupColumn("##Filename", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableHeadersRow();

            int pressed_key = -1;
            ImGuiStyle& style = ImGui::GetStyle();
            if (style.DisabledAlpha != style.Alpha) {
              if (ImGui::IsWindowFocused()) {
                for (int i = ImGuiKey_A; i != ImGuiKey_Z + 1; ++i) {
                  if (ImGui::IsKeyPressed((ImGuiKey)i, false)) {
                    pressed_key = (i - ImGuiKey_A) + 'A';
                    break;
                  }
                }
              }
            }

            // Render table view
            for (std::size_t i = 0; i < directory_cache.size(); ++i) {

              // reference somehow corrupts
              auto file_info = directory_cache[i];

              if (pressed_key != -1 && ImGui::IsWindowFocused()) {
                if (!file_info.filename.empty() && std::tolower(file_info.filename[0]) == std::tolower(pressed_key)) {
                  ImGui::SetScrollHereY();
                }
              }

              if (search_buffer.size() && strstr(file_info.filename.c_str(), search_buffer.c_str()) == nullptr) {
                continue;
              }
              ImGui::TableNextRow();
              ImGui::TableSetColumnIndex(0); // Icon column
              fan::vec2 cursor_pos = fan::vec2(ImGui::GetWindowPos()) + fan::vec2(ImGui::GetCursorPos()) + fan::vec2(ImGui::GetScrollX(), -ImGui::GetScrollY());
              fan::vec2 image_size = ImVec2(thumbnail_size / 4, thumbnail_size / 4);
              ImGuiStyle& style = ImGui::GetStyle();
              std::string space = "";
              while (ImGui::CalcTextSize(space.c_str()).x < image_size.x) {
                space += " ";
              }
              auto str = space + file_info.filename;

              ImGui::Selectable(str.c_str());
              bool item_hovered = ImGui::IsItemHovered();
              if (item_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                item_right_clicked_name = str;
                item_right_clicked_name.erase(std::remove_if(item_right_clicked_name.begin(), item_right_clicked_name.end(), isspace), item_right_clicked_name.end());
                item_right_clicked = true;
              }
              if (file_info.preview_image.iic() == false) {
                ImGui::GetWindowDrawList()->AddImage((ImTextureID)gloco->image_get_handle(file_info.preview_image), cursor_pos, cursor_pos + image_size);
              }
              else if (file_info.is_directory) {
                ImGui::GetWindowDrawList()->AddImage((ImTextureID)gloco->image_get_handle(icon_directory), cursor_pos, cursor_pos + image_size);
              }
              else {
                ImGui::GetWindowDrawList()->AddImage((ImTextureID)gloco->image_get_handle(icon_file), cursor_pos, cursor_pos + image_size);
              }

              handle_item_interaction(file_info);
            }

            ImGui::EndTable();
          }
        }
        void handle_item_interaction(const file_info_t& file_info) {
          if (file_info.is_directory == false) {

            if (ImGui::BeginDragDropSource()) {
              ImGui::SetDragDropPayload("CONTENT_BROWSER_ITEM", file_info.item_path.data(), (file_info.item_path.size() + 1) * sizeof(wchar_t));
              ImGui::Text("%s", file_info.filename.c_str());
              ImGui::EndDragDropSource();
            }
          }

          if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            if (file_info.is_directory) {
              current_directory /= file_info.some_path;
              update_directory_cache();
            }
          }
        }

        // [](const std::filesystem::path& path) {}
        void receive_drag_drop_target(auto receive_func) {
          ImGui::Dummy(ImGui::GetContentRegionAvail());

          if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("CONTENT_BROWSER_ITEM")) {
              const wchar_t* path = (const wchar_t*)payload->Data;
              receive_func(std::filesystem::absolute(std::filesystem::path(asset_path) / path));
            }
            ImGui::EndDragDropTarget();
          }
        }
      };

      struct dialogue_box_t {

        dialogue_box_t() {
          gloco->input_action.add(fan::mouse_left, "skip or continue dialog");
        }

        // 0-1
        void set_cursor_position(const fan::vec2& pos) {
          this->cursor_position = pos;
        }
        fan::event::task_t text(const std::string& text) {
          active_dialogue = text;
          render_pos = 0;
          finish_dialog = false;
          while (render_pos < active_dialogue.size() && !finish_dialog) {
            ++render_pos;
            co_await fan::co_sleep(1000 / character_per_s);
          }
          render_pos = active_dialogue.size();
        }

        fan::event::task_t button(const std::string& text, const fan::vec2& position = -1, const fan::vec2& size = { 0, 0 }) {
          button_choice = -1;
          button_t button;
          button.position = position;
          button.size = size;
          button.text = text;
          buttons.push_back(button);
          co_return;
        }
        int get_button_choice() const {
          return button_choice;
        }

        fan::event::task_t wait_user_input() {
          wait_user = true;
          fan::time::clock c;
          c.start(0.5e9);
          int prev_render = render_pos;
          while (wait_user) {
            if (c.finished()) {
              if (prev_render == render_pos) {
                render_pos = std::max(prev_render - 1, 0);
              }
              else {
                render_pos = prev_render;
              }
              c.restart();
            }
            co_await fan::co_sleep(10);
          }
          render_pos = prev_render;
        }

        void render(const std::string& window_name, ImFont* font, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing) {
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
          ImGui::SetCursorPos(ImVec2(100.0f, 100.f));
          ImGui::BeginChild((window_name + "child").c_str(), fan::vec2(wrap_width, 0), 0, ImGuiWindowFlags_NoNavInputs | ImGuiWindowFlags_NoTitleBar |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoBackground);
          if (wait_user == false) {
            ImGui::SetScrollY(ImGui::GetScrollMaxY());
          }
          fan::graphics::text_partial_render(active_dialogue.c_str(), render_pos, wrap_width, line_spacing);
          ImGui::EndChild();
          if (wait_user) {
            fan::vec2 first_pos = -1;


            // calculate biggest button
            fan::vec2 button_size = 0;
            for (std::size_t i = 0; i < buttons.size(); ++i) {
              fan::vec2 text_size = ImGui::CalcTextSize(buttons[i].text.c_str());
              float padding_x = ImGui::GetStyle().FramePadding.x;
              float padding_y = ImGui::GetStyle().FramePadding.y;
              ImVec2 bs = ImVec2(text_size.x + padding_x * 2.0f, text_size.y + padding_y * 2.0f);
              button_size = button_size.max(fan::vec2(bs));
            }

            for (std::size_t i = 0; i < buttons.size(); ++i) {
              const auto& button = buttons[i];
              if (button.position != -1) {
                first_pos = button.position;
                ImGui::SetCursorPos((button.position * window_size) - button_size / 2);
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() + ImGui::GetScrollY());
              }
              else {
                ImGui::SetCursorPosX(first_pos.x * window_size.x - button_size.x / 2);
              }
              ImGui::PushID(i);

              if (fan::graphics::gui::image_text_button(gloco->default_texture, button.text.c_str(), fan::colors::white, button.size == 0 ? button_size : button.size)) {
                button_choice = i;
                buttons.clear();
                wait_user = false;
                ImGui::PopID();
                break;
              }
              ImGui::PopID();
            }
          }
          if (gloco->input_action.is_active("skip or continue dialog") && ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows | ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem)) {
            finish_dialog = true;
            wait_user = false;
          }
          ImGui::End();
          ImGui::PopFont();
        }

        bool finish_dialog = false; // for skipping
        bool wait_user = false;
        std::string active_dialogue;

        uint64_t character_per_s = 20;
        std::size_t render_pos = 0;
        fan::vec2 cursor_position = -1;
        struct button_t {
          fan::vec2 position = -1;
          fan::vec2 size = 0;
          std::string text;
        };
        std::vector<button_t> buttons;
        int button_choice = -1;
      };
    }
  }
}

template fan::graphics::gui::imgui_fs_var_t::imgui_fs_var_t(
  loco_t::shader_t shader_nr,
  const std::string& var_name,
  fan::vec2 initial_,
  f32_t speed,
  f32_t min,
  f32_t max
);
template fan::graphics::gui::imgui_fs_var_t::imgui_fs_var_t(
  loco_t::shader_t shader_nr,
  const std::string& var_name,
  double initial_,
  f32_t speed,
  f32_t min,
  f32_t max
);
#endif