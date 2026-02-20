module;

#include <vector>

#if defined(FAN_GUI)
  #include <fan/imgui/imgui.h>
  #include <fan/imgui/imgui_internal.h>
  #include <fan/imgui/imgui_impl_glfw.h>
  #include <fan/imgui/implot.h>
#endif

#include <string>
#include <algorithm> // topmost find

export module fan.graphics.gui.types;

import fan.types.compile_time_string;

#if defined(FAN_GUI)
export namespace fan::graphics::gui {
  struct label_t : std::string_view {
    using std::string_view::string_view;

    label_t(const char* s) : std::string_view(s) {}
    label_t(std::string_view s) : std::string_view(s) {}
    label_t(const std::string&) = delete;
    label_t(std::string&) = delete;
    label_t(std::string&&) = delete;

    operator const char* () const {
      return data();
    }
  };

  struct topmost_window_data_t {
    std::vector<std::string> windows;

    void register_window(std::string_view name) {
      if (std::find(windows.begin(), windows.end(), name) == windows.end()) {
        windows.push_back(std::string(name));
      }
    }

    void unregister_window(std::string_view name) {
      auto it = std::find(windows.begin(), windows.end(), name);
      if (it != windows.end()) {
        windows.erase(it);
      }
    }
  };

  enum dock_flags_e {
    dock_space = ImGuiDockNodeFlags_DockSpace,
    dock_central_node = ImGuiDockNodeFlags_CentralNode,
    dock_no_tab_bar = ImGuiDockNodeFlags_NoTabBar,
    dock_hidden_tab_bar = ImGuiDockNodeFlags_HiddenTabBar,
    dock_no_window_menu_button = ImGuiDockNodeFlags_NoWindowMenuButton,
    dock_no_close_button = ImGuiDockNodeFlags_NoCloseButton,
    dock_no_resize_x = ImGuiDockNodeFlags_NoResizeX,
    dock_no_resize_y = ImGuiDockNodeFlags_NoResizeY,
    dock_focus_routed_windows = ImGuiDockNodeFlags_DockedWindowsInFocusRoute,

    // Docking restrictions
    dock_no_docking_split_other = ImGuiDockNodeFlags_NoDockingSplitOther,
    dock_no_docking_over_me = ImGuiDockNodeFlags_NoDockingOverMe,
    dock_no_docking_over_other = ImGuiDockNodeFlags_NoDockingOverOther,
    dock_no_docking_over_empty = ImGuiDockNodeFlags_NoDockingOverEmpty,
    dock_no_docking = ImGuiDockNodeFlags_NoDocking,

    // Masks
    dock_inherit_mask = ImGuiDockNodeFlags_SharedFlagsInheritMask_,
    dock_no_resize_mask = ImGuiDockNodeFlags_NoResizeFlagsMask_,
    dock_transfer_mask = ImGuiDockNodeFlags_LocalFlagsTransferMask_,
    dock_saved_mask = ImGuiDockNodeFlags_SavedFlagsMask_,
  };


  enum dock_node_flags_e {
    dock_node_flags_none = ImGuiDockNodeFlags_None,                   
    dock_node_flags_keep_alive_only = ImGuiDockNodeFlags_KeepAliveOnly,   //       // Don't display the dockspace node but keep it alive. Windows docked into this dockspace node won't be undocked.
    //dock_node_flags_none =    ImGuiDockNodeFlags_NoCentralNode        ,   //       // Disable Central Node (the node which can stay empty)
    dock_node_flags_no_docking_over_cenrtal_node = ImGuiDockNodeFlags_NoDockingOverCentralNode,   //       // Disable docking over the Central Node, which will be always kept empty.
    dock_node_flags_passthru_central_node = ImGuiDockNodeFlags_PassthruCentralNode,   //       // Enable passthru dockspace: 1) DockSpace() will render a ImGuiCol_WindowBg background covering everything excepted the Central Node when empty. Meaning the host window should probably use SetNextWindowBgAlpha(0.0f) prior to Begin() when using this. 2) When Central Node is empty: let inputs pass-through + won't display a DockingEmptyBg background. See demo for detaiodee_s_keflagls.
    dock_node_flags_no_docking_split = ImGuiDockNodeFlags_NoDockingSplit,   //       // Disable other windows/nodes from splitting this node.
    dock_node_flags_no_resize = ImGuiDockNodeFlags_NoResize,   // Saved // Disable resizing node using the splitter/separators. Useful with programmatically setup dockspaces.
    dock_node_flags_auto_hide_tab_bar = ImGuiDockNodeFlags_AutoHideTabBar,   //       // Tab bar will automatically hide when there is a single window in the dock node.
    dock_node_flags_no_undocking = ImGuiDockNodeFlags_NoUndocking,   //       // Disable undocking this node.
  };

  enum window_flags_e : uint64_t {
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
    window_flags_override_input = 1ULL << 31, // ignores this window from want_io()
    window_flags_topmost = 1ULL << 32,
  };
  enum child_flags_e {
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
  enum tab_item_flags_e{
    tab_item_flags_none = ImGuiTabItemFlags_None,
    tab_item_flags_unsaved_document = ImGuiTabItemFlags_UnsavedDocument,
    tab_item_flags_set_selected = ImGuiTabItemFlags_SetSelected,
    tab_item_flags_no_close_with_middle_button = ImGuiTabItemFlags_NoCloseWithMiddleMouseButton,
    tab_item_flags_no_push_id = ImGuiTabItemFlags_NoPushId,
    tab_item_flags_no_tooltip = ImGuiTabItemFlags_NoTooltip,
    tab_item_flags_no_reorder = ImGuiTabItemFlags_NoReorder,
    tab_item_flags_leading = ImGuiTabItemFlags_Leading,
    tab_item_flags_trailing = ImGuiTabItemFlags_Trailing,
    tab_item_flags_no_assumed_closure = ImGuiTabItemFlags_NoAssumedClosure
  };
  enum hover_flags_e {
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
  enum selectable_flags_e {
    selectable_flags_none = ImGuiSelectableFlags_None,
    selectable_flags_no_auto_close_popups = ImGuiSelectableFlags_NoAutoClosePopups,  // Clicking this doesn't close parent popup window (overrides ImGuiItemFlags_AutoClosePopups).
    selectable_flags_span_all_columns = ImGuiSelectableFlags_SpanAllColumns,     // Frame will span all columns of its container table (text will still fit in current column).
    selectable_flags_allow_double_click = ImGuiSelectableFlags_AllowDoubleClick,   // Generate press events on double clicks too.
    selectable_flags_disabled = ImGuiSelectableFlags_Disabled,           // Cannot be selected, display grayed-out text.
    selectable_flags_allow_overlap = ImGuiSelectableFlags_AllowOverlap,       // (WIP) Hit testing to allow subsequent widgets to overlap this one.
    selectable_flags_highlight = ImGuiSelectableFlags_Highlight,          // Make the item be displayed as if it is hovered.
  };
  enum table_flags_e {
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
  // Flags for ImGui::TableNextRow()
  enum table_row_flags_e {
    table_row_flags_none = ImGuiTableRowFlags_None,
    table_row_flags_headers = ImGuiTableRowFlags_Headers,   // Identify header row (set default background color + width of its contents accounted differently for auto column width)
  };
  enum table_column_flags_e {
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
  enum slider_flags_e {
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
  enum input_text_flags_e {
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
  enum input_flags_e {
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
  enum color_edit_flags_e : uint64_t {
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
    color_edit_flags_init_once = 1ULL << 32,
  };
  enum cond_e {
    cond_none = ImGuiCond_None,        // No condition (always set the variable), same as _Always
    cond_always = ImGuiCond_Always,   // No condition (always set the variable), same as _None
    cond_once = ImGuiCond_Once,   // Set the variable once per runtime session (only the first call will succeed)
    cond_first_use_ever = ImGuiCond_FirstUseEver,   // Set the variable if the object/window has no persistently saved data (no entry in .ini file)
    cond_appearing = ImGuiCond_Appearing,   // Set the variable if the object/window is appearing after being hidden/inactive (or the first time)
  };
  enum col_e {
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
  enum cursor_e {
    mouse_cursor_none = ImGuiMouseCursor_None,
    mouse_cursor_arrow = ImGuiMouseCursor_Arrow,
    mouse_cursor_text_input = ImGuiMouseCursor_TextInput,           // When hovering over InputText, etc.
    mouse_cursor_resize_all = ImGuiMouseCursor_ResizeAll,           // (Unused by Dear ImGui functions)
    mouse_cursor_resize_ns = ImGuiMouseCursor_ResizeNS,             // When hovering over a horizontal border
    mouse_cursor_resize_ew = ImGuiMouseCursor_ResizeEW,             // When hovering over a vertical border or a column
    mouse_cursor_resize_nesw = ImGuiMouseCursor_ResizeNESW,         // When hovering over the bottom-left corner of a window
    mouse_cursor_resize_nwse = ImGuiMouseCursor_ResizeNWSE,         // When hovering over the bottom-right corner of a window
    mouse_cursor_hand = ImGuiMouseCursor_Hand,                      // (Unused by Dear ImGui functions. Use for e.g. hyperlinks)
    mouse_cursor_not_allowed = ImGuiMouseCursor_NotAllowed,         // When hovering something with disallowed interaction
    mouse_cursor_count = ImGuiMouseCursor_COUNT,
  };
  enum style_var_e {
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
  enum tree_node_flags_e {
    tree_node_flags_none = 0,
    tree_node_flags_selected = 1 << 0,   // Draw as selected
    tree_node_flags_framed = 1 << 1,   // Draw frame with background (e.g. for CollapsingHeader)
    tree_node_flags_allow_overlap = 1 << 2,   // Hit testing to allow subsequent widgets to overlap this one
    tree_node_flags_no_tree_push_on_open = 1 << 3,   // Don't do a TreePush() when open (e.g. for CollapsingHeader) = no extra indent nor pushing on ID stack
    tree_node_flags_no_auto_open_on_log = 1 << 4,   // Don't automatically and temporarily open node when Logging is active (by default logging will automatically open tree nodes)
    tree_node_flags_default_open = 1 << 5,   // Default node to be open
    tree_node_flags_open_on_double_click = 1 << 6,   // Open on double-click instead of simple click (default for multi-select unless any _OpenOnXXX behavior is set explicitly). Both behaviors may be combined.
    tree_node_flags_open_on_arrow = 1 << 7,   // Open when clicking on the arrow part (default for multi-select unless any _OpenOnXXX behavior is set explicitly). Both behaviors may be combined.
    tree_node_flags_leaf = 1 << 8,   // No collapsing, no arrow (use as a convenience for leaf nodes).
    tree_node_flags_bullet = 1 << 9,   // Display a bullet instead of arrow. IMPORTANT: node can still be marked open/close if you don't set the _Leaf flag!
    tree_node_flags_frame_padding = 1 << 10,  // Use FramePadding (even for an unframed text node) to vertically align text baseline to regular widget height. Equivalent to calling AlignTextToFramePadding() before the node.
    tree_node_flags_span_avail_width = 1 << 11,  // Extend hit box to the right-most edge, even if not framed. This is not the default in order to allow adding other items on the same line without using AllowOverlap mode.
    tree_node_flags_span_full_width = 1 << 12,  // Extend hit box to the left-most and right-most edges (cover the indent area).
    tree_node_flags_span_label_width = 1 << 13,  // Narrow hit box + narrow hovering highlight, will only cover the label text.
    tree_node_flags_span_all_columns = 1 << 14,  // Frame will span all columns of its container table (label will still fit in current column)
    tree_node_flags_label_span_all_columns = 1 << 15,  // Label will span all columns of its container table
    // tree_node_flags_no_scroll_on_open  = 1 << 16,  // FIXME: TODO: Disable automatic scroll on TreePop() if node got just open and contents is not visible
    tree_node_flags_nav_left_jumps_back_here = 1 << 17,  // (WIP) Nav: left direction may move to this TreeNode() from any of its child (items submitted between TreeNode and TreePop)
    tree_node_flags_collapsing_header = tree_node_flags_framed | tree_node_flags_no_tree_push_on_open | tree_node_flags_no_auto_open_on_log,

  #ifndef IMGUI_DISABLE_OBSOLETE_FUNCTIONS
    tree_node_flags_allow_item_overlap = tree_node_flags_allow_overlap,   // Renamed in 1.89.7
    tree_node_flags_span_text_width = tree_node_flags_span_label_width,// Renamed in 1.90.7
  #endif
  };
  enum config_flags_e {
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

  enum item_flags_e {
    item_flags_none = ImGuiItemFlags_None,        // (Default)
    item_flags_no_tab_stop = ImGuiItemFlags_NoTabStop,   // false    // Disable keyboard tabbing. This is a "lighter" version of ImGuiItemFlags_NoNav.
    item_flags_no_nav = ImGuiItemFlags_NoNav,   // false    // Disable any form of focusing (keyboard/gamepad directional navigation and SetKeyboardFocusHere() calls).
    item_flags_no_nav_default_focus = ImGuiItemFlags_NoNavDefaultFocus,   // false    // Disable item being a candidate for default focus (e.g. used by title bar items).
    item_flags_button_repeat = ImGuiItemFlags_ButtonRepeat,   // false    // Any button-like behavior will have repeat mode enabled (based on io.KeyRepeatDelay and io.KeyRepeatRate values). Note that you can also call IsItemActive() after any button to tell if it is being held.
    item_flags_auto_close_popups = ImGuiItemFlags_AutoClosePopups,   // true     // MenuItem()/Selectable() automatically close their parent popup window.
    item_flags_allow_duplicate_id = ImGuiItemFlags_AllowDuplicateId,   // false    // Allow submitting an item with the same identifier as an item already submitted this frame without triggering a warning tooltip if io.ConfigDebugHighlightIdConflicts is set.
  };

  using context_t = ImGuiContext;
  using window_handle_t = ImGuiWindow;
  using io_t = ImGuiIO;
  using input_text_callback_t = ImGuiInputTextCallback;
  using draw_list_t = ImDrawList;
  using table_column_flags_t = int;
  using slider_flags_t = int;
  using input_text_flags_t = int;
  using input_flags_t = int;
  using color_edit_flags_t = uint64_t;
  using cond_t = int;
  using col_t = int;
  using cursor_t = int;
  using tree_node_flags_t = int;
  using style_t = ImGuiStyle;
  using font_t = ImFont;
  using font_config_t = ImFontConfig;
  using dock_flag_t = int;
  using window_flags_t = uint64_t;
  using child_window_flags_t = int;
  using tab_item_flags_t = int;
  using hovered_flag_t = int;
  using selectable_flag_t = int;
  using table_data_t = ImGuiTable;
  using table_flags_t = int;
  using table_row_flags_t = int;
  using style_var_t = int;
  using id_t = ImGuiID;
  using viewport_t = ImGuiViewport;
  using data_type_t = ImGuiDataType;
  using texture_id_t = ImTextureID;
  using rect_t = ImRect;
  using item_flags_t = ImGuiItemFlags;
  using key_t = ImGuiKey;
  using class_t = ImGuiWindowClass;
  using payload_t = ImGuiPayload;
  using storage_t = ImGuiStorage;
  using draw_data_t = ImDrawData;

  using u32_t = ImU32;
  using vec4_t = ImVec4;

  struct dir_t {
    constexpr dir_t(ImGuiDir dir) : d(dir) {}
    constexpr operator ImGuiDir&() { return d; }
    constexpr operator const ImGuiDir&() const { return d; }
    ImGuiDir d;
  };

  inline constexpr dir_t dir_none = ImGuiDir_None;
  inline constexpr dir_t dir_left = ImGuiDir_Left;
  inline constexpr dir_t dir_right = ImGuiDir_Right;
  inline constexpr dir_t dir_up = ImGuiDir_Up;
  inline constexpr dir_t dir_down = ImGuiDir_Down;
  inline constexpr dir_t dir_count = ImGuiDir_COUNT;

  struct InputTextCallback_UserData {
    std::string* Str;
    ImGuiInputTextCallback  ChainCallback;
    void* ChainCallbackUserData;
  };
}

export namespace fan::graphics::gui::plot {
  using flags_t = int;
  using line_flags_t = int;
  using axis_flags_t = int;
  using scatter_flags_t = int;
  using bars_flags_t = int;
  using item_flags_t = int;
  using cond_t = int;
  using col_t = int;
  using marker_t = int;
  using location_t = int;
  using axis_t = ImAxis;
  using formatter_t = ImPlotFormatter;
  using point_t = ImPlotPoint;
  inline constexpr int plot_auto = IMPLOT_AUTO;

  enum flags_e{
    flags_none = ImPlotFlags_None,
    flags_no_title = ImPlotFlags_NoTitle,
    flags_no_legend = ImPlotFlags_NoLegend,
    flags_no_mouse_text = ImPlotFlags_NoMouseText,
    flags_no_inputs = ImPlotFlags_NoInputs,
    flags_no_menus = ImPlotFlags_NoMenus,
    flags_no_box_select = ImPlotFlags_NoBoxSelect,
    flags_no_frame = ImPlotFlags_NoFrame,
    flags_equal = ImPlotFlags_Equal,
    flags_crosshairs = ImPlotFlags_Crosshairs,
    flags_canvas_only = ImPlotFlags_CanvasOnly
  };
  enum axis_flags_e {
    axis_flags_none = ImPlotAxisFlags_None,
    axis_flags_no_label = ImPlotAxisFlags_NoLabel,
    axis_flags_no_grid_lines = ImPlotAxisFlags_NoGridLines,
    axis_flags_no_tick_marks = ImPlotAxisFlags_NoTickMarks,
    axis_flags_no_tick_labels = ImPlotAxisFlags_NoTickLabels,
    axis_flags_no_initial_fit = ImPlotAxisFlags_NoInitialFit,
    axis_flags_no_menus = ImPlotAxisFlags_NoMenus,
    axis_flags_no_side_switch = ImPlotAxisFlags_NoSideSwitch,
    axis_flags_no_highlight = ImPlotAxisFlags_NoHighlight,
    axis_flags_opposite = ImPlotAxisFlags_Opposite,
    axis_flags_foreground = ImPlotAxisFlags_Foreground,
    axis_flags_invert = ImPlotAxisFlags_Invert,
    axis_flags_auto_fit = ImPlotAxisFlags_AutoFit,
    axis_flags_range_fit = ImPlotAxisFlags_RangeFit,
    axis_flags_pan_stretch = ImPlotAxisFlags_PanStretch,
    axis_flags_lock_min = ImPlotAxisFlags_LockMin,
    axis_flags_lock_max = ImPlotAxisFlags_LockMax,
    axis_flags_lock = ImPlotAxisFlags_Lock,
    axis_flags_no_decorations = ImPlotAxisFlags_NoDecorations,
    axis_flags_aux_default = ImPlotAxisFlags_AuxDefault
  };
  enum line_flags_e {
    line_flags_none = ImPlotLineFlags_None,
    line_flags_segments = ImPlotLineFlags_Segments,
    line_flags_loop = ImPlotLineFlags_Loop,
    line_flags_skip_nan = ImPlotLineFlags_SkipNaN,
    line_flags_no_clip = ImPlotLineFlags_NoClip,
    line_flags_shaded = ImPlotLineFlags_Shaded
  };
  enum scatter_flags_e {
    scatter_flags_none = ImPlotScatterFlags_None,
    scatter_flags_no_clip = ImPlotScatterFlags_NoClip
  };
  enum bar_flags_e {
    bars_flags_none = ImPlotBarsFlags_None,
    bars_flags_horizontal = ImPlotBarsFlags_Horizontal
  };
  enum item_flags_e {
    item_flags_none = ImPlotItemFlags_None,
    item_flags_no_legend = ImPlotItemFlags_NoLegend,
    item_flags_no_fit = ImPlotItemFlags_NoFit
  };
  enum cond_e {
    cond_none = ImPlotCond_None,
    cond_always = ImPlotCond_Always,
    cond_once = ImPlotCond_Once
  };
  enum col_e {
    col_line = ImPlotCol_Line,
    col_fill = ImPlotCol_Fill,
    col_marker_outline = ImPlotCol_MarkerOutline,
    col_marker_fill = ImPlotCol_MarkerFill,
    col_error_bar = ImPlotCol_ErrorBar,
    col_frame_bg = ImPlotCol_FrameBg,
    col_plot_bg = ImPlotCol_PlotBg,
    col_plot_border = ImPlotCol_PlotBorder,
    col_legend_bg = ImPlotCol_LegendBg,
    col_legend_border = ImPlotCol_LegendBorder,
    col_legend_text = ImPlotCol_LegendText,
    col_title_text = ImPlotCol_TitleText,
    col_inlay_text = ImPlotCol_InlayText,
    col_axis_text = ImPlotCol_AxisText,
    col_axis_grid = ImPlotCol_AxisGrid,
    col_axis_tick = ImPlotCol_AxisTick,
    col_axis_bg = ImPlotCol_AxisBg,
    col_axis_bg_hovered = ImPlotCol_AxisBgHovered,
    col_axis_bg_active = ImPlotCol_AxisBgActive,
    col_selection = ImPlotCol_Selection,
    col_crosshairs = ImPlotCol_Crosshairs
  };
  enum marker_e {
    marker_none = ImPlotMarker_None,
    marker_circle = ImPlotMarker_Circle,
    marker_square = ImPlotMarker_Square,
    marker_diamond = ImPlotMarker_Diamond,
    marker_up = ImPlotMarker_Up,
    marker_down = ImPlotMarker_Down,
    marker_left = ImPlotMarker_Left,
    marker_right = ImPlotMarker_Right,
    marker_cross = ImPlotMarker_Cross,
    marker_plus = ImPlotMarker_Plus,
    marker_asterisk = ImPlotMarker_Asterisk
  };
  enum location_e {
    location_center = ImPlotLocation_Center,
    location_north = ImPlotLocation_North,
    location_south = ImPlotLocation_South,
    location_west = ImPlotLocation_West,
    location_east = ImPlotLocation_East,
    location_north_west = ImPlotLocation_NorthWest,
    location_north_east = ImPlotLocation_NorthEast,
    location_south_west = ImPlotLocation_SouthWest,
    location_south_east = ImPlotLocation_SouthEast
  };
  enum axis_e {
    axis_x1 = ImAxis_X1,
    axis_x2 = ImAxis_X2,
    axis_x3 = ImAxis_X3,
    axis_y1 = ImAxis_Y1,
    axis_y2 = ImAxis_Y2,
    axis_y3 = ImAxis_Y3
  };
  enum style_var_e {
    // item styling variables
    style_var_line_weight = ImPlotStyleVar_LineWeight,         // float,  plot item line weight in pixels
    style_var_marker = ImPlotStyleVar_Marker,                  // int,    marker specification
    style_var_marker_size = ImPlotStyleVar_MarkerSize,         // float,  marker size in pixels (roughly the marker's "radius")
    style_var_marker_weight = ImPlotStyleVar_MarkerWeight,     // float,  plot outline weight of markers in pixels
    style_var_fill_alpha = ImPlotStyleVar_FillAlpha,           // float,  alpha modifier applied to all plot item fills
    style_var_error_bar_size = ImPlotStyleVar_ErrorBarSize,    // float,  error bar whisker width in pixels
    style_var_error_bar_weight = ImPlotStyleVar_ErrorBarWeight,// float,  error bar whisker weight in pixels
    style_var_digital_bit_height = ImPlotStyleVar_DigitalBitHeight, // float,  digital channels bit height (at 1) in pixels
    style_var_digital_bit_gap = ImPlotStyleVar_DigitalBitGap,  // float,  digital channels bit padding gap in pixels
    // plot styling variables
    style_var_plot_border_size = ImPlotStyleVar_PlotBorderSize,     // float,  thickness of border around plot area
    style_var_minor_alpha = ImPlotStyleVar_MinorAlpha,         // float,  alpha multiplier applied to minor axis grid lines
    style_var_major_tick_len = ImPlotStyleVar_MajorTickLen,       // ImVec2, major tick lengths for X and Y axes
    style_var_minor_tick_len = ImPlotStyleVar_MinorTickLen,       // ImVec2, minor tick lengths for X and Y axes
    style_var_major_tick_size = ImPlotStyleVar_MajorTickSize,      // ImVec2, line thickness of major ticks
    style_var_minor_tick_size = ImPlotStyleVar_MinorTickSize,      // ImVec2, line thickness of minor ticks
    style_var_major_grid_size = ImPlotStyleVar_MajorGridSize,      // ImVec2, line thickness of major grid lines
    style_var_minor_grid_size = ImPlotStyleVar_MinorGridSize,      // ImVec2, line thickness of minor grid lines
    style_var_plot_padding = ImPlotStyleVar_PlotPadding,        // ImVec2, padding between widget frame and plot area, labels, or outside legends (i.e. main padding)
    style_var_label_padding = ImPlotStyleVar_LabelPadding,       // ImVec2, padding between axes labels, tick labels, and plot edge
    style_var_legend_padding = ImPlotStyleVar_LegendPadding,      // ImVec2, legend padding from plot edges
    style_var_legend_inner_padding = ImPlotStyleVar_LegendInnerPadding, // ImVec2, legend inner padding from legend edges
    style_var_legend_spacing = ImPlotStyleVar_LegendSpacing,      // ImVec2, spacing between legend entries
    style_var_mouse_pos_padding = ImPlotStyleVar_MousePosPadding,    // ImVec2, padding between plot edge and interior info text
    style_var_annotation_padding = ImPlotStyleVar_AnnotationPadding,  // ImVec2, text padding around annotation labels
    style_var_fit_padding = ImPlotStyleVar_FitPadding,         // ImVec2, additional fit padding as a percentage of the fit extents (e.g. ImVec2(0.1f,0.1f) adds 10% to the fit extents of X and Y)
    style_var_plot_default_size = ImPlotStyleVar_PlotDefaultSize,    // ImVec2, default size used when ImVec2(0,0) is passed to BeginPlot
    style_var_plot_min_size = ImPlotStyleVar_PlotMinSize,        // ImVec2, minimum size plot frame can be when shrunk
    style_var_count = ImPlotStyleVar_COUNT
  };
} // namespace plot

#endif