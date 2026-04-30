module;

#if defined(FAN_GUI)
#endif

export module fan.graphics.gui.slot_renderer;

import std;

#if defined(FAN_GUI)
import fan.types;
import fan.types.vector;
import fan.graphics.gui.drag_drop;
import fan.graphics.gameplay.types;

export namespace fan::graphics::gui {

  struct slot_layout_t {
    fan::vec2 slot_size;
    fan::vec2 slot_padding;
    f32_t border_thickness = 2.0f;
    f32_t corner_rounding = 0.0f;
    std::uint32_t columns = 1;
    bool horizontal = false;
  };

  struct slot_callbacks_t {
    using use_cb_t = void(*)(std::uint32_t slot_index, const gameplay::item_t&);
    
    use_cb_t on_use = nullptr;
    bool is_secondary = false;
    bool enable_ctrl_transfer = false;
  };

  struct slot_visual_state_t {
    std::uint32_t selected_slot = std::numeric_limits<std::uint32_t>::max();
    bool show_selection = false;
  };

  void handle_slot_click(
    std::uint32_t slot_index,
    gameplay::item_slot_t& slot,
    bool hovered,
    bool left_pressed,
    bool right_pressed,
    gui::drag_drop::drag_state_t& drag_state,
    const slot_callbacks_t& callbacks
  );

  std::uint32_t render_slot_grid(
    std::vector<gameplay::item_slot_t>& slots,
    std::uint32_t start_idx,
    std::uint32_t count,
    const slot_layout_t& layout,
    const gameplay::gui_theme_t& theme,
    gui::drag_drop::drag_state_t& drag_state,
    const slot_callbacks_t& callbacks,
    const slot_visual_state_t& visual = {}
  );
}
#endif