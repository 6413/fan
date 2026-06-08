module;

#if defined(FAN_GUI)
#endif

export module fan.graphics.gui.drag_drop;

#if defined (FAN_WINDOW)

import std;

#if defined(FAN_GUI)
import fan.graphics.gameplay.types;

export namespace fan::graphics::gui::drag_drop {

  struct drag_state_t {
    bool active = false;
    bool from_secondary = false;
    std::uint32_t slot_index = 0;
    std::uint32_t id = 0;
    std::uint32_t stack_size = 0;
  };

  void begin_from_slot(drag_state_t& drag_state, gameplay::item_slot_t& slot, bool from_secondary, std::uint32_t slot_index);
  void apply_to_slot(drag_state_t& drag_state, gameplay::item_slot_t& dst_slot);
  void cancel(drag_state_t& drag_state);
  void render_visual(const gameplay::gui_theme_t& theme, const drag_state_t& drag_state);

  struct drop_target_t {
    virtual bool hover(std::uint32_t& out_index) = 0;
    virtual bool drop(std::uint32_t index, gui::drag_drop::drag_state_t& s) = 0;
    virtual ~drop_target_t() {}
  };
}
#endif

#endif