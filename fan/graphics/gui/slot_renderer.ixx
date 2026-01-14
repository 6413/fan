#include <cstdint>

export module fan.graphics.gui.slot_renderer;

import fan.graphics;
import fan.graphics.gui.base;
import fan.graphics.gui.input;
import fan.graphics.gui.drag_drop;
import fan.graphics.gameplay.items;
export import fan.graphics.gameplay.types;

using namespace fan::graphics;

export namespace fan::graphics::gui {

  struct slot_layout_t {
    fan::vec2 slot_size;
    fan::vec2 slot_padding;
    f32_t border_thickness = 2.0f;
    f32_t corner_rounding = 0.0f;
    uint32_t columns = 1;
    bool horizontal = false;
  };

  struct slot_callbacks_t {
    using use_cb_t = void(*)(uint32_t slot_index, const gameplay::item_t&);
    
    use_cb_t on_use = nullptr;
    bool is_secondary = false;
    bool enable_ctrl_transfer = false;
  };

  struct slot_visual_state_t {
    uint32_t selected_slot = UINT32_MAX;
    bool show_selection = false;
  };

  inline void handle_slot_click(
    uint32_t slot_index,
    gameplay::item_slot_t& slot,
    bool hovered,
    bool left_pressed,
    bool right_pressed,
    gui::drag_drop::drag_state_t& drag_state,
    const slot_callbacks_t& callbacks
  ) {
    if (left_pressed && hovered && !slot.is_empty()) {
      if (drag_state.active) {
        drag_drop::apply_to_slot(drag_state, slot);
        return;
      }
      drag_drop::begin_from_slot(drag_state, slot, callbacks.is_secondary, slot_index);
      return;
    }

    if (right_pressed && hovered && !slot.is_empty()) {
      if (callbacks.on_use) {
        if (callbacks.is_secondary) {
          *slot.stack_size -= 1;
          if (*slot.stack_size == 0) {
            slot.id.reset();
            slot.stack_size.reset();
          }
        }
        
        auto& reg = fan::graphics::gameplay::items::get_registry();
        auto* def = reg.get_definition(*slot.id);
        gameplay::item_t temp_item = reg.create_item(def->id);
        temp_item.stack_size = callbacks.is_secondary ? 1 : *slot.stack_size;
        callbacks.on_use(slot_index, temp_item);
      }
    }
  }

  inline uint32_t render_slot_grid(
    std::vector<gameplay::item_slot_t>& slots,
    uint32_t start_idx,
    uint32_t count,
    const slot_layout_t& layout,
    const gameplay::gui_theme_t& theme,
    gui::drag_drop::drag_state_t& drag_state,
    const slot_callbacks_t& callbacks,
    const slot_visual_state_t& visual = {}
  ) {
    fan::vec2 origin = gui::get_cursor_screen_pos();
    uint32_t hovered_slot = UINT32_MAX;

    for (uint32_t i = start_idx; i < start_idx + count && i < slots.size(); ++i) {
      uint32_t local = i - start_idx;
      uint32_t col, row;
      
      if (layout.horizontal) {
        col = local;
        row = 0;
      } else {
        col = local % layout.columns;
        row = local / layout.columns;
      }

      fan::vec2 slot_pos = origin + fan::vec2(
        col * (layout.slot_size.x + layout.slot_padding.x),
        row * (layout.slot_size.y + layout.slot_padding.y)
      );
      gui::set_cursor_screen_pos(slot_pos);

      auto& slot = slots[i];
      fan::vec2 cursor = gui::get_cursor_screen_pos();

      gui::push_id(static_cast<int>(i));
      gui::invisible_button("slot", layout.slot_size);

      bool left_pressed = gui::input::left_click();
      bool right_pressed = gui::input::right_click();

      fan::vec2 p_min = cursor;
      fan::vec2 p_max = cursor + layout.slot_size;

      bool hovered = gui::input::hover(p_min, p_max);

      if (hovered) {
        hovered_slot = i;
      }

      handle_slot_click(i, slot, hovered, left_pressed, right_pressed, drag_state, callbacks);

      auto* dl = gui::get_window_draw_list();

      fan::color bg_color = hovered ? theme.slot_bg_hover : theme.slot_bg;

      gui::slot::background(dl, p_min, p_max, bg_color, layout.corner_rounding);
      gui::slot::border(dl, p_min, p_max, theme.slot_border, layout.corner_rounding, layout.border_thickness);

      if (visual.show_selection && i == visual.selected_slot) {
        gui::slot::selected_border(dl, p_min, p_max, theme.selected_border_color, 4.0f, 3.0f);
      }

      if (!slot.is_empty()) {
        auto& reg = fan::graphics::gameplay::items::get_registry();
        auto* def = reg.get_definition(*slot.id);
        gui::slot::icon(def->icon, p_min, p_max, fan::vec2(8, 8));
        gui::slot::stack_count(*slot.stack_size, p_min, p_max);
        gui::slot::tooltip(def->name, hovered && !drag_state.active);
      }

      gui::pop_id();
    }

    return hovered_slot;
  }
}