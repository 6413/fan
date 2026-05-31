module;

#if defined(FAN_GUI)
#endif

module fan.graphics.gui.drag_drop;

#if defined (FAN_WINDOW)

import std;

#if defined(FAN_GUI)
import fan.types.vector;
import fan.types.color;
import fan.window.input;
import fan.graphics.common_context;
import fan.graphics.gui.base;
import fan.graphics.gui.input;
import fan.graphics.gameplay.items;

namespace fan::graphics::gui::drag_drop {

  void begin_from_slot(drag_state_t& drag_state, gameplay::item_slot_t& slot, bool from_secondary, std::uint32_t slot_index) {
    if (slot.is_empty()) {
      return;
    }

    drag_state.active = true;
    drag_state.from_secondary = from_secondary;
    drag_state.slot_index = slot_index;
    drag_state.id = *slot.id;
    drag_state.stack_size = *slot.stack_size;

    bool split = gui::input::shift() && drag_state.stack_size > 1;
    if (split) {
      std::uint32_t half = drag_state.stack_size / 2;
      if (half == 0) {
        half = 1;
      }
      drag_state.stack_size = half;
      *slot.stack_size -= half;
      if (*slot.stack_size == 0) {
        slot.id.reset();
        slot.stack_size.reset();
      }
    }
    else {
      slot.id.reset();
      slot.stack_size.reset();
    }
  }

  void apply_to_slot(drag_state_t& drag_state, gameplay::item_slot_t& dst_slot) {
    if (!drag_state.active) {
      return;
    }

    auto& reg = fan::graphics::gameplay::items::get_registry();
    auto* def = reg.get_definition(drag_state.id);
    if (!def) {
      drag_state.active = false;
      return;
    }

    if (dst_slot.is_empty()) {
      dst_slot.id = drag_state.id;
      dst_slot.stack_size = drag_state.stack_size;
      drag_state.active = false;
      return;
    }

    if (*dst_slot.id == drag_state.id) {
      std::uint32_t free_space = 0;
      if (*dst_slot.stack_size < def->max_stack) {
        free_space = def->max_stack - *dst_slot.stack_size;
      }

      if (free_space == 0) {
        std::uint32_t tmp_id = *dst_slot.id;
        std::uint32_t tmp_stack = *dst_slot.stack_size;
        dst_slot.id = drag_state.id;
        dst_slot.stack_size = drag_state.stack_size;
        drag_state.id = tmp_id;
        drag_state.stack_size = tmp_stack;
        return;
      }

      std::uint32_t to_add = std::min(free_space, drag_state.stack_size);
      *dst_slot.stack_size += to_add;
      drag_state.stack_size -= to_add;

      if (drag_state.stack_size == 0) {
        drag_state.active = false;
      }
      return;
    }

    std::uint32_t tmp_id = *dst_slot.id;
    std::uint32_t tmp_stack = *dst_slot.stack_size;
    dst_slot.id = drag_state.id;
    dst_slot.stack_size = drag_state.stack_size;
    drag_state.id = tmp_id;
    drag_state.stack_size = tmp_stack;
  }

  void cancel(drag_state_t& drag_state) {
    drag_state.active = false;
  }

  void render_visual(const gameplay::gui_theme_t& theme, const drag_state_t& drag_state) {
    if (!drag_state.active) {
      return;
    }

    auto& reg = fan::graphics::gameplay::items::get_registry();
    auto* def = reg.get_definition(drag_state.id);
    if (!def) {
      return;
    }

    auto& io = gui::get_io();
    fan::vec2 mouse_pos = io.MousePos;

    auto* dl = gui::get_foreground_draw_list();
    if (!dl) {
      return;
    }

    fan::vec2 size(64, 64);
    fan::vec2 p_min = mouse_pos - size * 0.5f;
    fan::vec2 p_max = p_min + size;

    dl->AddRectFilled(p_min, p_max, theme.slot_bg_hover.get_gui_color(), 6);
    dl->AddRect(p_min, p_max, theme.slot_border.get_gui_color(), 6, 0, 2);

    if (def->icon.valid()) {
      fan::vec2 pad(6, 6);
      dl->AddImage(
        fan::graphics::image_get_handle(def->icon),
        p_min + pad,
        p_max - pad
      );
    }

    if (drag_state.stack_size > 1) {
      std::string s = std::to_string(drag_state.stack_size);
      fan::vec2 ts = gui::calc_text_size(s);
      fan::vec2 pos(p_max.x - ts.x - 6, p_max.y - ts.y - 6);
      dl->AddText(pos, fan::colors::white.get_gui_color(), s.c_str());
    }
  }
}
#endif

#endif