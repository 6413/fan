module;

#include <cstdint>
#include <vector>

export module fan.graphics.gui.hotbar;

import fan.graphics;
import fan.graphics.gui.base;
import fan.graphics.gui.input;
import fan.graphics.gui.drag_drop;
import fan.graphics.gui.slot_renderer;
export import fan.graphics.gameplay.types;
export import fan.graphics.gameplay.items;

using namespace fan::graphics;

export namespace fan::graphics::gui {

  struct hotbar_t {
    using slot_select_cb_t = void(*)(uint32_t);
    using item_use_cb_t = void(*)(uint32_t, const gameplay::item_t&);

    void create(uint32_t slot_count = 9) {
      slots.resize(slot_count);
      if (selected_slot >= slots.size()) {
        selected_slot = 0;
      }
    }

    void handle_input() {
      if (!gui::input::ctrl()) {
        uint32_t n;
        if (gui::input::number(n)) {
          if (n < slots.size()) {
            select_slot(n);
          }
        }
      }

      f32_t scroll = gui::input::scroll();
      if (scroll > 0 && !slots.empty()) {
        uint32_t new_slot = (selected_slot + slots.size() - 1) % slots.size();
        select_slot(new_slot);
      }
      else if (scroll < 0 && !slots.empty()) {
        uint32_t new_slot = (selected_slot + 1) % slots.size();
        select_slot(new_slot);
      }
    }

    void select_slot(uint32_t idx) {
      if (idx >= slots.size()) {
        return;
      }
      selected_slot = idx;
      if (on_slot_select) {
        on_slot_select(idx);
      }
    }

    bool add_item(const gameplay::item_t& item, uint32_t amount = 1, int32_t preferred_slot = -1) {
      if (amount == 0) {
        return false;
      }

      if (preferred_slot >= 0 && preferred_slot < static_cast<int32_t>(slots.size())) {
        auto& slot = slots[preferred_slot];
        if (slot.can_add(item.id, item.max_stack, amount)) {
          return slot.add(item.id, item.max_stack, amount);
        }
      }

      for (auto& slot : slots) {
        if (!slot.is_empty() && slot.can_add(item.id, item.max_stack, amount)) {
          return slot.add(item.id, item.max_stack, amount);
        }
      }

      for (auto& slot : slots) {
        if (slot.is_empty() && slot.can_add(item.id, item.max_stack, amount)) {
          return slot.add(item.id, item.max_stack, amount);
        }
      }

      return false;
    }

    void render(const gameplay::gui_theme_t& theme, gui::drag_drop::drag_state_t& drag_state, uint32_t& hovered_secondary_slot) {
      if (slots.empty()) {
        return;
      }

      auto& io = gui::get_io();

      fan::vec2 window_size = io.DisplaySize;
      f32_t padding_x = 16;
      f32_t padding_y = 16;
      f32_t total_width = slots.size() * (slot_size.x + slot_padding.x) - slot_padding.x;
      f32_t window_width = total_width + padding_x * 2;
      f32_t window_height = slot_size.y + padding_y * 2;

      gui::set_next_window_pos(fan::vec2(
        (window_size.x - window_width) * 0.5f,
        window_size.y - window_height - 40
      ));
      gui::set_next_window_size(fan::vec2(window_width, window_height));

      gui::push_style_var(gui::style_var_window_border_size, 4.0f);
      gui::push_style_var(gui::style_var_window_rounding, 8.0f);
      gui::push_style_color(gui::col_window_bg, theme.panel_bg);
      gui::push_style_color(gui::col_border, theme.panel_border);

      if (gui::begin(
        "Hotbar",
        nullptr,
        gui::window_flags_no_title_bar |
        gui::window_flags_no_resize |
        gui::window_flags_no_move |
        gui::window_flags_no_scrollbar |
        gui::window_flags_no_scroll_with_mouse
      )) {
        if (!drag_state.active) {
          hovered_secondary_slot = UINT32_MAX;
        }

        gui::set_cursor_pos(fan::vec2(padding_x, padding_y));

        slot_layout_t layout{
          .slot_size = slot_size,
          .slot_padding = slot_padding,
          .border_thickness = 2.0f,
          .corner_rounding = 0.0f,
          .columns = static_cast<uint32_t>(slots.size()),
          .horizontal = true
        };

        slot_callbacks_t callbacks {
          .on_use = on_item_use,
          .is_secondary = true
        };

        slot_visual_state_t visual{
          .selected_slot = selected_slot,
          .show_selection = true
        };

        hovered_slot = render_slot_grid(
          slots, 0, static_cast<uint32_t>(slots.size()),
          layout, theme, drag_state, callbacks, visual
        );

        hovered_secondary_slot = hovered_slot;

        drag_drop::render_visual(theme, drag_state);
      }
      gui::end();

      gui::pop_style_color(2);
      gui::pop_style_var(2);
    }

    bool consume_slot(uint32_t slot_index, item_use_cb_t use_cb) {
      auto& slot = slots[slot_index];
      if (slot.is_empty()) {
        return false;
      }

      auto& reg = fan::graphics::gameplay::items::get_registry();
      auto* def = reg.get_definition(*slot.id);
      if (!def) {
        return false;
      }

      gameplay::item_t temp = reg.create_item(def->id);
      temp.stack_size = *slot.stack_size;

      if (use_cb) {
        use_cb(slot_index, temp);
      }

      *slot.stack_size -= 1;
      if (*slot.stack_size == 0) {
        slot.id.reset();
        slot.stack_size.reset();
      }

      return true;
    }

    bool try_drop_here(uint32_t index, gui::drag_drop::drag_state_t& drag_state) {
      auto& dst = slots[index];
      gui::drag_drop::apply_to_slot(drag_state, dst);
      return true;
    }

    uint32_t get_hovered_slot() const {
      return hovered_slot;
    }

    uint32_t hovered_slot = UINT32_MAX;
    std::vector<gameplay::item_slot_t> slots;
    uint32_t selected_slot = 0;
    fan::vec2 slot_size = fan::vec2(64, 64);
    fan::vec2 slot_padding = fan::vec2(8, 8);
    slot_select_cb_t on_slot_select = nullptr;
    item_use_cb_t on_item_use = nullptr;
  };
}