module;

#include <cstdint>
#include <vector>
#include <string>
#include <optional>

export module fan.graphics.gui.hotbar;

import fan.graphics;
import fan.graphics.gui.base;
import fan.graphics.gui.input;
import fan.graphics.gui.drag_drop;
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
        fan::vec2 origin = gui::get_cursor_screen_pos();

        for (uint32_t i = 0; i < slots.size(); ++i) {
          fan::vec2 slot_pos = origin + fan::vec2(
            i * (slot_size.x + slot_padding.x),
            0
          );
          gui::set_cursor_screen_pos(slot_pos);

          auto& slot = slots[i];
          fan::vec2 cursor = gui::get_cursor_screen_pos();

          gui::push_id(static_cast<int>(i));

          gui::invisible_button("hotbar_slot", slot_size);

          bool left_pressed = gui::input::left_click();
          bool right_pressed = gui::input::right_click();

          fan::vec2 p_min = cursor;
          fan::vec2 p_max = cursor + slot_size;

          bool hovered = gui::input::hover(p_min, p_max);

          if (hovered) {
            hovered_secondary_slot = i;
            hovered_slot = i;
          }

          handle_clicks(i, slot, hovered, left_pressed, right_pressed, drag_state);

          auto* dl = gui::get_window_draw_list();

          bool is_selected = i == selected_slot;

          fan::color bg_color =
            hovered ? theme.slot_bg_hover :
            theme.slot_bg;

          gui::slot::background(dl, p_min, p_max, bg_color, 0.0f);
          gui::slot::border(dl, p_min, p_max, theme.slot_border, 0.0f, 2.0f);

          if (is_selected) {
            gui::slot::selected_border(dl, p_min, p_max, theme.selected_border_color, 4.0f, 3.0f);
          }

          if (!slot.is_empty()) {
            auto& reg = fan::graphics::gameplay::items::get_registry();
            auto* def = reg.get_definition(*slot.id);

            gui::slot::icon(def->icon, p_min, p_max, fan::vec2(4, 4));
            gui::slot::stack_count(*slot.stack_size, p_min, p_max);
            gui::slot::tooltip(def->name, hovered && !drag_state.active);
          }

          gui::pop_id();
        }

        drag_drop::render_visual(theme, drag_state);
      }
      gui::end();

      gui::pop_style_color(2);
      gui::pop_style_var(2);
    }

    void handle_clicks(
      uint32_t slot_index,
      gameplay::item_slot_t& slot,
      bool hovered,
      bool left_pressed,
      bool right_pressed,
      gui::drag_drop::drag_state_t& drag_state
    ) {
      if (right_pressed && hovered && !slot.is_empty()) {
        consume_slot(slot_index, on_item_use);
        return;
      }

      if (!drag_state.active && left_pressed && hovered && !slot.is_empty()) {
        drag_drop::begin_from_slot(drag_state, slot, true, slot_index);
      }
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