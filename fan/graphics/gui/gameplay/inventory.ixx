module;

#include <optional>

export module fan.graphics.gui.inventory;

import fan.graphics;
import fan.graphics.gui.base;
import fan.graphics.gui.input;
import fan.graphics.gui.drag_drop;
import fan.graphics.gui.slot_renderer;
import fan.graphics.gameplay.items;
export import fan.graphics.gameplay.types;

using namespace fan::graphics;

export namespace fan::graphics::gui {

  struct hotbar_t;

  struct inventory_style_t {
    fan::vec2 slot_size = fan::vec2(81, 81);
    fan::vec2 slot_padding = fan::vec2(12, 12);
    f32_t border_thickness = 3;
    f32_t corner_rounding = 8;
    f32_t panel_border_thickness = 4;
    f32_t panel_corner_rounding = 12;
    f32_t padding_left = 32;
    f32_t padding_right = 32;
    f32_t padding_top = 399;
    f32_t padding_bottom = 20;
    gameplay::gui_theme_t theme;
  };

  struct inventory_t {
    using slot_click_cb_t = void(*)(uint32_t, const gameplay::item_slot_t&);
    using item_use_cb_t = void(*)(uint32_t, const gameplay::item_t&);

    void create(uint32_t slot_count, uint32_t cols = 8) {
      slots.resize(slot_count);
      columns = cols;
    }

    bool add_item(const gameplay::item_t& item, uint32_t amount = 1) {
      if (amount == 0) {
        return false;
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

    bool remove_item(uint32_t id, uint32_t amount = 1) {
      if (amount == 0) {
        return false;
      }
      for (auto& slot : slots) {
        if (!slot.is_empty() && *slot.id == id) {
          return slot.remove(amount);
        }
      }
      return false;
    }

    std::optional<uint32_t> find_item(uint32_t id) const {
      for (uint32_t i = 0; i < slots.size(); ++i) {
        if (!slots[i].is_empty() && *slots[i].id == id) {
          return i;
        }
      }
      return std::nullopt;
    }

    fan::vec2 get_content_size() const {
      uint32_t rows = (slots.size() + columns - 1) / columns;
      return fan::vec2(
        columns * (style.slot_size.x + style.slot_padding.x) - style.slot_padding.x,
        rows * (style.slot_size.y + style.slot_padding.y) - style.slot_padding.y
      );
    }

    fan::vec2 get_total_size() const {
      fan::vec2 content = get_content_size();
      return fan::vec2(
        content.x + style.padding_left + style.padding_right,
        content.y + style.padding_top + style.padding_bottom
      );
    }

    fan::vec2 get_slot_grid_origin() const {
      return fan::vec2(style.padding_left, style.padding_top);
    }

    template <typename inside_window_cb_t = std::nullptr_t>
    void render(inside_window_cb_t inside_window_cb = nullptr) {
      if (!visible) {
        return;
      }

      auto& io = gui::get_io();

      if (!drag_state.active) {
        hovered_inventory_slot = UINT32_MAX;
        hovered_secondary_slot = UINT32_MAX;
      }

      fan::vec2 window_size = io.DisplaySize;
      fan::vec2 total_size = get_total_size();

      gui::set_next_window_pos(fan::vec2(
        (window_size.x - total_size.x) * 0.5f,
        (window_size.y - total_size.y) * 0.5f
      ));
      gui::set_next_window_size(total_size);

      gui::push_style_color(gui::col_window_bg, style.theme.panel_bg);
      gui::push_style_var(gui::style_var_window_rounding, style.panel_corner_rounding);
      gui::push_style_var(gui::style_var_window_border_size, style.panel_border_thickness);
      gui::push_style_color(gui::col_border, style.theme.panel_border);

      if (gui::begin(
        "Inventory",
        &visible,
        gui::window_flags_no_title_bar |
        gui::window_flags_no_resize |
        gui::window_flags_no_move |
        gui::window_flags_no_saved_settings |
        gui::window_flags_no_scrollbar |
        gui::window_flags_no_scroll_with_mouse |
        gui::window_flags_override_input
      )) {
        gui::set_cursor_pos(fan::vec2(style.padding_left, style.padding_top));

        slot_layout_t layout {
          .slot_size = style.slot_size,
          .slot_padding = style.slot_padding,
          .border_thickness = style.border_thickness,
          .corner_rounding = style.corner_rounding,
          .columns = columns,
          .horizontal = false
        };

        slot_callbacks_t callbacks {
          .on_use = on_item_use,
          .is_secondary = false,
          .enable_ctrl_transfer = secondary != nullptr
        };

        hovered_inventory_slot = render_slot_grid(
          slots, 0, static_cast<uint32_t>(slots.size()),
          layout, style.theme, drag_state, callbacks
        );

        if (hovered_inventory_slot != UINT32_MAX && secondary && !drag_state.active &&
          !slots[hovered_inventory_slot].is_empty() && gui::input::ctrl()) {
          uint32_t n;
          if (gui::input::number(n)) {
            transfer_to_secondary_exact(hovered_inventory_slot, n);
          }
        }

        drag_drop::render_visual(style.theme, drag_state);

        if constexpr (!std::is_same_v<inside_window_cb_t, std::nullptr_t>) {
          inside_window_cb();
        }
      }
      gui::end();

      gui::pop_style_color(2);
      gui::pop_style_var(2);

      if (drag_state.active && gui::input::left_released()) {
        uint32_t src = drag_state.slot_index;
        uint32_t dst = hovered_inventory_slot;

        if (dst == src) {
          auto& slot = slots[src];
          slot.id = drag_state.id;
          slot.stack_size = drag_state.stack_size;
          drag_state.active = false;
          return;
        }

        if (hovered_inventory_slot != UINT32_MAX) {
          drop_to_inventory_slot(hovered_inventory_slot);
        }
        else if (secondary && hovered_secondary_slot != UINT32_MAX) {
          drop_to_secondary_slot(hovered_secondary_slot);
        }
        else {
          if (destroy_on_drop_outside) {
            drag_drop::cancel(drag_state);
          }
          else {
            return_drag_to_source();
          }
        }
      }
}

    void drop_to_inventory_slot(uint32_t slot_index) {
      if (!drag_state.active || slot_index >= slots.size()) {
        return;
      }
      auto& dst_slot = slots[slot_index];
      drag_drop::apply_to_slot(drag_state, dst_slot);
    }

    void drop_to_secondary_slot(uint32_t slot_index);
    void return_drag_to_source();
    void transfer_to_secondary_exact(uint32_t inventory_slot_index, uint32_t secondary_slot_index);

    bool try_drop_here(uint32_t slot_index, gui::drag_drop::drag_state_t& drag_state) {
      drop_to_inventory_slot(slot_index);
      return true;
    }

    uint32_t get_hovered_slot() const {
      return hovered_inventory_slot;
    }

    std::vector<gameplay::item_slot_t> slots;
    inventory_style_t style;
    uint32_t columns = 8;
    bool visible = false;
    slot_click_cb_t on_slot_click = nullptr;
    item_use_cb_t on_item_use = nullptr;
    gui::drag_drop::drag_state_t drag_state;
    uint32_t hovered_inventory_slot = UINT32_MAX;
    uint32_t hovered_secondary_slot = UINT32_MAX;
    bool destroy_on_drop_outside = false;
    hotbar_t* secondary = nullptr;
  };
}