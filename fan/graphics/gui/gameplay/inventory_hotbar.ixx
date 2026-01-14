module;

#include <cstdint>
#include <algorithm>
#include <string>

export module fan.graphics.gui.inventory_hotbar;

import fan.graphics.gui.drag_drop;
export import fan.graphics.gui.hotbar;
export import fan.graphics.gui.inventory;
import fan.graphics.gameplay.items;

export namespace fan::graphics::gui {
  void inventory_t::drop_to_secondary_slot(uint32_t slot_index) {
    if (!drag_state.active || !secondary || slot_index >= secondary->slots.size()) {
      return;
    }
    auto& dst_slot = secondary->slots[slot_index];
    drag_drop::apply_to_slot(drag_state, dst_slot);
  }

  void inventory_t::return_drag_to_source() {
    if (!drag_state.active) {
      return;
    }

    if (drag_state.from_secondary) {
      if (!secondary) {
        drag_drop::cancel(drag_state);
        return;
      }
      if (drag_state.slot_index < secondary->slots.size()) {
        auto& slot = secondary->slots[drag_state.slot_index];
        if (slot.is_empty()) {
          slot.id = drag_state.id;
          slot.stack_size = drag_state.stack_size;
        }
        else {
          drag_drop::apply_to_slot(drag_state, slot);
        }
      }
      drag_state.active = false;
    }
    else {
      if (drag_state.slot_index < slots.size()) {
        auto& slot = slots[drag_state.slot_index];
        if (slot.is_empty()) {
          slot.id = drag_state.id;
          slot.stack_size = drag_state.stack_size;
        }
        else {
          drag_drop::apply_to_slot(drag_state, slot);
        }
      }
      drag_state.active = false;
    }
  }

  void inventory_t::transfer_to_secondary_exact(uint32_t inventory_slot_index, uint32_t secondary_slot_index) {
    if (!secondary) {
      return;
    }
    if (inventory_slot_index >= slots.size()) {
      return;
    }
    if (secondary_slot_index >= secondary->slots.size()) {
      return;
    }
    auto& src_slot = slots[inventory_slot_index];
    auto& dst_slot = secondary->slots[secondary_slot_index];
    if (src_slot.is_empty()) {
      return;
    }
    auto& reg = fan::graphics::gameplay::items::get_registry();
    auto* def = reg.get_definition(*src_slot.id);
    if (!def) {
      return;
    }
    if (dst_slot.is_empty()) {
      dst_slot.id = src_slot.id;
      dst_slot.stack_size = src_slot.stack_size;
      src_slot.id.reset();
      src_slot.stack_size.reset();
      return;
    }
    if (*dst_slot.id == *src_slot.id) {
      uint32_t free_space = 0;
      if (*dst_slot.stack_size < def->max_stack) {
        free_space = def->max_stack - *dst_slot.stack_size;
      }
      if (free_space == 0) {
        uint32_t tmp_id = *dst_slot.id;
        uint32_t tmp_stack = *dst_slot.stack_size;
        dst_slot.id = *src_slot.id;
        dst_slot.stack_size = *src_slot.stack_size;
        src_slot.id = tmp_id;
        src_slot.stack_size = tmp_stack;
        return;
      }
      uint32_t to_add = std::min(free_space, *src_slot.stack_size);
      *dst_slot.stack_size += to_add;
      *src_slot.stack_size -= to_add;
      if (*src_slot.stack_size == 0) {
        src_slot.id.reset();
        src_slot.stack_size.reset();
      }
      return;
    }
    uint32_t tmp_id = *dst_slot.id;
    uint32_t tmp_stack = *dst_slot.stack_size;
    dst_slot.id = *src_slot.id;
    dst_slot.stack_size = *src_slot.stack_size;
    src_slot.id = tmp_id;
    src_slot.stack_size = tmp_stack;
  }

  void bind_inventory_hotbar(inventory_t& inventory, hotbar_t& hotbar) {
    inventory.secondary = &hotbar;
  }

  void render_theme_editor(
    gameplay::gui_theme_t& theme,
    inventory_t& inventory,
    hotbar_t& hotbar
  ) {
    if (!gui::begin("Theme Editor")) {
      gui::end();
      return;
    }

    gui::color_edit4("Panel BG", &theme.panel_bg);
    gui::color_edit4("Panel Border", &theme.panel_border);
    gui::color_edit4("Panel Corner Accent", &theme.panel_corner_accent);
    gui::color_edit4("Slot BG", &theme.slot_bg);
    gui::color_edit4("Slot BG Hover", &theme.slot_bg_hover);
    gui::color_edit4("Slot Border", &theme.slot_border);
    gui::color_edit4("Selected Border", &theme.selected_border_color);

    float slot_size = inventory.style.slot_size.x;
    gui::drag("Slot Size", &slot_size);
    inventory.style.slot_size = fan::vec2(slot_size, slot_size);

    gui::drag("Slot Padding", &inventory.style.slot_padding);
    gui::drag("Border Thickness", &inventory.style.border_thickness);
    gui::drag("Corner Rounding", &inventory.style.corner_rounding);
    gui::drag("Panel Border Thickness", &inventory.style.panel_border_thickness);
    gui::drag("Panel Corner Rounding", &inventory.style.panel_corner_rounding);

    gui::drag("Padding Left", &inventory.style.padding_left);
    gui::drag("Padding Right", &inventory.style.padding_right);
    gui::drag("Padding Top", &inventory.style.padding_top);
    gui::drag("Padding Bottom", &inventory.style.padding_bottom);

    gui::end();
  }


}