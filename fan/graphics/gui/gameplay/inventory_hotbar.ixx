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
    if (!drag_state.active) {
      return;
    }
    secondary_drop_to_slot(slot_index);
  }

  void inventory_t::secondary_drop_to_slot(uint32_t slot_index) {
    if (!drag_state.active) {
      return;
    }
    if (!secondary) {
      return;
    }
    if (slot_index >= secondary->slots.size()) {
      return;
    }
    auto& dst_slot = secondary->slots[slot_index];
    drag_drop::apply_to_slot(drag_state, dst_slot);
  }

  void inventory_t::secondary_return_to_source() {
    if (!drag_state.active) {
      return;
    }
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
 /* bool move_item_to_hotbar(uint32_t inv_slot, hotbar_t& hotbar, uint32_t hotbar_slot) {
    auto& src = slots[inv_slot];
    auto& dst = hotbar.slots[hotbar_slot];

    if (src.is_empty()) return false;

    dst.id = src.id;
    dst.stack_size = src.stack_size;
    src.clear();
    return true;
  }*/
}