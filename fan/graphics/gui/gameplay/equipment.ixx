module;

#if defined(FAN_GUI)

#include <cstdint>
#include <array>

#endif

export module fan.graphics.gui.gameplay.equipment;

#if defined(FAN_GUI)

import fan.types.vector;
import fan.graphics;
import fan.graphics.gui.base;
import fan.graphics.gui.input;
import fan.graphics.gui.drag_drop;
export import fan.graphics.gameplay.types;
export import fan.graphics.gameplay.items;
import fan.graphics.gui.hotbar;
import fan.graphics.gui.inventory;

using namespace fan::graphics;

export namespace fan::graphics::gui::gameplay {

  struct equipment_t {
    // sword, shield
    std::array<fan::graphics::gameplay::item_slot_t, 2> slots;

    fan::vec2 slot_size = fan::vec2(81, 81);
    fan::vec2 slot_padding = fan::vec2(12, 12);

    using equip_cb_t = void(*)(uint32_t slot_index, const fan::graphics::gameplay::item_t&);
    equip_cb_t on_equip = nullptr;

    fan::vec2 get_slot_size() const {
      return slot_size;
    }

    fan::vec2 get_total_size() const {
      return fan::vec2(
        slot_size.x * slots.size() + slot_padding.x * (slots.size() - 1),
        slot_size.y
      );
    }


    bool can_equip(uint32_t slot_index, uint32_t id) const {
      //if (slot_index == 0) return id == items::id_e::sword;
      //if (slot_index == 1) return id == items::id_e::shield;
      return false;
    }

    bool try_place_item(uint32_t slot_index, const fan::graphics::gameplay::item_t& item) {
      if (!can_equip(slot_index, item.id)) {
        return false;
      }

      auto& slot = slots[slot_index];
      slot.id = item.id;
      slot.stack_size = 1;
      //slot.max_stack = 1;

      if (on_equip) {
        on_equip(slot_index, item);
      }

      return true;
    }

    void render_inside_inventory(
      const fan::graphics::gameplay::gui_theme_t& theme,
      gui::drag_drop::drag_state_t& drag_state,
      fan::vec2 offset = fan::vec2(0, 0)
    ) {
      gui::set_cursor_pos(gui::get_cursor_pos() + offset);

      fan::vec2 child_size(
        slot_size.x * slots.size() + slot_padding.x * (slots.size() - 1),
        slot_size.y
      );

      gui::set_next_window_bg_alpha(0.0f);
      gui::begin_child(
        "EquipmentChild",
        child_size + fan::vec2(100, 100),
        false,
        gui::window_flags_no_scrollbar |
        gui::window_flags_no_scroll_with_mouse
      );

      gui::set_cursor_pos(fan::vec2(8, 8));

      for (uint32_t i = 0; i < slots.size(); ++i) {
        gui::push_id(static_cast<int>(i));

        fan::vec2 local_min = gui::get_cursor_pos();
        fan::vec2 local_max = local_min + slot_size;

        fan::vec2 abs_min = gui::get_cursor_screen_pos();
        fan::vec2 abs_max = abs_min + slot_size;

        gui::invisible_button("equip_slot", slot_size);

        bool hovered = gui::input::hover(abs_min, abs_max);
        bool left_pressed = gui::input::left_click();

        if (hovered) hovered_slot = i;

        if (!drag_state.active && left_pressed && hovered && !slots[i].is_empty()) {
          drag_drop::begin_from_slot(drag_state, slots[i], true, i);
        }

        auto* dl = gui::get_window_draw_list();
        gui::slot::background(dl, abs_min, abs_max, theme.slot_bg, 0.0f);
        gui::slot::border(dl, abs_min, abs_max, theme.slot_border, 0.0f, 2.0f);

        if (!slots[i].is_empty()) {
          auto& reg = fan::graphics::gameplay::items::get_registry();
          auto* def = reg.get_definition(*slots[i].id);
          gui::slot::icon(def->icon, abs_min, abs_max, fan::vec2(4, 4));
          gui::slot::tooltip(def->name, hovered && !drag_state.active);
        }

        gui::pop_id();

        gui::set_cursor_pos(
          local_min + fan::vec2(slot_size.x + slot_padding.x * 2, 0)
        );
      }

      gui::end_child();
    }
    bool try_drop_here(uint32_t index, gui::drag_drop::drag_state_t& drag_state) {
      auto& slot = slots[index];
      gui::drag_drop::apply_to_slot(drag_state, slot);
      return true;
    }
    uint32_t get_hovered_slot() const {
      return hovered_slot;
    }
    bool has_item(uint32_t id) const {
      for (auto& slot : slots) {
        if (!slot.is_empty() && *slot.id == id) {
          return true;
        }
      }
      return false;
    }
    bool equip_item(uint32_t inv_slot, inventory_t& inv, uint32_t idx) {
      auto& src = inv.slots[inv_slot];
      auto& dst = slots[idx];

      if (src.is_empty()) return false;

      dst.id = src.id;
      dst.stack_size = src.stack_size;
      //src.set_empty();
      return true;
    }

    uint32_t hovered_slot = (uint32_t)-1;
  };
}

#endif