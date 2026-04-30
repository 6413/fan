module;

#if defined(FAN_GUI)
#endif

export module fan.graphics.gui.gameplay.equipment;

import std;

#if defined(FAN_GUI)
import fan.types.vector;
import fan.graphics.gameplay.types;
import fan.graphics.gui.drag_drop;
import fan.graphics.gui.inventory;

export namespace fan::graphics::gui::gameplay {

  struct equipment_t {
    // sword, shield
    std::array<fan::graphics::gameplay::item_slot_t, 2> slots;

    fan::vec2 slot_size = fan::vec2(81, 81);
    fan::vec2 slot_padding = fan::vec2(12, 12);
    uint32_t hovered_slot = (uint32_t)-1;

    using equip_cb_t = void(*)(uint32_t slot_index, const fan::graphics::gameplay::item_t&);
    equip_cb_t on_equip = nullptr;

    fan::vec2 get_slot_size() const;
    fan::vec2 get_total_size() const;
    bool can_equip(uint32_t slot_index, uint32_t id) const;
    bool try_place_item(uint32_t slot_index, const fan::graphics::gameplay::item_t& item);
    void render_inside_inventory(const fan::graphics::gameplay::gui_theme_t& theme, gui::drag_drop::drag_state_t& drag_state, fan::vec2 offset = fan::vec2(0, 0));
    bool try_drop_here(uint32_t index, gui::drag_drop::drag_state_t& drag_state);
    uint32_t get_hovered_slot() const;
    bool has_item(uint32_t id) const;
    bool equip_item(uint32_t inv_slot, inventory_t& inv, uint32_t idx);
  };

}
#endif