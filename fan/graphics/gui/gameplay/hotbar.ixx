module;

export module fan.graphics.gui.hotbar;

#if defined (FAN_WINDOW)

import std;

#if defined(FAN_GUI)

import fan.types;
import fan.types.vector;
import fan.graphics.common_context;
import fan.graphics.gui.input;
import fan.graphics.gui.drag_drop;
import fan.graphics.gui.slot_renderer;
import fan.graphics.gameplay.types;
import fan.graphics.gameplay.items;

export namespace fan::graphics::gui {

  struct hotbar_t {
    using slot_select_cb_t = void(*)(std::uint32_t);
    using item_use_cb_t = void(*)(std::uint32_t, const gameplay::item_t&);

    void create(std::uint32_t slot_count = 9);
    void handle_input();
    void select_slot(std::uint32_t idx);
    bool add_item(const gameplay::item_t& item, std::uint32_t amount = 1, std::int32_t preferred_slot = -1);
    void render(const gameplay::gui_theme_t& theme, gui::drag_drop::drag_state_t& drag_state, std::uint32_t& hovered_secondary_slot);
    bool consume_slot(std::uint32_t slot_index, item_use_cb_t use_cb);
    bool try_drop_here(std::uint32_t index, gui::drag_drop::drag_state_t& drag_state);
    std::uint32_t get_hovered_slot() const;

    std::uint32_t hovered_slot = std::numeric_limits<std::uint32_t>::max();
    std::vector<gameplay::item_slot_t> slots;
    std::uint32_t selected_slot = 0;
    fan::vec2 slot_size = fan::vec2(64, 64);
    fan::vec2 slot_padding = fan::vec2(8, 8);
    slot_select_cb_t on_slot_select = nullptr;
    item_use_cb_t on_item_use = nullptr;
  };
}

#endif

#endif