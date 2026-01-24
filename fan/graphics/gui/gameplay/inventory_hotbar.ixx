module;

#if defined(FAN_GUI)

#include <cstdint>
#include <algorithm>
#include <string>

#endif

export module fan.graphics.gui.inventory_hotbar;

#if defined(FAN_GUI)

import fan.types.vector;
import fan.graphics.gui.base;
export import fan.graphics.gui.drag_drop;
export import fan.graphics.gui.hotbar;
export import fan.graphics.gui.inventory;
import fan.graphics.gameplay.items;

export namespace fan::graphics::gui {
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

#endif