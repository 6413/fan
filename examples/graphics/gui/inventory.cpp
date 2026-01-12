import fan;
import fan.graphics.gui.inventory;

using namespace fan::graphics;

int main() {
  engine_t engine;

  gui::gui_theme_t theme;
  theme.panel_bg = fan::color(0.08f, 0.08f, 0.1f, 0.95f);
  theme.panel_border = fan::color(0.5f, 0.5f, 0.6f, 1.f);
  theme.panel_corner_accent = fan::color(0.6f, 0.65f, 0.7f, 1.f);
  theme.slot_bg = fan::color(0.15f, 0.15f, 0.17f, 0.95f);
  theme.slot_bg_hover = fan::color(0.25f, 0.25f, 0.28f, 0.95f);
  theme.slot_border = fan::color(0.4f, 0.4f, 0.45f, 1.f);

  gui::hotbar_t hotbar;
  hotbar.create(9);
  hotbar.theme = &theme;
  hotbar.on_item_use = [](uint32_t slot, const gui::item_t& item) {
  };

  gui::inventory_t inventory;
  inventory.create(32, 8);
  inventory.visible = true;
  inventory.style.theme = &theme;

  engine.loop([&] {
    hotbar.handle_input();
    gui::render_theme_editor(theme);
    hotbar.render();
    if (inventory.visible) {
      inventory.render();
    }
  });
}
