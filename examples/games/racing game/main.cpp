#include <fan/utility.h>

#include <string>
#include <filesystem>
#include <fan/graphics/algorithm/astar.h>

import fan;
import fan.graphics.network;
import fan.graphics.gui.tilemap_editor.renderer;

#include <fan/graphics/types.h>

//fan_track_allocations();

#include "pile.h"

std::string shadow_fragment;

using namespace fan::graphics;

int main() {
  bool shader_compiled = true;

  fan::vec3 l_pos = 0;

  pile.engine.loop([&] {
    if (fan::graphics::gui::begin("light properties")) {
      fan_graphics_gui_slider_property(pile.lights.back(), position);
      fan_graphics_gui_slider_property(pile.lights.back(), size);
      fan_graphics_gui_slider_property(pile.lights.back(), color);
      fan::graphics::gui::end();
    }

    fan::graphics::gui::fragment_shader_editor(fan::graphics::shape_type_e::shadow, &shadow_fragment, &shader_compiled);
  });
}