#include <fan/utility.h>
#include <fan/types/dme.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>
#include <coroutine>
#include <array>
#include <unordered_set>
#include <algorithm>
#include <vector>
#include <fstream>
#include <source_location>
#include <variant>

#include <fan/utility.h>

#include <source_location>
#include <set>
#include <stacktrace>
#include <map>
#include <box2d/box2d.h>
#include <iostream>


import fan;
import fan.graphics.gui.tilemap_editor.renderer;
import fan.graphics.gameplay;
import fan.graphics.spatial;

// inlines
#include <fan/graphics/tilemap_helpers.h>
#include <fan/graphics/entity/enemy.h>


namespace actions {
  static constexpr const char* drink_potion = "Drink Potion";
  static constexpr const char* interact = "Interact";
}
////
#include "pile.h"

int main() {
  //fan::heap_profiler_t::instance().enabled = true;
  pile = (pile_t*)std::malloc(sizeof(pile_t));
  std::construct_at(pile);
  
  pile->engine.input_action.insert_or_assign({fan::key_r, fan::gamepad_x}, actions::drink_potion);
  pile->engine.input_action.insert_or_assign({fan::key_e, fan::gamepad_y}, actions::interact);

  pile->engine.settings_menu.keybind_menu.refresh_input_actions();

  pile->engine.console.commands.add("set_checkpoint", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    if (args.size() != 1) {
      pile->engine.console.commands.print_invalid_arg_count();
      return;
    }
    pile->checkpoint_system.set_checkpoint(std::stoi(args[0]));
    pile->player.checkpoint_position = 0;
    pile->get_level().reload_map();
  });

  pile->engine.loop([&] {

  });
}