#include <fan/utils/utility.h>
#include <box2d/box2d.h>

import std;
import fan;

using namespace fan::graphics;

#include <fan/graphics/tilemap_helpers.h>
#include <fan/graphics/entity/enemy.h>

#include "items.h"

namespace actions {
  static constexpr const char* drink_potion = "Drink Potion";
  static constexpr const char* interact = "Interact";
  static constexpr const char* toggle_inventory = "Toggle inventory";
}

#include "pile.h"

void register_inputs() {
  pile->engine.get_input_action().insert_or_assign({fan::key_r, fan::gamepad_x}, actions::drink_potion);
  pile->engine.get_input_action().insert_or_assign({fan::key_e, fan::gamepad_y}, actions::interact);
  pile->engine.get_input_action().insert_or_assign({fan::key_tab}, actions::toggle_inventory);
}

int main(){
  pile = (pile_t*)std::malloc(sizeof(pile_t));
  std::construct_at(pile);

  register_inputs();

  ((fan::graphics::gui::settings_menu_t*)pile->engine.get_settings_menu())->keybind_menu.refresh_input_actions();

  pile->engine.get_console().commands.add("set_checkpoint", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    if (args.size() != 1) {
      pile->engine.get_console().commands.print_invalid_arg_count();
      return;
    }
    pile->checkpoint_system.set_checkpoint(std::stoi(args[0]));
    pile->player.checkpoint_position = 0;
    pile->get_level().reload_map();
  });

  pile->engine.loop();
}