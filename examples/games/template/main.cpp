#include <fan/types/types.h>
#include <fan/math/math.h>

#include <string>
#include <filesystem>

import fan;
import fan.graphics.gui.tilemap_editor.renderer;

//fan_track_allocations();

std::string current_path = [] {
  std::string dir_path = __FILE__;
  return dir_path.substr(0, dir_path.rfind((char)std::filesystem::path::preferred_separator)) + 
    (char)std::filesystem::path::preferred_separator;
  }();

#include "pile.h"

int main() {
  pile.loco.clear_color = 0;
  pile.loco.lighting.ambient = 1;
  pile.player.player.impulse = 3;
  pile.player.player.force = 15;
  pile.player.player.max_speed = 270;

  pile.loco.loop([&] {

  });
}