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
  pile.engine.loop([&] {

  });
}