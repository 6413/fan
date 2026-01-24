#include <fan/utility.h>

#include <string>
#include <filesystem>
#include <functional>
#include <vector>

import fan;
import fan.graphics.gui.tilemap_editor.renderer;

//fan_track_allocations();

#include "pile.h"

int main() {
  pile.engine.loop([&] {
    pile.step();
  });
}