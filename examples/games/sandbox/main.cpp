#include <fan/utility.h>

#include <string>
#include <filesystem>
#include <functional>
#include <vector>

import fan;
import fan.graphics.gui.tilemap_editor.renderer;

#include "pile.h"

int main() {
  pile.engine.loop([&] {
    pile.step();
  });
}