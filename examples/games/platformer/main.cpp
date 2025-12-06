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

import fan;
import fan.graphics.gui.tilemap_editor.renderer;


#include "pile.h"

int main() {
  //fan::heap_profiler_t::instance().enabled = true;
  pile = (pile_t*)std::malloc(sizeof(pile_t));
  std::construct_at(pile);
  pile->engine.loop();
}