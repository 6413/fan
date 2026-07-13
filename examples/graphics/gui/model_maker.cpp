#include <fan/utility.h>

#include <string>
#include <vector>
#include <functional>
#include <filesystem>
#include <cstring>
#include <fstream>

import fan;
import fan.fmt;

import fan.graphics.editor;

int main(int argc, char** argv) {
  {
    //if (argc < 2) {//
    //  fan::throw_error("usage: TexturePackCompiled");
    //}////
    //////////
    //////////
    fan::graphics::engine_t engine;////

    fan::graphics::editor::fgm_t mm;//// 

    mm.open("", "");
    // mm.fin("normal_map_tests.json");

    engine.loop([&] {
      mm.render();
      });
  }

  return 0;
}