#include <fan/utility.h>

#include <cstring>

#include <WITCH/WITCH.h>
#include <WITCH/PR/PR.h>
#include <WITCH/MD/SCR/SCR.h>

import fan;

using namespace fan::graphics;

struct screen_reader_t {
  MD_SCR_t mdscr{};

  screen_reader_t() {
    if (int ret = MD_SCR_open(&mdscr); ret != 0) {
      fan::throw_error("failed to open screen:" + std::to_string(ret));
    }
  }

  ~screen_reader_t() {
    MD_SCR_close(&mdscr);
  }

  uint8_t* read() {
    return MD_SCR_read(&mdscr);
  }
};

int main() {
  engine_t engine;
  screen_reader_t screen;
  image_t image({2560, 1440}, 4, {.format=image_format_e::bgra});
  sprite_t s(engine.whs(), engine.whs(), image);
  engine.loop([&]{
    if (fan::time::every(1.f / 144.f * 1000.f))
    if (uint8_t* buffer = screen.read()) {
      image.reload({buffer, *(fan::vec2ui*)&screen.mdscr.Geometry.Resolution});
    }
  });
}