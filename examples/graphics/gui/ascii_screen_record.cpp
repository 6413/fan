#include <fan/utility.h>

#include <WITCH/WITCH.h>
#include <WITCH/PR/PR.h>
#include <WITCH/MD/SCR/SCR.h>

import fan;
import fan.ascii_renderer;

using namespace fan::graphics;

struct ascii_overlay_t {
  engine_t engine;
  MD_SCR_t mdscr{};
  ascii_renderer_t ascii;

  ascii_overlay_t() {
    if (int ret = MD_SCR_open(&mdscr); ret != 0) {
      fan::throw_error("failed to open screen:" + std::to_string(ret));
    }
  }

  void update() {
    if (uint8_t* buffer = MD_SCR_read(&mdscr)) {
      ascii.render(buffer, mdscr.Geometry.Resolution.x, mdscr.Geometry.Resolution.y, 4, mdscr.Geometry.LineSize);
    }

    gui::begin("controls");
    gui::slider("brightness", &ascii.properties.brightness_multiplier, 0.1f, 3.f);
    gui::slider("font height", &ascii.properties.font_height, 4.f, 20.f);
    gui::text("fps: ", 1.0f / engine.delta_time);
    gui::end();
  }
};

int main() {
  ascii_overlay_t ao;
  ao.engine.loop([&] { 
    ao.update(); 
  });
}