#include <fan/pch.h>

#include _FAN_PATH(graphics/gui/fgm/fgm.h)

int main() {
  loco_t loco;

  fgm_t fgm;
  fgm.open("texture_packs/tilemap");
  fgm.fin("file.fgm");

  loco.loop([&] {

  });

  return 0;
}
