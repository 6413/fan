#include fan_pch

#include _FAN_PATH(graphics/gui/fgm/fgm.h)

int main() {
  loco_t loco;

  fgm_t fgm;
  fgm.open("TexturePack");
  fgm.fin("file.fgm");

  loco.loop([&] {

  });

  return 0;
}
