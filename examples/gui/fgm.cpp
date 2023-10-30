#include fan_pch

#include _FAN_PATH(graphics/gui/fgm.h)

int main() {
  loco_t loco;

  fgm_t fgm("TexturePack");
  fgm.fin("file.fgm");

  loco.loop([&] {

  });

  return 0;
}
