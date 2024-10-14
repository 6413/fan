#define fgm_mark
#define fgm_sprite
#define fgm_build_model_maker 
#include _FAN_PATH(graphics/gui/fgm/fgm.h)

struct model_maker_t : fgm_t {

  model_maker_t() {

  }

  void open(const char* texturepack_name) {
    fgm_t::open(texturepack_name);
  }

  void load(const char* filename) {
    fgm_t::fin(filename);
  }
};