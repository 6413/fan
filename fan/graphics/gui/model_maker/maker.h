#define fgm_mark
#define fgm_sprite
#define fgm_build_model_maker 
#include <fan/graphics/gui/fgm/fgm.h>

struct model_maker_t : fgm_t {

  model_maker_t() {

  }

  void open(const char* texturepack_name, const std::wstring& asset_path) {
    fgm_t::open(texturepack_name, asset_path);
  }

  void load(const char* filename) {
    fgm_t::fin(filename);
  }
};