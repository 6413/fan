#define fgm_mark
#define fgm_sprite
#define fgm_build_model_maker 
#include _FAN_PATH(graphics/gui/fgm/fgm.h)

struct model_maker_t : fgm_t {
  static constexpr const char* filename_out = "model.fmm";

  void open(const char* texturepack_name) {
    fgm_t::open(texturepack_name);
    keys_nr = gloco->window.add_keys_callback([this](const auto& d) {
      switch (d.key) {
        case fan::key_q: {
          if (d.state != fan::keyboard_state::press) {
            break;
          }

          fgm_t::fout(filename_out);
          break;
        }
      }
      });
  }
  ~model_maker_t() {
    gloco->window.remove_keys_callback(keys_nr);
  }

  void load(const char* filename) {
    fgm_t::fin(filename);
  }
  fan::window_t::keys_callback_NodeReference_t keys_nr;
};