#define fgm_mark
#define fgm_sprite
#define fgm_build_model_maker 
#include _FAN_PATH(graphics/gui/fgm/fgm.h)

struct model_maker_t : fgm_t {
  static constexpr const char* filename_out = "model.fmm";

  void open(const char* texturepack_name) {
    fgm_t::open(texturepack_name);
    keys_nr = pile->loco.get_window()->add_keys_callback([this](const auto& d) {
      switch (d.key) {
        case fan::key_q: {
          if (d.state != fan::keyboard_state::press) {
            break;
          }

          fgm_t::write_to_file(filename_out);
          fan::print("file saved to", filename_out);
          break;
        }
      }
      });
  }
  ~model_maker_t() {
    pile->loco.get_window()->remove_keys_callback(keys_nr);
  }

  using fgm_t::load;

  void load(const char* path) {
    fgm_t::load();

    fan::string f;
    uint64_t offset = 0;
    if (!fan::io::file::exists(path)) {
      return;
    }
    fan::io::file::read(path, &f);

    if (f.empty()) {
      return;
    }

    // read header
    uint32_t header = fan::read_data<uint32_t>(f, offset);

    read_from_file(path);

   /* fgm_t::iterate_masterpiece([&](auto& d) {
      while (offset < f.size()) {
        iterate_masterpiece([&f, &offset](auto& o) {
          offset += o.from_string(f.substr(offset));
        });
      }
    });*/
  }
  fan::window_t::keys_callback_NodeReference_t keys_nr;
};