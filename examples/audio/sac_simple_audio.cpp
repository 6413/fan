import fan;
#include <fan/graphics/types.h>

// argv[1] == audio/w_voice.sac 
int main(int argc, char** argv) {
  fan::graphics::engine_t engine;

  fan::audio::piece_t piece = fan::audio::open_piece("audio/output.sac");
  uint32_t group_id = 0;
  bool loop = true;
  fan::audio::play(piece, group_id, loop);

  fan::audio::set_volume(0.01);
  f32_t volume = fan::audio::get_volume();

  fan_window_loop {
    fan_graphics_gui_window("audio controls"){
      if (fan::graphics::gui::button("toggle pause")) {
        static int audio_toggle = 0;
        ((audio_toggle++)& 1) == 0 ? fan::audio::pause() : fan::audio::resume();

      }
      if (fan::graphics::gui::drag_float("volume", &volume, 0.01f, 0.0f, 1.0f)) {
        fan::audio::set_volume(volume);
      }
    }
  };

  return 0;
}