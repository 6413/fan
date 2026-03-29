module;

#if defined(FAN_AUDIO)
  #include <fan/utility.h>
#endif

module fan.graphics.audio_subsystem;

import fan.utility;

namespace fan::graphics {
  void audio_subsystem_t::init() {
#if defined(FAN_AUDIO)
    if (system_audio.Open() != 0) {
      fan::throw_error_impl("failed to open fan audio");
    }
    audio.bind(&system_audio);
    fan::audio::piece_hover.open_piece("audio/hover.sac", 0);
    fan::audio::piece_click.open_piece("audio/click.sac", 0);
    fan::audio::gaudio() = &audio;
#endif
  }

  void audio_subsystem_t::destroy() {
#if defined(FAN_AUDIO)
    audio.unbind();
    system_audio.Close();
#endif
  }
}