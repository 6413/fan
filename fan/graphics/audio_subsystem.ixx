module;

#if defined(FAN_AUDIO)
#endif

export module fan.graphics.audio_subsystem;

import std;

#if defined(FAN_AUDIO)
  export import fan.audio;

  export namespace fan::graphics {
    struct audio_subsystem_t {
      void init();
      void destroy();

      fan::system_audio_t system_audio;
      fan::audio_t audio;
    };
  }
#endif