module;

#if defined(FAN_AUDIO)
  #include <fan/utility.h>
#endif

export module fan.graphics.audio_subsystem;

#if defined(FAN_AUDIO)
  export import fan.audio;
#endif

export namespace fan::graphics {
  struct audio_subsystem_t {
    void init();
    void destroy();

#if defined(FAN_AUDIO)
    fan::system_audio_t system_audio;
    fan::audio_t audio;
#endif
  };
}