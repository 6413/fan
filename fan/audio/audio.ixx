module;

#include <fan/types/types.h>

#ifndef fan_audio_set_backend
  #if defined(fan_platform_unix)
    #if defined(fan_platform_linux)
      #define fan_audio_set_backend 1
    #else
      #define fan_audio_set_backend 0
    #endif
  #elif defined(fan_platform_windows)
    #define fan_audio_set_backend 0
  #else
    #error ?
  #endif
#endif

#define WITCH_PRE_is_not_allowed

#include <WITCH/WITCH.h>

#include <WITCH/TH/TH.h>

#include <WITCH/FS/FS.h>
#include <WITCH/T/T.h>

// transform todo remove
#include <algorithm>
#include <string>
#include <cmath>
#include <chrono>
#include <functional>
#include <opus/opus.h>
#include <cstring>

#if fan_audio_set_backend == 0
  #include <xaudio2.h>
#elif fan_audio_set_backend == 1
  #include <pulse/simple.h>
#endif

export module fan.audio;

import fan.print;
import fan.io.file;

export namespace fan {
  struct system_audio_t {
    #include "SystemAudio/CommonTypes.h"
  };
  struct audio_t{
    #include "audio/audio.h"
  };
}
