#pragma once

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

#include _WITCH_PATH(WITCH.h)

#include _WITCH_PATH(TH/TH.h)

#include _WITCH_PATH(FS/FS.h)
#include _WITCH_PATH(T/T.h)

// transform
#include <algorithm>

#include <opus/opus.h>

#if fan_audio_set_backend == 0
  #include <xaudio2.h>
#elif fan_audio_set_backend == 1
  #include <pulse/simple.h>
#endif

import fan.types.print;
import fan.io.file;

namespace fan {
  struct system_audio_t {
    #include "SystemAudio/CommonTypes.h"
  };
  struct audio_t{
    #include "audio/audio.h"
  };
}
