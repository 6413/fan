#pragma once

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

#include _FAN_PATH(io/file.h)

#include <opus/opus.h>

namespace fan {
  namespace audio {
    #include "CommonTypes.h"

    #include "CommonDefine.h"

    #if fan_audio_set_backend == 0
      #include "backend/uni/miniaudio/a.h"
    #elif fan_audio_set_backend == 1
      #include "backend/unix/linux/alsa/a.h"
    #else
      #error ?
    #endif
  }
}
