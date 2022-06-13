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
