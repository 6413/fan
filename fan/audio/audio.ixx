module;

#include <fan/utility.h> // abort

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

#include <WITCH/FS/FS.h>
#include <WITCH/T/T.h>
#include <WITCH/TH/TH.h>

export module fan.audio;

import fan.print;
import fan.utility;
import fan.io.file;
import fan.math;

export namespace fan {
  struct system_audio_t {
    #include "SystemAudio/CommonTypes.h"
  };
  struct audio_t{
    #include "audio/audio.h"
  };
}

#if defined(fan_platform_windows) && defined(fan_compiler_clang)
extern "C" {
  volatile void* __cdecl RtlSetVolatileMemory(
    volatile void* Destination,
    int Fill,
    size_t Length
  ) {
    volatile unsigned char* dest = (volatile unsigned char*)Destination;
    for (size_t i = 0; i < Length; ++i) {
      dest[i] = (unsigned char)Fill;
    }
    return Destination;
  }
}

extern "C" {
  __m128i __cdecl _mm_loadu_si128(const __m128i* p) {
    __m128i result;
    memcpy(&result, p, sizeof(__m128i));
    return result;
  }

  __m128i __cdecl _mm_cmpeq_epi16(__m128i a, __m128i b) {
    __m128i result;
    short* ap = (short*)&a;
    short* bp = (short*)&b;
    short* rp = (short*)&result;
    for (int i = 0; i < 8; i++) {
      rp[i] = (ap[i] == bp[i]) ? 0xFFFF : 0;
    }
    return result;
  }

  int __cdecl _mm_movemask_epi8(__m128i a) {
    unsigned char* ap = (unsigned char*)&a;
    int result = 0;
    for (int i = 0; i < 16; i++) {
      if (ap[i] & 0x80) {
        result |= (1 << i);
      }
    }
    return result;
  }
}
#endif