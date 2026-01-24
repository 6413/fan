module;

#if defined(FAN_AUDIO)

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
#include <source_location>
#include <type_traits>


#if fan_audio_set_backend == 0
  #include <xaudio2.h>
#elif fan_audio_set_backend == 1
  #include <pulse/simple.h>
#endif

#include <WITCH/FS/FS.h>
#include <WITCH/T/T.h>
#include <WITCH/TH/TH.h>

#endif

export module fan.audio;

#if defined(FAN_AUDIO)

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
  // __m128i __cdecl _mm_loadu_si128(const __m128i* p) {
  //   __m128i result;
  //   memcpy(&result, p, sizeof(__m128i));
  //   return result;
  // }

  // __m128i __cdecl _mm_cmpeq_epi16(__m128i a, __m128i b) {
  //   __m128i result;
  //   short* ap = (short*)&a;
  //   short* bp = (short*)&b;
  //   short* rp = (short*)&result;
  //   for (int i = 0; i < 8; i++) {
  //     rp[i] = (ap[i] == bp[i]) ? 0xFFFF : 0;
  //   }
  //   return result;
  // }

  // int __cdecl _mm_movemask_epi8(__m128i a) {
  //   unsigned char* ap = (unsigned char*)&a;
  //   int result = 0;
  //   for (int i = 0; i < 16; i++) {
  //     if (ap[i] & 0x80) {
  //       result |= (1 << i);
  //     }
  //   }
  //   return result;
  // }
}
#endif

#if defined(FAN_AUDIO)
export namespace fan {
  namespace audio {

    // doesnt use thread_local since audio has own thread
    struct g_audio_t {
      g_audio_t() = default;
      g_audio_t(fan::audio_t* paudio) : audio(paudio) {}

      operator fan::audio_t* (){
        return audio;
      }
      fan::audio_t* operator->(){
        return audio;
      }

      fan::audio_t* audio = nullptr;
      std::unordered_map<std::string, fan::audio_t::piece_t> cache;
    };

    g_audio_t& gaudio(){
      static g_audio_t g_audio;
      return g_audio;
    }

    using sound_play_id_t = fan::audio_t::SoundPlayID_t;

    struct piece_t : fan::audio_t::piece_t {
      using fan::audio_t::piece_t::piece_t;

      piece_t() : fan::audio_t::piece_t{ nullptr } {}
      piece_t(const fan::audio_t::piece_t& piece)
        : fan::audio_t::piece_t(piece) {}
      piece_t(
        const std::string& path,
        fan::audio_t::PieceFlag::t flags = 0,
        const std::source_location& callers_path = std::source_location::current()
      ) : fan::audio_t::piece_t(open_piece(path, flags, callers_path)) {}

      operator fan::audio_t::piece_t& (){
        return *dynamic_cast<fan::audio_t::piece_t*>(this);
      }

      piece_t open_piece(
        const std::string& path,
        fan::audio_t::PieceFlag::t flags = 0,
        const std::source_location& callers_path = std::source_location::current()
      ){
        
        auto it = gaudio().cache.find(path);
        if (it != gaudio().cache.end()) {
          *(fan::audio_t::piece_t*)this = it->second;
          return *this;
        }

        std::string resolved_path = fan::io::file::find_relative_path(path, callers_path).generic_string();
        fan::audio_t::piece_t* piece = &(fan::audio_t::piece_t&)*this;
        sint32_t err = gaudio()->Open(piece, resolved_path, flags);
        if (err != 0) {
          fan::print("failed to open piece:" + path, "with error:", err);
        }
        
        gaudio().cache[path] = *piece;
        return *this;
      }

      bool is_valid(){
        char test_block[sizeof(fan::audio_t::piece_t)];
        memset(test_block, 0, sizeof(fan::audio_t::piece_t));
        return memcmp(&(fan::audio_t::piece_t&)*this, test_block, sizeof(fan::audio_t::piece_t));
      }

      sound_play_id_t play(uint32_t group_id = 0, bool loop = false){
        fan::audio_t::PropertiesSoundPlay_t p{};
        p.Flags.Loop = loop;
        p.GroupID = 0;
        return gaudio()->SoundPlay(&*this, &p);
      }

      void stop(sound_play_id_t id){
        fan::audio_t::PropertiesSoundStop_t p{};
        p.FadeOutTo = 0;
        gaudio()->SoundStop(id, &p);
      }

      void resume(uint32_t group_id = 0){
        gaudio()->Resume();
      }

      void pause(uint32_t group_id = 0){
        gaudio()->Pause();
      }

      f32_t get_volume(){
        return gaudio()->GetVolume();
      }

      void set_volume(f32_t volume){
        gaudio()->SetVolume(volume);
      }
    };

    piece_t piece_invalid;

    inline piece_t open_piece(
      const std::string& path,
      fan::audio_t::PieceFlag::t flags = 0,
      const std::source_location& callers_path = std::source_location::current()
    ){
      return piece_t(path, flags, callers_path);
    }

    inline bool is_piece_valid(piece_t piece){
      return piece.is_valid();
    }

    inline sound_play_id_t play(piece_t piece, uint32_t group_id = 0, bool loop = false){
      return piece.play(group_id, loop);
    }

    inline void stop(sound_play_id_t id){
      fan::audio_t::PropertiesSoundStop_t p{};
      p.FadeOutTo = 0;
      gaudio()->SoundStop(id, &p);
    }

    inline void resume(uint32_t group_id = 0){
      gaudio()->Resume();
    }

    inline void pause(uint32_t group_id = 0){
      gaudio()->Pause();
    }

    inline f32_t get_volume(){
      return gaudio()->GetVolume();
    }

    inline void set_volume(f32_t volume){
      gaudio()->SetVolume(volume);
    }

    fan::audio::piece_t piece_hover, piece_click;

    struct sound_t {
      sound_t(const std::string& path,
        fan::audio_t::PieceFlag::t flags = 0,
        const std::source_location& callers_path = std::source_location::current()
      ) : piece(path, flags, callers_path) {}

      void play(bool loop = false){
        play_id = fan::audio::play(piece, 0, loop);
      }

      void play_once(){
        play(false);
      }

      void play_looped(){
        play(true);
      }

      void stop(){
        fan::audio::stop(play_id);
      }

      fan::audio::piece_t piece;
      fan::audio::sound_play_id_t play_id;
    };

    sound_play_id_t play_once(fan::audio::piece_t piece){
      return fan::audio::play(piece);
    }

    sound_play_id_t play_looped(fan::audio::piece_t piece, uint32_t group_id = 0){
      return fan::audio::play(piece, group_id, true);
    }
  }
}
#endif

#endif