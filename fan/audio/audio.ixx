module;

#include <fan/utility.h> // abort

#include <cstring>

#if defined(FAN_AUDIO)
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
// transform todo remove
#include <opus/opus.h>
#include <WITCH/WITCH.h>
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

import std;

#if defined(FAN_AUDIO)

import fan.print.error;
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
    std::size_t Length
  ) {
    volatile unsigned char* dest = (volatile unsigned char*)Destination;
    for (std::size_t i = 0; i < Length; ++i) {
      dest[i] = (unsigned char)Fill;
    }
    return Destination;
  }
}
#endif

#if defined(FAN_AUDIO)
export namespace fan {
  namespace audio {
    using pcm_format = fan::system_audio_t::pcm_format;

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
      std::unordered_map<std::string, fan::audio_t::SoundPlayID_t> active;
    };

    g_audio_t& gaudio(){
      static g_audio_t g_audio;
      return g_audio;
    }

    inline std::uint8_t pcm_sample_size(std::uint8_t format){
      switch (format) {
        case fan::system_audio_t::pcm_format::int8: return 1;
        case fan::system_audio_t::pcm_format::int16: return 2;
        case fan::system_audio_t::pcm_format::int24: return 3;
        case fan::system_audio_t::pcm_format::int32: return 4;
        case fan::system_audio_t::pcm_format::float64: return 8;
        case fan::system_audio_t::pcm_format::half_float: fan::throw_error_impl("half_float pcm not implemented");
        default: return 4;
      }
    }

    inline void pcm_to_f32(const std::uint8_t* src, f32_t* dst, std::uint64_t sample_count, std::uint8_t format){
      switch (format) {
        case fan::system_audio_t::pcm_format::int8: {
          auto* s = (const std::int8_t*)src;
          for (std::uint64_t i = 0; i < sample_count; ++i) { dst[i] = s[i] / 128.f; }
          break;
        }
        case fan::system_audio_t::pcm_format::int16: {
          auto* s = (const std::int16_t*)src;
          for (std::uint64_t i = 0; i < sample_count; ++i) { dst[i] = s[i] / 32768.f; }
          break;
        }
        case fan::system_audio_t::pcm_format::int24: {
          for (std::uint64_t i = 0; i < sample_count; ++i) {
            std::int32_t v = (src[i * 3] << 8) | (src[i * 3 + 1] << 16) | (src[i * 3 + 2] << 24);
            dst[i] = (v >> 8) / 8388608.f;
          }
          break;
        }
        case fan::system_audio_t::pcm_format::int32: {
          auto* s = (const std::int32_t*)src;
          for (std::uint64_t i = 0; i < sample_count; ++i) { dst[i] = s[i] / 2147483648.f; }
          break;
        }
        case fan::system_audio_t::pcm_format::float64: {
          auto* s = (const f64_t*)src;
          for (std::uint64_t i = 0; i < sample_count; ++i) { dst[i] = (f32_t)s[i]; }
          break;
        }
        case fan::system_audio_t::pcm_format::half_float: {
          fan::throw_error_impl("half_float pcm not implemented");
          break;
        }
        default: {
          std::memcpy(dst, src, sample_count * sizeof(f32_t));
          break;
        }
      }
    }

    using sound_play_id_t = fan::audio_t::SoundPlayID_t;
  }
}

bool is_playing_locked(fan::audio::sound_play_id_t id) {
  TH_lock(&fan::audio::gaudio()->system_audio->Process.PlayInfoListMutex);
  bool active = false;
  if (!fan::audio::gaudio()->system_audio->Process.PlayInfoList.inri(id.nr)) {
    if (!fan::audio::gaudio()->system_audio->Process.PlayInfoList.IsNodeReferenceRecycled(id.nr)) {
      auto node = &fan::audio::gaudio()->system_audio->Process.PlayInfoList[id.nr];
      if (node->unique == id.unique) {
        active = true;
      }
    }
  }
  TH_unlock(&fan::audio::gaudio()->system_audio->Process.PlayInfoListMutex);
  return active;
}

export namespace fan {
  namespace audio {

    inline bool is_playing(sound_play_id_t id) {
      if (id.nr.iic()) { return false; }
      return is_playing_locked(id);
    }
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
          fan::throw_error("failed to open piece:" + path, "with error:", err);
        }
        
        gaudio().cache[path] = *piece;
        return *this;
      }

      bool is_valid(){
        char test_block[sizeof(fan::audio_t::piece_t)];
        memset(test_block, 0, sizeof(fan::audio_t::piece_t));
        return memcmp(&(fan::audio_t::piece_t&)*this, test_block, sizeof(fan::audio_t::piece_t));
      }

      sound_play_id_t play(std::uint32_t group_id = 0, bool loop = false){
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

      void resume(std::uint32_t group_id = 0){
        gaudio()->Resume();
      }

      void pause(std::uint32_t group_id = 0){
        gaudio()->Pause();
      }

      f32_t get_volume(){
        return gaudio()->GetVolume();
      }

      void set_volume(f32_t volume){
        gaudio()->SetVolume(volume);
      }
    };

    struct stream_t : fan::audio_t::stream_t {
      using fan::audio_t::stream_t::stream_t;

      stream_t() : fan::audio_t::stream_t{ nullptr } {}
      stream_t(const fan::audio_t::stream_t& stream)
        : fan::audio_t::stream_t(stream) {}
      stream_t(
        const void* data, std::uint64_t frame_count, std::uint8_t channel_count = 2,
        std::uint8_t format = fan::system_audio_t::pcm_format::float32
      ) : fan::audio_t::stream_t(open_stream(data, frame_count, channel_count, format)) {}

      operator fan::audio_t::stream_t& (){
        return *dynamic_cast<fan::audio_t::stream_t*>(this);
      }

      stream_t open_stream(const void* data, std::uint64_t frame_count, std::uint8_t channel_count, std::uint8_t format){
        fan::audio_t::stream_t* stream = &(fan::audio_t::stream_t&)*this;
        gaudio()->OpenStream(stream);
        stream->_stream->ChannelAmount = channel_count;
        stream->_stream->FrameAmount = frame_count;
        stream->_stream->type = format;

        std::uint64_t byte_count = frame_count * channel_count * pcm_sample_size(format);
        auto pcm = std::make_shared<std::vector<std::uint8_t>>(
          (const std::uint8_t*)data, (const std::uint8_t*)data + byte_count);

        stream->set_buffer_end_cb([pcm, channel_count, format](fan::system_audio_t::Process_t*, f32_t* out, std::uint32_t frames, std::uint64_t offset){
          std::uint64_t byte_offset = offset * channel_count * pcm_sample_size(format);
          
          if (channel_count == 1) {
            std::vector<f32_t> mono(frames);
            pcm_to_f32(pcm->data() + byte_offset, mono.data(), frames, format);
            for (std::uint32_t i = 0; i < frames; ++i) {
              out[i * 2 + 0] = mono[i];
              out[i * 2 + 1] = mono[i];
            }
          } else {
            pcm_to_f32(pcm->data() + byte_offset, out, (std::uint64_t)frames * channel_count, format);
          }
        });

        return *this;
      }

      sound_play_id_t play(std::uint32_t group_id = 0, bool loop = false){
        fan::audio_t::PropertiesSoundPlay_t p{};
        p.Flags.Loop = loop;
        p.GroupID = group_id;
        return gaudio()->StreamPlay(&*this, &p);
      }

      void stop(sound_play_id_t id){
        fan::audio_t::PropertiesSoundStop_t p{};
        p.FadeOutTo = 0;
        gaudio()->SoundStop(id, &p);
      }

      void close(){
        // gaudio()->Close(&*this); // TODO: implement CloseStream in audio.h
      }
    };

    inline void close_stream(stream_t& stream){
      stream.close();
    }

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

    inline sound_play_id_t play(piece_t piece, std::uint32_t group_id = 0, bool loop = false){
      return piece.play(group_id, loop);
    }

    inline sound_play_id_t play(const std::string& path, std::uint32_t group_id = 0, bool loop = false,
      const std::source_location& callers_path = std::source_location::current()) 
    {
      auto id = piece_t(path, 0, callers_path).play(group_id, loop);
      gaudio().active[path] = id;
      return id;
    }

    inline void stop(sound_play_id_t id, f32_t fade_out_seconds = 0.f) {
      fan::audio_t::PropertiesSoundStop_t p {};
      p.FadeOutTo = fade_out_seconds;
      gaudio()->SoundStop(id, &p);
    }

    inline void stop(const std::string& path, f32_t fade_out_seconds = 0.f) {
      auto it = gaudio().active.find(path);
      if (it == gaudio().active.end()) { return; }
      stop(it->second, fade_out_seconds);
      gaudio().active.erase(it);
    }

    inline bool is_playing(const std::string& path) {
      auto it = gaudio().active.find(path);
      if (it == gaudio().active.end()) { return false; }
      return is_playing(it->second);
    }

    inline void resume(std::uint32_t group_id = 0){
      gaudio()->Resume();
    }

    inline void pause(std::uint32_t group_id = 0){
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
      sound_t() = default;
      sound_t(const std::string& path,
        fan::audio_t::PieceFlag::t flags = 0,
        const std::source_location& callers_path = std::source_location::current()
      ) : piece(path, flags, callers_path) {}

      sound_t(const void* data, std::uint64_t frame_count, std::uint8_t channel_count = 2,
        std::uint8_t format = fan::system_audio_t::pcm_format::float32
      ) : stream(data, frame_count, channel_count, format), from_stream(true) {}

      void play(bool loop = false){
        play_id = from_stream ? stream.play(0, loop) : fan::audio::play(piece, 0, loop);
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

      bool is_playing() const {
        return fan::audio::is_playing(play_id);
      }

      fan::audio::piece_t piece;
      fan::audio::stream_t stream;
      bool from_stream = false;
      fan::audio::sound_play_id_t play_id = { 
        []{fan::system_audio_t::_PlayInfoList_NodeReference_t nr; nr.sic(); return nr; }(),
        0
      };
    };

    sound_play_id_t play_once(fan::audio::piece_t piece){
      return fan::audio::play(piece);
    }

    sound_play_id_t play_looped(fan::audio::piece_t piece, std::uint32_t group_id = 0){
      return fan::audio::play(piece, group_id, true);
    }
  }
}
#endif

#endif