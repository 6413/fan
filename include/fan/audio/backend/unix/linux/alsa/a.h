#define MA_NO_DECODING
#define MA_NO_ENCODING
#define MINIAUDIO_IMPLEMENTATION
#include "lib.h"

struct audio_t {
  _audio_common_t common;

  ma_context context;
  ma_device device;
};

namespace {
  void BackendDataCallback(ma_device* Device, void* Output, const void* Input, ma_uint32 FrameCount) {
    #if fan_debug >= 0
      if (FrameCount != _constants::CallFrameCount) {
        fan::throw_error("fan_debug");
      }
    #endif
    audio_t *audio = (audio_t *)Device->pUserData;
    _DataCallback(&audio->common, (f32_t *)Output);
  }
}

void audio_close(audio_t* audio) {
  ma_device_uninit(&audio->device);
  ma_context_uninit(&audio->context);

  _audio_common_close(&audio->common);
}
void audio_open(audio_t *audio, uint32_t GroupAmount) {
  _audio_common_open(&audio->common, GroupAmount);

  ma_result r;
  if ((r = ma_context_init(NULL, 0, NULL, &audio->context)) != MA_SUCCESS) {
    fan::throw_error("error" + ::std::to_string(r));
  }

  ma_device_config config = ma_device_config_init(ma_device_type_playback);
  config.playback.format = ma_format_f32;
  config.playback.channels = _constants::ChannelAmount;
  config.sampleRate = _constants::opus_decode_sample_rate;
  config.dataCallback = BackendDataCallback;
  config.pUserData = audio;
  config.periodSizeInFrames = _constants::CallFrameCount;

  if ((r = ma_device_init(&audio->context, &config, &audio->device)) != MA_SUCCESS) {
    fan::throw_error("ma_device_init" + ::std::to_string(r));
  }
}

void audio_stop(audio_t* audio) {
  if (ma_device_stop(&audio->device) != MA_SUCCESS) {
    fan::throw_error("audio_stop");
  }
}
void audio_start(audio_t* audio) {
  if (ma_device_start(&audio->device) != MA_SUCCESS) {
    fan::throw_error("audio_start");
  }
}

void audio_stop_group(audio_t* audio, uint32_t GroupID) {

}

void audio_set_volume(audio_t* audio, f32_t Volume) {
  if (ma_device_set_master_volume(&audio->device, Volume) != MA_SUCCESS) {
    fan::throw_error("audio_set_volume");
  }
}
f32_t audio_get_volume(audio_t* audio) {
  f32_t Volume;
  if (ma_device_get_master_volume(&audio->device, &Volume) != MA_SUCCESS) {
    fan::throw_error("audio_get_volume");
  }
  return Volume;
}
