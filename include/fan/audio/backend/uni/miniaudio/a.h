audio_t *audio(){
  return OFFSETLESS(this, audio_t, Out);
}

ma_context context;
ma_device device;

static void _miniaudio_DataCallback(ma_device *Device, void *Output, const void *Input, ma_uint32 FrameCount) {
  audio_t *audio = (audio_t *)Device->pUserData;

  #if fan_debug >= 0
    if (FrameCount != _constants::CallFrameCount) {
      fan::throw_error("fan_debug");
    }
  #endif

  audio->Process._DataCallback((f32_t *)Output);
}

sint32_t Open() {
  ma_result r;
  if ((r = ma_context_init(NULL, 0, NULL, &this->context)) != MA_SUCCESS) {
    fan::throw_error("error" + ::fan::to_string(r));
  }

  ma_device_config config = ma_device_config_init(ma_device_type_playback);
  config.playback.format = ma_format_f32;
  config.playback.channels = _constants::ChannelAmount;
  config.sampleRate = _constants::opus_decode_sample_rate;
  config.dataCallback = _miniaudio_DataCallback;
  config.pUserData = audio();
  config.periodSizeInFrames = _constants::CallFrameCount;

  if ((r = ma_device_init(&this->context, &config, &this->device)) != MA_SUCCESS) {
    fan::throw_error("ma_device_init" + ::fan::to_string(r));
  }

  if (ma_device_start(&this->device) != MA_SUCCESS) {
    fan::throw_error("ma_device_start");
  }

  return 0;
}
void Close() {
  ma_device_uninit(&this->device);
  ma_context_uninit(&this->context);
}

void Pause() {
  fan::throw_error("TODO not yet");
  if (ma_device_stop(&this->device) != MA_SUCCESS) {
    fan::throw_error("ma_device_stop");
  }
}
void Resume() {
  fan::throw_error("TODO not yet");
}

void SetVolume(f32_t Volume) {
  if (ma_device_set_master_volume(&this->device, Volume) != MA_SUCCESS) {
    fan::throw_error("ma_device_set_master_volume");
  }
}
f32_t GetVolume() {
  f32_t Volume;
  if (ma_device_get_master_volume(&this->device, &Volume) != MA_SUCCESS) {
    fan::throw_error("ma_device_get_master_volume");
  }
  return Volume;
}
