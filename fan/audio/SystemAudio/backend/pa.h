system_audio_t *system_audio(){
  return OFFSETLESS(this, system_audio_t, Out);
}

f32_t InternalVolume = 1;
f32_t Volume = 1;

TH_id_t thid;
pa_simple *pas;

static void *_thread_func(void *p) {
  auto system_audio = (system_audio_t *)p;
  auto This = &system_audio->Out;

  while(1){
    f32_t frames[_constants::CallFrameCount * _constants::ChannelAmount] = {0};

    system_audio->Process._DataCallback(frames);

    f32_t Volume = This->Volume * This->InternalVolume;

    for(uint32_t i = 0; i < _constants::CallFrameCount * _constants::ChannelAmount; i++){
      frames[i] *= Volume;
    }

    int err;
    int ret = pa_simple_write(This->pas, frames, _constants::CallFrameCount * sizeof(f32_t) * _constants::ChannelAmount, &err);
    if(ret != 0){
      fan::throw_error("pa", __LINE__);
    }
  }

  return 0;
}

sint32_t Open(){
  pa_sample_spec ss;

  ss.format = PA_SAMPLE_FLOAT32;
  ss.channels = _constants::ChannelAmount;
  ss.rate = _constants::opus_decode_sample_rate;

  pas = pa_simple_new(
    NULL,
    "Fooapp", /* TODO get application name */
    PA_STREAM_PLAYBACK,
    NULL,
    "Music", /* TODO get audio description */
    &ss,
    NULL,
    NULL,
    NULL);

  if(pas == NULL){
    fan::throw_error("pa", __LINE__);
  }

  this->thid = TH_open((void *)_thread_func, system_audio());

  return 0;
}
void Close(){
  TH_close_block(this->thid);
  pa_simple_free(pas);
}

void SetVolume(f32_t Volume) {
  __atomic_store(&this->Volume, &Volume, __ATOMIC_RELAXED);
}
f32_t GetVolume() {
  f32_t r;
  __atomic_store(&r, &this->Volume, __ATOMIC_RELAXED);
  return r;
}

void Pause() {
  fan::throw_error("TODO not yet");
}
void Resume() {
  fan::throw_error("TODO not yet");
}
