system_audio_t *system_audio(){
  return OFFSETLESS(this, system_audio_t, Out);
}

f32_t Volume;

TH_id_t thid;
snd_pcm_t *snd_pcm;

static void *_thread_func(void *p) {
  auto system_audio = (system_audio_t *)p;
  auto This = &system_audio->Out;

  while(1){
    f32_t frames[_constants::CallFrameCount * _constants::ChannelAmount] = {0};

    system_audio->Process._DataCallback(frames);

    for(uint32_t i = 0; i < _constants::CallFrameCount * _constants::ChannelAmount; i++){
      frames[i] *= This->Volume;
    }

    snd_pcm_sframes_t rframes = snd_pcm_writei(This->snd_pcm, frames, _constants::CallFrameCount);
    if (rframes != _constants::CallFrameCount) {
      switch(rframes){
        case -EPIPE:{
          int r = snd_pcm_recover(This->snd_pcm, -EPIPE, 0);
          if(r != 0){
            fan::throw_error("failed to recover");
          }
          break;
        }
        default:{
          fan::throw_error("snd_pcm_writei");
        }
      }
    }
  }

  return 0;
}

sint32_t Open(){
  this->Volume = 1;

  int ierr = snd_pcm_open(&this->snd_pcm, "default", SND_PCM_STREAM_PLAYBACK, 0);
  if(ierr < 0){
    fan::throw_error("a");
  }

  snd_pcm_hw_params_t *params;
  snd_pcm_hw_params_alloca(&params);
  snd_pcm_hw_params_any(this->snd_pcm, params);

  if((ierr = snd_pcm_hw_params_set_access(this->snd_pcm, params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0){
    fan::throw_error("a");
  }

  if((ierr = snd_pcm_hw_params_set_format(this->snd_pcm, params, SND_PCM_FORMAT_FLOAT_LE)) < 0){
    fan::throw_error("a");
  }

  if((ierr = snd_pcm_hw_params_set_channels(this->snd_pcm, params, _constants::ChannelAmount)) < 0){
    fan::throw_error("a");
  }

  if((ierr = snd_pcm_hw_params_set_rate(this->snd_pcm, params, _constants::opus_decode_sample_rate, 0)) < 0){
    fan::throw_error("a");
  }

  if((ierr = snd_pcm_hw_params_set_periods(this->snd_pcm, params, 3, 0)) < 0){
    fan::throw_error("w");
  }

  if((ierr = snd_pcm_hw_params_set_period_size(this->snd_pcm, params, _constants::CallFrameCount, 0)) < 0){
    fan::throw_error("a");
  }

  if((ierr = snd_pcm_hw_params(this->snd_pcm, params) < 0)){
    fan::throw_error("a");
  }

  this->thid = TH_open((void *)_thread_func, system_audio());

  return 0;
}
void Close(){
  fan::throw_error("TODO not yet");
}

void Pause() {
  fan::throw_error("TODO not yet");
}
void Resume() {
  fan::throw_error("TODO not yet");
}

void SetVolume(f32_t Volume) {
  __atomic_store(&this->Volume, &Volume, __ATOMIC_RELAXED);
}
f32_t GetVolume() {
  f32_t r;
  __atomic_store(&r, &this->Volume, __ATOMIC_RELAXED);
  return r;
}
