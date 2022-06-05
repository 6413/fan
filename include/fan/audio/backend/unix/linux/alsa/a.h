#include <alsa/asoundlib.h>

struct audio_t {
  _audio_common_t common;

  f32_t Volume;

  TH_id_t thid;
  snd_pcm_t *snd_pcm;
};

void *_audio_thread_func(void *p) {
  audio_t *audio = (audio_t *)p;
  while(1){
    f32_t frames[_constants::CallFrameCount * _constants::ChannelAmount] = {0};

    _DataCallback(&audio->common, frames);

    for(uint32_t i = 0; i < _constants::CallFrameCount * _constants::ChannelAmount; i++){
      frames[i] *= audio->Volume;
    }

    snd_pcm_sframes_t rframes = snd_pcm_writei(audio->snd_pcm, frames, _constants::CallFrameCount);
    if (rframes != _constants::CallFrameCount) {
      switch(rframes){
        case -EPIPE:{
          int r = snd_pcm_recover(audio->snd_pcm, -EPIPE, 0);
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

void audio_close(audio_t* audio) {
  fan::throw_error("TODO not yet");

  _audio_common_close(&audio->common);
}
void audio_open(audio_t *audio, uint32_t GroupAmount) {
  _audio_common_open(&audio->common, GroupAmount);

  audio->Volume = 1;

  int ierr = snd_pcm_open(&audio->snd_pcm, "default", SND_PCM_STREAM_PLAYBACK, 0);
  if(ierr < 0){
    fan::throw_error("a");
  }

  snd_pcm_hw_params_t *params;
  snd_pcm_hw_params_alloca(&params);
  snd_pcm_hw_params_any(audio->snd_pcm, params);

  if((ierr = snd_pcm_hw_params_set_access(audio->snd_pcm, params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0){
    fan::throw_error("a");
  }

  if((ierr = snd_pcm_hw_params_set_format(audio->snd_pcm, params, SND_PCM_FORMAT_FLOAT_LE)) < 0){
    fan::throw_error("a");
  }

  if((ierr = snd_pcm_hw_params_set_channels(audio->snd_pcm, params, _constants::ChannelAmount)) < 0){
    fan::throw_error("a");
  }

  if((ierr = snd_pcm_hw_params_set_rate(audio->snd_pcm, params, _constants::opus_decode_sample_rate, 0)) < 0){
    fan::throw_error("a");
  }

  if((ierr = snd_pcm_hw_params_set_periods(audio->snd_pcm, params, 3, 0)) < 0){
    fan::throw_error("w");
  }

  if((ierr = snd_pcm_hw_params_set_period_size(audio->snd_pcm, params, _constants::CallFrameCount, 0)) < 0){
    fan::throw_error("a");
  }

  if((ierr = snd_pcm_hw_params(audio->snd_pcm, params) < 0)){
    fan::throw_error("a");
  }
}

void audio_stop(audio_t* audio) {
  fan::throw_error("TODO not yet");
}
void audio_start(audio_t* audio) {
  audio->thid = TH_open((void *)_audio_thread_func, audio);
}

void audio_stop_group(audio_t *audio, uint32_t GroupID) {

}

void audio_set_volume(audio_t *audio, f32_t Volume) {
  audio->Volume = Volume;
}
f32_t audio_get_volume(audio_t *audio) {
  return audio->Volume;
}
