system_audio_t *system_audio(){
  return OFFSETLESS(this, system_audio_t, Out);
}

f32_t InternalVolume = 1;
f32_t Volume = 1;

IXAudio2 *ctx = NULL;
IXAudio2MasteringVoice *MasterVoice = NULL;
IXAudio2SourceVoice *SourceVoice = NULL;

static constexpr int buffer_count = 2;

f32_t frames[buffer_count][_constants::CallFrameCount * _constants::ChannelAmount];
uint8_t framesi = 0;
struct xacb_t : IXAudio2VoiceCallback{
  void __stdcall OnStreamEnd() { }

  void __stdcall OnVoiceProcessingPassEnd() { }
  void __stdcall OnVoiceProcessingPassStart(UINT32 SamplesRequired) { }
  void __stdcall OnBufferEnd(void* p) {
    auto system_audio = (system_audio_t*)p;
    auto This = &system_audio->Out;

    f32_t* frames = This->frames[This->framesi];
    __builtin_memset(frames, 0, sizeof(This->frames[0]));

    system_audio->Process._DataCallback(frames);

    //system_audio->Process._DataCallbackPcm(frames);


    f32_t Volume = This->Volume * This->InternalVolume;
    for (uint32_t i = 0; i < _constants::CallFrameCount * _constants::ChannelAmount; i++) {
      frames[i] *= Volume;
    }

    XAUDIO2_BUFFER xabuf = { 0 };
    xabuf.AudioBytes = _constants::CallFrameCount * _constants::ChannelAmount * sizeof(f32_t);
    xabuf.pAudioData = (uint8_t*)frames;
    xabuf.pContext = p;

    HRESULT hr = This->SourceVoice->SubmitSourceBuffer(&xabuf);
    if (FAILED(hr)) {
      fan::throw_error("xaudio2", __LINE__);
    }

    This->framesi++;
    This->framesi %= buffer_count;
  }


  void __stdcall OnBufferStart(void* pBufferContext) { }
  void __stdcall OnLoopEnd(void* pBufferContext) { }
  void __stdcall OnVoiceError(void* pBufferContext, HRESULT Error) { }
};

sint32_t Open(){
  HRESULT hr;

  hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
  if(FAILED(hr)){
    fan::throw_error("xaudio2", __LINE__);
    return 1;
  }

  hr = XAudio2Create(&this->ctx, 0, XAUDIO2_DEFAULT_PROCESSOR);
  if(FAILED(hr)){
    fan::throw_error("xaudio2", __LINE__);
    return 1;
  }

  hr = this->ctx->CreateMasteringVoice(&this->MasterVoice);
  if(FAILED(hr)){
    fan::throw_error("xaudio2", __LINE__);
    return 1;
  }

  WAVEFORMATEX waveFormat;
  waveFormat.wFormatTag = WAVE_FORMAT_IEEE_FLOAT;
  waveFormat.nChannels = _constants::ChannelAmount;
  waveFormat.nSamplesPerSec = _constants::opus_decode_sample_rate;
  waveFormat.wBitsPerSample = 32;
  waveFormat.nBlockAlign = waveFormat.nChannels * waveFormat.wBitsPerSample / 8;
  waveFormat.nAvgBytesPerSec = waveFormat.nSamplesPerSec * waveFormat.nBlockAlign;
  waveFormat.cbSize = 0;

  static xacb_t xacb;
  hr = this->ctx->CreateSourceVoice(
    &this->SourceVoice,
    &waveFormat,
    0,
    XAUDIO2_DEFAULT_FREQ_RATIO,
    &xacb
  );
  if(FAILED(hr)){
    fan::throw_error("xaudio2", __LINE__);
    return 1;
  }

  this->SourceVoice->Start(0);

  for(uint8_t i = 0; i < buffer_count; i++){
    __builtin_memset(frames[framesi], 0, sizeof(frames[0]));

    XAUDIO2_BUFFER xabuf = {0};
    xabuf.AudioBytes = _constants::CallFrameCount * _constants::ChannelAmount * sizeof(f32_t);
    xabuf.pAudioData = (uint8_t *)frames[framesi];
    xabuf.pContext = (void *)system_audio();

    hr = this->SourceVoice->SubmitSourceBuffer(&xabuf);
    if(FAILED(hr)){
      fan::throw_error("xaudio2", __LINE__);
    }

    framesi++;
    framesi = framesi % buffer_count;
  }

  return 0;
}
void Close(){
  this->SourceVoice->DestroyVoice();
  this->MasterVoice->DestroyVoice();
  this->ctx->Release();
  CoUninitialize();
}

void SetVolume(f32_t Volume) {
  __atomic_store_n(&this->Volume, Volume, __ATOMIC_RELAXED);
}
f32_t GetVolume() {
  return __atomic_load_n(&this->Volume, __ATOMIC_RELAXED);
}

void Pause() {
  this->SourceVoice->Stop(0);
}
void Resume() {
  this->SourceVoice->Start(0);
}