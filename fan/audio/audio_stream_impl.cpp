module;

#if defined(FAN_AUDIO_STREAM)
  #include <cstdint>
  #include <cstring>
  #if defined(_WIN32)
    #ifndef NOMINMAX
      #define NOMINMAX
    #endif
  #endif
  #include <opus/opus.h>
  #define MA_NO_DECODING
  #define MA_NO_ENCODING
  #define MA_NO_GENERATION
  #define MA_NO_RESOURCE_MANAGER
  #define MA_NO_NODE_GRAPH
  #define MA_NO_ENGINE
  #define MINIAUDIO_IMPLEMENTATION
  #include <fan/audio/SystemAudio/backend/uni/miniaudio/lib.h>
#endif

module fan.audio.stream;

#if defined(FAN_AUDIO_STREAM)

namespace fan::audio {
  namespace {
    struct device_t {
      ma_context context;
      ma_device device;
      bool has_context = false;
      bool has_device = false;
    };

    void destroy_device(void*& handle) {
      auto* d = (device_t*)handle;
      if (!d) {
        return;
      }
      if (d->has_device) {
        ma_device_uninit(&d->device);
      }
      if (d->has_context) {
        ma_context_uninit(&d->context);
      }
      delete d;
      handle = nullptr;
    }

    ma_device_config make_config(ma_device_type type, ma_device_data_proc proc, void* user) {
      ma_device_config config = ma_device_config_init(type);
      config.playback.format = config.capture.format = ma_format_f32;
      config.playback.channels = config.capture.channels = stream_channels;
      config.sampleRate = stream_sample_rate;
      config.periodSizeInFrames = stream_frame_size / 2;
      config.performanceProfile = ma_performance_profile_low_latency;
      config.dataCallback = proc;
      config.pUserData = user;
      return config;
    }

  #if defined(__linux__)
    bool default_monitor_name(ma_context* context, ma_device_id& id) {
      ma_device_info* playback = nullptr;
      ma_uint32 playback_count = 0;
      if (ma_context_get_devices(context, &playback, &playback_count, nullptr, nullptr) != MA_SUCCESS) {
        return false;
      }
      for (ma_uint32 i = 0; i < playback_count; ++i) {
        if (playback[i].isDefault) {
          std::string name = std::string(playback[i].id.pulse) + ".monitor";
          if (name.size() >= sizeof(id.pulse)) {
            return false;
          }
          std::memset(&id, 0, sizeof(id));
          std::memcpy(id.pulse, name.c_str(), name.size());
          return true;
        }
      }
      return false;
    }
  #endif
  }

  loopback_capture_t::~loopback_capture_t() {
    stop();
  }

  bool loopback_capture_t::is_supported() {
  #if defined(_WIN32) || defined(__linux__)
    return true;
  #else
    return false;
  #endif
  }

  bool loopback_capture_t::start(callback_t cb) {
    stop();
    callback = std::move(cb);
    pending.clear();
    error.clear();
    auto* d = new device_t;
    device = d;
    auto proc = [](ma_device* dev, void*, const void* input, ma_uint32 frame_count) {
      ((loopback_capture_t*)dev->pUserData)->on_data((const f32_t*)input, frame_count);
    };
  #if defined(_WIN32)
    ma_backend backends[] = {ma_backend_wasapi};
    ma_device_config config = make_config(ma_device_type_loopback, proc, this);
  #elif defined(__linux__)
    ma_backend backends[] = {ma_backend_pulseaudio};
    ma_device_config config = make_config(ma_device_type_capture, proc, this);
  #else
    error = "loopback capture is not supported on this platform";
    destroy_device(device);
    return false;
  #endif
  #if defined(_WIN32) || defined(__linux__)
    if (ma_context_init(backends, 1, nullptr, &d->context) != MA_SUCCESS) {
      error = "audio backend is not available";
      destroy_device(device);
      return false;
    }
    d->has_context = true;
    ma_result result;
  #if defined(__linux__)
    ma_device_id id {};
    std::strcpy(id.pulse, "@DEFAULT_MONITOR@");
    config.capture.pDeviceID = &id;
    result = ma_device_init(&d->context, &config, &d->device);
    if (result != MA_SUCCESS && default_monitor_name(&d->context, id)) {
      result = ma_device_init(&d->context, &config, &d->device);
    }
  #else
    result = ma_device_init(&d->context, &config, &d->device);
  #endif
    if (result != MA_SUCCESS) {
      error = std::string("failed to open loopback device: ") + ma_result_description(result);
      destroy_device(device);
      return false;
    }
    d->has_device = true;
    if (ma_device_start(&d->device) != MA_SUCCESS) {
      error = "failed to start loopback device";
      destroy_device(device);
      return false;
    }
    return true;
  #endif
  }

  void loopback_capture_t::stop() {
    destroy_device(device);
    pending.clear();
  }

  bool loopback_capture_t::is_running() const {
    return device != nullptr;
  }

  const std::string& loopback_capture_t::get_error() const {
    return error;
  }

  void loopback_capture_t::on_data(const f32_t* pcm, std::uint32_t frame_count) {
    if (!pcm || !callback) {
      return;
    }
    constexpr std::size_t packet_samples = stream_frame_size * stream_channels;
    pending.insert(pending.end(), pcm, pcm + frame_count * stream_channels);
    std::size_t offset = 0;
    for (; pending.size() - offset >= packet_samples; offset += packet_samples) {
      callback(pending.data() + offset, stream_frame_size);
    }
    pending.erase(pending.begin(), pending.begin() + offset);
  }

  live_player_t::~live_player_t() {
    stop();
  }

  bool live_player_t::start() {
    if (device) {
      return true;
    }
    clear();
    auto* d = new device_t;
    device = d;
    auto proc = [](ma_device* dev, void* output, const void*, ma_uint32 frame_count) {
      ((live_player_t*)dev->pUserData)->pull((f32_t*)output, frame_count);
    };
    ma_device_config config = make_config(ma_device_type_playback, proc, this);
    if (ma_device_init(nullptr, &config, &d->device) != MA_SUCCESS) {
      destroy_device(device);
      return false;
    }
    d->has_device = true;
    if (ma_device_start(&d->device) != MA_SUCCESS) {
      destroy_device(device);
      return false;
    }
    return true;
  }

  void live_player_t::stop() {
    destroy_device(device);
    clear();
  }

  bool live_player_t::is_running() const {
    return device != nullptr;
  }

  void live_player_t::push(const f32_t* pcm, std::uint32_t frame_count) {
    std::lock_guard<std::mutex> lock(mutex);
    std::size_t capacity = ring.size() / stream_channels;
    if (frame_count > capacity) {
      pcm += (frame_count - capacity) * stream_channels;
      frame_count = (std::uint32_t)capacity;
    }
    if (buffered_frames + frame_count > max_frames) {
      std::size_t skip = std::min<std::size_t>(buffered_frames + frame_count - target_frames, buffered_frames);
      read_frame = (read_frame + skip) % capacity;
      buffered_frames -= skip;
    }
    std::size_t write_frame = (read_frame + buffered_frames) % capacity;
    std::size_t first = std::min<std::size_t>(frame_count, capacity - write_frame);
    std::memcpy(&ring[write_frame * stream_channels], pcm, first * stream_channels * sizeof(f32_t));
    std::memcpy(ring.data(), pcm + first * stream_channels, (frame_count - first) * stream_channels * sizeof(f32_t));
    buffered_frames += frame_count;
  }

  void live_player_t::pull(f32_t* out, std::uint32_t frame_count) {
    std::lock_guard<std::mutex> lock(mutex);
    std::size_t capacity = ring.size() / stream_channels;
    primed = primed || buffered_frames >= target_frames;
    std::size_t count = primed ? std::min<std::size_t>(frame_count, buffered_frames) : 0;
    f32_t gain = volume.load(std::memory_order_relaxed);
    for (std::size_t i = 0; i < count; ++i) {
      std::size_t at = ((read_frame + i) % capacity) * stream_channels;
      for (std::uint32_t c = 0; c < stream_channels; ++c) {
        out[i * stream_channels + c] = ring[at + c] * gain;
      }
    }
    std::memset(out + count * stream_channels, 0, (frame_count - count) * stream_channels * sizeof(f32_t));
    read_frame = (read_frame + count) % capacity;
    buffered_frames -= count;
    primed = primed && count == frame_count;
  }

  void live_player_t::clear() {
    std::lock_guard<std::mutex> lock(mutex);
    read_frame = 0;
    buffered_frames = 0;
    primed = false;
  }

  void live_player_t::set_volume(f32_t value) {
    volume.store(std::clamp(value, 0.f, 4.f), std::memory_order_relaxed);
  }

  f32_t live_player_t::get_volume() const {
    return volume.load(std::memory_order_relaxed);
  }

  std::uint32_t live_player_t::get_buffered_frames() {
    std::lock_guard<std::mutex> lock(mutex);
    return (std::uint32_t)buffered_frames;
  }

  opus_encoder_t::~opus_encoder_t() {
    close();
  }

  bool opus_encoder_t::open(std::int32_t bitrate) {
    close();
    int err = 0;
    auto* encoder = opus_encoder_create(stream_sample_rate, stream_channels, OPUS_APPLICATION_AUDIO, &err);
    if (err != OPUS_OK || !encoder) {
      return false;
    }
    opus_encoder_ctl(encoder, OPUS_SET_BITRATE(bitrate));
    state = encoder;
    return true;
  }

  void opus_encoder_t::close() {
    if (state) {
      opus_encoder_destroy((OpusEncoder*)state);
      state = nullptr;
    }
  }

  std::int32_t opus_encoder_t::encode(const f32_t* pcm, std::uint8_t* out, std::int32_t capacity) {
    if (!state) {
      return -1;
    }
    return opus_encode_float((OpusEncoder*)state, pcm, stream_frame_size, out, capacity);
  }

  opus_decoder_t::~opus_decoder_t() {
    close();
  }

  bool opus_decoder_t::open() {
    close();
    int err = 0;
    auto* decoder = opus_decoder_create(stream_sample_rate, stream_channels, &err);
    if (err != OPUS_OK || !decoder) {
      return false;
    }
    state = decoder;
    return true;
  }

  void opus_decoder_t::close() {
    if (state) {
      opus_decoder_destroy((OpusDecoder*)state);
      state = nullptr;
    }
  }

  std::int32_t opus_decoder_t::decode(const std::uint8_t* data, std::int32_t size, f32_t* pcm) {
    if (!state) {
      return -1;
    }
    return opus_decode_float((OpusDecoder*)state, data, size, pcm, stream_frame_size, 0);
  }

  std::int32_t opus_decoder_t::conceal(f32_t* pcm) {
    return decode(nullptr, 0, pcm);
  }
}

#endif