module;

export module fan.audio.stream;

#if defined(FAN_AUDIO_STREAM)

import std;

export import fan.types;

export namespace fan::audio {
  inline constexpr std::uint32_t stream_sample_rate = 48000;
  inline constexpr std::uint32_t stream_channels = 2;
  inline constexpr std::uint32_t stream_frame_size = 960;
  inline constexpr std::uint32_t stream_max_packet = 1275;

  struct loopback_capture_t {
    using callback_t = std::function<void(const f32_t* pcm, std::uint32_t frame_count)>;

    loopback_capture_t() = default;
    loopback_capture_t(const loopback_capture_t&) = delete;
    loopback_capture_t& operator=(const loopback_capture_t&) = delete;
    ~loopback_capture_t();

    static bool is_supported();
    bool start(callback_t cb);
    void stop();
    bool is_running() const;
    const std::string& get_error() const;
    void on_data(const f32_t* pcm, std::uint32_t frame_count);

    void* device = nullptr;
    callback_t callback;
    std::vector<f32_t> pending;
    std::string error;
  };

  struct live_player_t {
    live_player_t() = default;
    live_player_t(const live_player_t&) = delete;
    live_player_t& operator=(const live_player_t&) = delete;
    ~live_player_t();

    bool start();
    void stop();
    bool is_running() const;
    void push(const f32_t* pcm, std::uint32_t frame_count);
    void clear();
    void set_volume(f32_t value);
    f32_t get_volume() const;
    std::uint32_t get_buffered_frames();
    void pull(f32_t* out, std::uint32_t frame_count);

    void* device = nullptr;
    std::mutex mutex;
    std::vector<f32_t> ring = std::vector<f32_t>(stream_sample_rate * stream_channels);
    std::size_t read_frame = 0;
    std::size_t buffered_frames = 0;
    bool primed = false;
    std::uint32_t target_frames = stream_frame_size * 3;
    std::uint32_t max_frames = stream_frame_size * 15;
    std::atomic<f32_t> volume = 1.f;
  };

  struct opus_encoder_t {
    opus_encoder_t() = default;
    opus_encoder_t(const opus_encoder_t&) = delete;
    opus_encoder_t& operator=(const opus_encoder_t&) = delete;
    ~opus_encoder_t();

    bool open(std::int32_t bitrate = 128000);
    void close();
    std::int32_t encode(const f32_t* pcm, std::uint8_t* out, std::int32_t capacity);

    void* state = nullptr;
  };

  struct opus_decoder_t {
    opus_decoder_t() = default;
    opus_decoder_t(const opus_decoder_t&) = delete;
    opus_decoder_t& operator=(const opus_decoder_t&) = delete;
    ~opus_decoder_t();

    bool open();
    void close();
    std::int32_t decode(const std::uint8_t* data, std::int32_t size, f32_t* pcm);
    std::int32_t conceal(f32_t* pcm);

    void* state = nullptr;
  };
}

#endif