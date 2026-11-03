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

  // an output whose mix can be recorded; an empty id follows the system default output
  struct loopback_device_t {
    std::string id;
    std::string name;
  };
  // the first entry is always the default output
  std::vector<loopback_device_t> get_loopback_devices();

  // outside loopback_capture_t so it can be a defaulted argument of its members
  struct loopback_capture_options_t {
    std::string device_id;
    // records every app except discord's process tree (windows only), device_id is then unused
    bool exclude_discord = false;
    f32_t gain = 1.f;
  };

  struct loopback_capture_t {
    using callback_t = std::function<void(const f32_t* pcm, std::uint32_t frame_count)>;
    using options_t = loopback_capture_options_t;

    loopback_capture_t() = default;
    loopback_capture_t(const loopback_capture_t&) = delete;
    loopback_capture_t& operator=(const loopback_capture_t&) = delete;
    ~loopback_capture_t();

    static bool is_supported();
    static bool can_exclude_processes();
    // root of the running discord process tree, 0 when discord is not running
    static std::uint32_t find_discord_pid();

    bool start(callback_t cb, const options_t& options = {});
    void stop();
    bool is_running() const;
    void set_gain(f32_t value);
    const std::string& get_error() const;
    // pid excluded by the running capture, 0 when recording a device mix
    std::uint32_t get_excluded_pid() const;
    void on_data(const f32_t* pcm, std::uint32_t frame_count);

    bool start_device(const std::string& device_id);
    bool start_process_exclude(std::uint32_t pid);

    void* device = nullptr;
    void* process_capture = nullptr;
    std::uint32_t excluded_pid = 0;
    callback_t callback;
    std::vector<f32_t> pending;
    std::string error;
    std::atomic<f32_t> gain = 1.f;
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