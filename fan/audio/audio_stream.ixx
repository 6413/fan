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

  using microphone_device_t = loopback_device_t;
  std::vector<microphone_device_t> get_microphone_devices();

  struct microphone_capture_t {
    using callback_t = std::function<void(const f32_t* pcm, std::uint32_t frame_count)>;

    microphone_capture_t() = default;
    microphone_capture_t(const microphone_capture_t&) = delete;
    microphone_capture_t& operator=(const microphone_capture_t&) = delete;
    ~microphone_capture_t();

    bool start(callback_t cb, const std::string& device_id = {});
    void stop();
    bool is_running() const;
    void set_gain(f32_t value);
    f32_t get_level() const;
    const std::string& get_error() const;
    void on_data(const f32_t* pcm, std::uint32_t frame_count);

    void* device = nullptr;
    callback_t callback;
    std::vector<f32_t> pending;
    std::string error;
    std::atomic<f32_t> gain = 1.f;
    std::atomic<f32_t> level = 0.f;
  };

  struct echo_canceller_t {
    static constexpr std::size_t fft_size = 512;
    static constexpr std::size_t hop = fft_size / 2;
    static constexpr std::size_t bins = fft_size / 2 + 1;
    static constexpr std::size_t bands = 24;
    static constexpr std::size_t lags = 96;
    static constexpr std::size_t mic_delay = fft_size * 2;
    using spectrum_t = std::array<std::complex<f32_t>, fft_size>;
    using band_powers_t = std::array<f32_t, bands>;

    echo_canceller_t() {
      for (std::size_t i = 0; i < fft_size; ++i) {
        window[i] = std::sin(std::numbers::pi_v<f32_t> * (i + 0.5f) / fft_size);
      }
      for (std::size_t b = 0; b <= bands; ++b) {
        band_edges[b] = std::max<std::size_t>(b, (std::size_t)std::lround((bins - 1) * std::pow((f32_t)b / bands, 2.f)));
      }
      band_edges[bands] = bins;
      reset();
    }
    void reset() {
      {
        std::lock_guard<std::mutex> lock(reference_mutex);
        reference.clear();
      }
      input.clear();
      output.assign(mic_delay + hop, 0.f);
      mic_window.fill(0.f);
      reference_window.fill(0.f);
      overlap.fill(0.f);
      for (auto& powers : played) {
        powers.fill(0.f);
      }
      match.fill(0.f);
      echo_tail.fill(0.f);
      coupling_top.fill(0.f);
      coupling_bottom.fill(0.f);
      gains.fill(1.f);
      played_at = 0;
      delay = 0;
    }
    void push_reference(const f32_t* pcm, std::uint32_t frame_count) {
      std::lock_guard<std::mutex> lock(reference_mutex);
      for (std::uint32_t i = 0; i < frame_count; ++i) {
        reference.push_back((pcm[i * stream_channels] + pcm[i * stream_channels + 1]) * 0.5f);
      }
      if (reference.size() > mic_delay) {
        reference.erase(reference.begin(), reference.end() - mic_delay);
      }
    }
    void process(f32_t* pcm, std::uint32_t frame_count) {
      for (std::uint32_t i = 0; i < frame_count; ++i) {
        input.push_back((pcm[i * stream_channels] + pcm[i * stream_channels + 1]) * 0.5f);
      }
      std::size_t used = 0;
      for (; input.size() - used >= hop + mic_delay; used += hop) {
        process_hop(input.data() + used);
      }
      input.erase(input.begin(), input.begin() + used);
      for (std::uint32_t i = 0; i < frame_count; ++i) {
        pcm[i * stream_channels] = pcm[i * stream_channels + 1] = output[i];
      }
      output.erase(output.begin(), output.begin() + frame_count);
    }
    void process_hop(const f32_t* mic) {
      std::copy(mic_window.begin() + hop, mic_window.end(), mic_window.begin());
      std::copy(mic, mic + hop, mic_window.end() - hop);
      std::copy(reference_window.begin() + hop, reference_window.end(), reference_window.begin());
      {
        std::lock_guard<std::mutex> lock(reference_mutex);
        std::size_t count = std::min(reference.size(), hop);
        std::fill(reference_window.end() - hop, reference_window.end() - count, 0.f);
        std::copy(reference.begin(), reference.begin() + count, reference_window.end() - count);
        reference.erase(reference.begin(), reference.begin() + count);
      }
      spectrum_t heard, sent;
      band_powers_t heard_power;
      played_at = (played_at + 1) % lags;
      analyse(reference_window, sent, played[played_at]);
      analyse(mic_window, heard, heard_power);

      f32_t heard_size = size_of(heard_power);
      for (std::size_t lag = 0; lag < lags; ++lag) {
        const band_powers_t& then = played[(played_at + lags - lag) % lags];
        f32_t then_size = size_of(then);
        if (then_size > quiet && heard_size > quiet) {
          f32_t alike = 0.f;
          for (std::size_t b = 0; b < bands; ++b) {
            alike += then[b] * heard_power[b];
          }
          match[lag] += (alike / (then_size * heard_size) - match[lag]) * 0.01f;
        }
      }
      delay = std::max_element(match.begin(), match.end()) - match.begin();

      std::array<f32_t, bands> wanted;
      for (std::size_t b = 0; b < bands; ++b) {
        f32_t now = 0.f;
        for (std::size_t lag = delay == 0 ? 0 : delay - 1; lag <= std::min(delay + 1, lags - 1); ++lag) {
          now = std::max(now, played[(played_at + lags - lag) % lags][b]);
        }
        echo_tail[b] = std::max(now, echo_tail[b] * tail_decay);
        if (echo_tail[b] > quiet) {
          coupling_top[b] += (heard_power[b] * echo_tail[b] - coupling_top[b]) * learn_rate;
          coupling_bottom[b] += (echo_tail[b] * echo_tail[b] - coupling_bottom[b]) * learn_rate;
        }
        f32_t coupling = coupling_bottom[b] > quiet * quiet ? coupling_top[b] / coupling_bottom[b] : unlearned_coupling;
        f32_t echo = coupling * echo_tail[b] * over_estimate;
        wanted[b] = std::clamp(1.f - echo / (heard_power[b] + quiet), min_gain, 1.f);
        gains[b] = wanted[b] < gains[b] ? wanted[b] : gains[b] + (wanted[b] - gains[b]) * 0.3f;
      }
      for (std::size_t b = 0; b < bands; ++b) {
        for (std::size_t i = band_edges[b]; i < band_edges[b + 1]; ++i) {
          heard[i] *= gains[b];
          if (i != 0 && i != fft_size / 2) {
            heard[fft_size - i] = std::conj(heard[i]);
          }
        }
      }
      transform(heard, true);
      for (std::size_t i = 0; i < fft_size; ++i) {
        overlap[i] += heard[i].real() * window[i] / fft_size;
      }
      output.insert(output.end(), overlap.begin(), overlap.begin() + hop);
      std::copy(overlap.begin() + hop, overlap.end(), overlap.begin());
      std::fill(overlap.end() - hop, overlap.end(), 0.f);
    }
    void analyse(const std::array<f32_t, fft_size>& samples, spectrum_t& spectrum, band_powers_t& powers) const {
      for (std::size_t i = 0; i < fft_size; ++i) {
        spectrum[i] = samples[i] * window[i];
      }
      transform(spectrum, false);
      for (std::size_t b = 0; b < bands; ++b) {
        powers[b] = 0.f;
        for (std::size_t i = band_edges[b]; i < band_edges[b + 1]; ++i) {
          powers[b] += std::norm(spectrum[i]);
        }
        powers[b] /= fft_size * fft_size;
      }
    }
    static f32_t size_of(const band_powers_t& powers) {
      f32_t sum = 0.f;
      for (f32_t power : powers) {
        sum += power * power;
      }
      return std::sqrt(sum);
    }
    static void transform(spectrum_t& data, bool inverse) {
      for (std::size_t i = 1, j = 0; i < fft_size; ++i) {
        std::size_t bit = fft_size >> 1;
        for (; j & bit; bit >>= 1) {
          j ^= bit;
        }
        j ^= bit;
        if (i < j) {
          std::swap(data[i], data[j]);
        }
      }
      for (std::size_t length = 2; length <= fft_size; length <<= 1) {
        f32_t angle = 2.f * std::numbers::pi_v<f32_t> / length * (inverse ? 1.f : -1.f);
        std::complex<f32_t> step(std::cos(angle), std::sin(angle));
        for (std::size_t i = 0; i < fft_size; i += length) {
          std::complex<f32_t> turn(1.f, 0.f);
          for (std::size_t k = 0; k < length / 2; ++k) {
            std::complex<f32_t> a = data[i + k], b = data[i + k + length / 2] * turn;
            data[i + k] = a + b;
            data[i + k + length / 2] = a - b;
            turn *= step;
          }
        }
      }
    }

    static constexpr f32_t quiet = 1e-10f;
    static constexpr f32_t tail_decay = 0.85f;
    static constexpr f32_t learn_rate = 0.005f;
    static constexpr f32_t unlearned_coupling = 4.f;
    static constexpr f32_t over_estimate = 3.f;
    static constexpr f32_t min_gain = 0.03f;

    std::mutex reference_mutex;
    std::vector<f32_t> reference;
    std::vector<f32_t> input;
    std::vector<f32_t> output;
    std::array<f32_t, fft_size> window;
    std::array<std::size_t, bands + 1> band_edges;
    std::array<f32_t, fft_size> mic_window;
    std::array<f32_t, fft_size> reference_window;
    std::array<f32_t, fft_size> overlap;
    std::array<band_powers_t, lags> played;
    std::array<f32_t, lags> match;
    band_powers_t echo_tail;
    band_powers_t coupling_top;
    band_powers_t coupling_bottom;
    band_powers_t gains;
    std::size_t played_at = 0;
    std::size_t delay = 0;
  };

  // Spectral noise suppressor for a microphone signal (steady noise: fans, hiss, hum, room tone).
  // The noise floor of every frequency bin is tracked with minimum statistics, so it needs no
  // calibration and follows the room as it changes. The speech-to-noise ratio is estimated with the
  // decision-directed method and applied as a Wiener gain, smoothed over frequency and time so that
  // it leaves no "musical" artefacts. Adds 512 samples (about 10 ms) of delay.
  struct noise_suppressor_t {
    static constexpr std::size_t fft_size = echo_canceller_t::fft_size;
    static constexpr std::size_t hop = fft_size / 2;
    static constexpr std::size_t bins = fft_size / 2 + 1;
    using spectrum_t = echo_canceller_t::spectrum_t;

    noise_suppressor_t() {
      for (std::size_t i = 0; i < fft_size; ++i) {
        window[i] = std::sin(std::numbers::pi_v<f32_t> * (i + 0.5f) / fft_size);
      }
      reset();
    }
    void reset() {
      input.clear();
      output.assign(hop, 0.f);
      frame.fill(0.f);
      overlap.fill(0.f);
      smoothed.fill(0.f);
      noise.fill(0.f);
      previous_clean.fill(0.f);
      gains.fill(1.f);
      current_min.fill(std::numeric_limits<f32_t>::max());
      for (auto& sub : sub_minima) {
        sub.fill(std::numeric_limits<f32_t>::max());
      }
      frames_seen = 0;
      sub_index = 0;
      sub_frames = 0;
    }
    void process(f32_t* pcm, std::uint32_t frame_count) {
      for (std::uint32_t i = 0; i < frame_count; ++i) {
        input.push_back((pcm[i * stream_channels] + pcm[i * stream_channels + 1]) * 0.5f);
      }
      std::size_t used = 0;
      for (; input.size() - used >= hop; used += hop) {
        process_hop(input.data() + used);
      }
      input.erase(input.begin(), input.begin() + used);
      for (std::uint32_t i = 0; i < frame_count; ++i) {
        pcm[i * stream_channels] = pcm[i * stream_channels + 1] = output[i];
      }
      output.erase(output.begin(), output.begin() + frame_count);
    }
    void process_hop(const f32_t* samples) {
      std::copy(frame.begin() + hop, frame.end(), frame.begin());
      std::copy(samples, samples + hop, frame.end() - hop);

      spectrum_t spectrum;
      for (std::size_t i = 0; i < fft_size; ++i) {
        spectrum[i] = frame[i] * window[i];
      }
      echo_canceller_t::transform(spectrum, false);

      std::array<f32_t, bins> power;
      for (std::size_t k = 0; k < bins; ++k) {
        power[k] = std::norm(spectrum[k]) / (fft_size * fft_size);
      }
      track_noise(power);

      std::array<f32_t, bins> wanted;
      for (std::size_t k = 0; k < bins; ++k) {
        f32_t floor_power = std::max(noise[k], tiny);
        f32_t posteriori = power[k] / floor_power;
        f32_t priori = decision_directed * previous_clean[k] / floor_power +
          (1.f - decision_directed) * std::max(posteriori - 1.f, 0.f);
        priori = std::max(priori, min_priori);
        wanted[k] = priori / (1.f + priori);
      }
      // smooth over frequency: single bins that pop above the noise are the "musical" artefacts
      std::array<f32_t, bins> blurred;
      for (std::size_t k = 0; k < bins; ++k) {
        f32_t left = wanted[k == 0 ? 0 : k - 1];
        f32_t right = wanted[k + 1 < bins ? k + 1 : bins - 1];
        blurred[k] = std::max(0.25f * left + 0.5f * wanted[k] + 0.25f * right, min_gain);
      }
      for (std::size_t k = 0; k < bins; ++k) {
        // attack fast so word starts are not clipped, release slowly so tails do not chop
        gains[k] += (blurred[k] - gains[k]) * (blurred[k] > gains[k] ? 0.8f : 0.35f);
        previous_clean[k] = gains[k] * gains[k] * power[k];
        spectrum[k] *= gains[k];
        if (k != 0 && k != fft_size / 2) {
          spectrum[fft_size - k] = std::conj(spectrum[k]);
        }
      }
      echo_canceller_t::transform(spectrum, true);
      for (std::size_t i = 0; i < fft_size; ++i) {
        overlap[i] += spectrum[i].real() * window[i] / fft_size;
      }
      output.insert(output.end(), overlap.begin(), overlap.begin() + hop);
      std::copy(overlap.begin() + hop, overlap.end(), overlap.begin());
      std::fill(overlap.end() - hop, overlap.end(), 0.f);
    }
    void track_noise(const std::array<f32_t, bins>& power) {
      ++frames_seen;
      for (std::size_t k = 0; k < bins; ++k) {
        // a plain running mean while warming up: a single frame is a poor estimate (its power is
        // exponentially distributed, many bins are far too low) and must never enter the minima
        smoothed[k] += (power[k] - smoothed[k]) * (frames_seen < warmup ? 1.f / frames_seen : 0.1f);
        if (frames_seen >= warmup) {
          current_min[k] = std::min(current_min[k], smoothed[k]);
        }
      }
      // the floor is the lowest smoothed power of the last sub_count * sub_length frames, a speech pause
      // always falls inside that window, then scaled up because a minimum underestimates the mean
      if (++sub_frames >= sub_length) {
        sub_minima[sub_index] = current_min;
        sub_index = (sub_index + 1) % sub_count;
        sub_frames = 0;
        current_min.fill(std::numeric_limits<f32_t>::max());
      }
      for (std::size_t k = 0; k < bins; ++k) {
        f32_t lowest = current_min[k];
        for (const auto& sub : sub_minima) {
          lowest = std::min(lowest, sub[k]);
        }
        f32_t estimate = lowest * bias;
        // the first frames have no history yet: assume they are noise
        noise[k] = frames_seen < warmup ? std::max(smoothed[k], tiny) : std::max(estimate, tiny);
      }
    }

    static constexpr f32_t tiny = 1e-12f;
    static constexpr f32_t bias = 1.6f;
    static constexpr f32_t decision_directed = 0.92f;
    static constexpr f32_t min_priori = 0.003f;
    static constexpr f32_t min_gain = 0.08f;
    static constexpr std::size_t sub_count = 8;
    static constexpr std::size_t sub_length = 40;
    static constexpr std::size_t warmup = 20;

    std::vector<f32_t> input;
    std::vector<f32_t> output;
    std::array<f32_t, fft_size> window;
    std::array<f32_t, fft_size> frame;
    std::array<f32_t, fft_size> overlap;
    std::array<f32_t, bins> smoothed;
    std::array<f32_t, bins> noise;
    std::array<f32_t, bins> previous_clean;
    std::array<f32_t, bins> gains;
    std::array<f32_t, bins> current_min;
    std::array<std::array<f32_t, bins>, sub_count> sub_minima;
    std::size_t frames_seen = 0;
    std::size_t sub_index = 0;
    std::size_t sub_frames = 0;
  };

  struct live_mixer_t {
    struct source_t {
      std::vector<f32_t> ring;
      std::size_t read_frame = 0;
      std::size_t buffered_frames = 0;
      bool primed = false;
    };
    using played_callback_t = std::function<void(const f32_t* pcm, std::uint32_t frame_count)>;

    live_mixer_t() = default;
    live_mixer_t(const live_mixer_t&) = delete;
    live_mixer_t& operator=(const live_mixer_t&) = delete;
    ~live_mixer_t();

    bool start(const std::string& device_id = {});
    void stop();
    bool is_running() const;
    void push(std::uint32_t source, const f32_t* pcm, std::uint32_t frame_count);
    void remove(std::uint32_t source);
    void set_volume(f32_t value);
    void pull(f32_t* out, std::uint32_t frame_count);

    void* device = nullptr;
    std::mutex mutex;
    std::unordered_map<std::uint32_t, source_t> sources;
    std::atomic<f32_t> volume = 1.f;
    played_callback_t played;
    std::uint32_t target_frames = stream_frame_size * 3;
    std::uint32_t max_frames = stream_frame_size * 15;
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

    bool open(std::int32_t bitrate = 128000, bool voice = false);
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