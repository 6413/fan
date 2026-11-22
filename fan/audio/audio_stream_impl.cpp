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
  #if defined(_WIN32)
    #include <windows.h>
    #include <mmdeviceapi.h>
    #include <audioclient.h>
    #include <audioclientactivationparams.h>
    #include <tlhelp32.h>
    #pragma comment(lib, "mmdevapi.lib")
  #endif
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

    ma_result init_loopback_context(ma_context* context) {
    #if defined(_WIN32)
      ma_backend backends[] = {ma_backend_wasapi};
    #else
      ma_backend backends[] = {ma_backend_pulseaudio};
    #endif
      return ma_context_init(backends, 1, nullptr, context);
    }

  #if defined(_WIN32)
    std::string wide_to_utf8(const wchar_t* s) {
      int size = WideCharToMultiByte(CP_UTF8, 0, s, -1, nullptr, 0, nullptr, nullptr);
      if (size <= 1) {
        return {};
      }
      std::string out(size - 1, '\0');
      WideCharToMultiByte(CP_UTF8, 0, s, -1, out.data(), size, nullptr, nullptr);
      return out;
    }

    // WASAPI process loopback: records the mix of every process except one process tree
    struct activate_handler_t : IActivateAudioInterfaceCompletionHandler, IAgileObject {
      activate_handler_t() : done(CreateEventW(nullptr, FALSE, FALSE, nullptr)) {}
      ~activate_handler_t() {
        if (client) {
          client->Release();
        }
        CloseHandle(done);
      }
      STDMETHOD(QueryInterface)(REFIID riid, void** out) override {
        if (riid == __uuidof(IUnknown) || riid == __uuidof(IActivateAudioInterfaceCompletionHandler)) {
          *out = static_cast<IActivateAudioInterfaceCompletionHandler*>(this);
        }
        else if (riid == __uuidof(IAgileObject)) {
          *out = static_cast<IAgileObject*>(this);
        }
        else {
          *out = nullptr;
          return E_NOINTERFACE;
        }
        AddRef();
        return S_OK;
      }
      STDMETHOD_(ULONG, AddRef)() override {
        return InterlockedIncrement(&refs);
      }
      STDMETHOD_(ULONG, Release)() override {
        ULONG r = InterlockedDecrement(&refs);
        if (r == 0) {
          delete this;
        }
        return r;
      }
      STDMETHOD(ActivateCompleted)(IActivateAudioInterfaceAsyncOperation* operation) override {
        HRESULT hr = E_FAIL;
        IUnknown* unknown = nullptr;
        if (SUCCEEDED(operation->GetActivateResult(&hr, &unknown)) && SUCCEEDED(hr) && unknown) {
          hr = unknown->QueryInterface(__uuidof(IAudioClient), (void**)&client);
        }
        if (unknown) {
          unknown->Release();
        }
        result = hr;
        SetEvent(done);
        return S_OK;
      }

      LONG refs = 1;
      HANDLE done;
      HRESULT result = E_FAIL;
      IAudioClient* client = nullptr;
    };

    struct process_capture_t {
      std::thread thread;
      std::atomic<bool> stop = false;
    };

    std::string hresult_text(const char* what, HRESULT hr) {
      char buffer[96];
      std::snprintf(buffer, sizeof(buffer), "%s failed (0x%08lx)", what, (unsigned long)hr);
      return buffer;
    }

    // runs on its own MTA thread; reports init through `ready`, then pumps packets until stop
    // include: only pid's process tree, else everything except it
    void process_capture_thread(loopback_capture_t* owner, process_capture_t* state, std::uint32_t pid, bool include, std::promise<std::string>* ready) {
      CoInitializeEx(nullptr, COINIT_MULTITHREADED);
      IAudioClient* client = nullptr;
      IAudioCaptureClient* capture = nullptr;
      HANDLE event = nullptr;
      std::string error;
      do {
        AUDIOCLIENT_ACTIVATION_PARAMS params = {};
        params.ActivationType = AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK;
        params.ProcessLoopbackParams.TargetProcessId = pid;
        params.ProcessLoopbackParams.ProcessLoopbackMode = include ? PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE : PROCESS_LOOPBACK_MODE_EXCLUDE_TARGET_PROCESS_TREE;
        PROPVARIANT activate_params = {};
        activate_params.vt = VT_BLOB;
        activate_params.blob.cbSize = sizeof(params);
        activate_params.blob.pBlobData = (BYTE*)&params;

        auto* handler = new activate_handler_t;
        IActivateAudioInterfaceAsyncOperation* operation = nullptr;
        HRESULT hr = ActivateAudioInterfaceAsync(VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK, __uuidof(IAudioClient), &activate_params, handler, &operation);
        if (FAILED(hr)) {
          handler->Release();
          error = hresult_text("ActivateAudioInterfaceAsync", hr);
          break;
        }
        bool completed = WaitForSingleObject(handler->done, 5000) == WAIT_OBJECT_0;
        operation->Release();
        if (!completed || FAILED(handler->result) || !handler->client) {
          error = completed ? hresult_text("process loopback activation", handler->result) : "process loopback activation timed out";
          handler->Release();
          break;
        }
        client = handler->client;
        handler->client = nullptr;
        handler->Release();

        // process loopback has no mix format; it converts to whatever is asked
        WAVEFORMATEX format = {};
        format.wFormatTag = WAVE_FORMAT_PCM;
        format.nChannels = stream_channels;
        format.nSamplesPerSec = stream_sample_rate;
        format.wBitsPerSample = 16;
        format.nBlockAlign = format.nChannels * format.wBitsPerSample / 8;
        format.nAvgBytesPerSec = format.nSamplesPerSec * format.nBlockAlign;
        hr = client->Initialize(AUDCLNT_SHAREMODE_SHARED,
          AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_EVENTCALLBACK | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM,
          200 * 10000, 0, &format, nullptr);
        if (FAILED(hr)) {
          error = hresult_text("IAudioClient::Initialize", hr);
          break;
        }
        event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (FAILED(hr = client->SetEventHandle(event)) ||
          FAILED(hr = client->GetService(__uuidof(IAudioCaptureClient), (void**)&capture)) ||
          FAILED(hr = client->Start())) {
          error = hresult_text("process loopback start", hr);
          break;
        }
      } while (0);

      bool ok = error.empty();
      ready->set_value(std::move(error));
      std::vector<f32_t> pcm;
      while (ok && !state->stop.load(std::memory_order_relaxed)) {
        WaitForSingleObject(event, 100);
        UINT32 packet = 0;
        while (SUCCEEDED(capture->GetNextPacketSize(&packet)) && packet) {
          BYTE* data = nullptr;
          UINT32 frames = 0;
          DWORD flags = 0;
          if (FAILED(capture->GetBuffer(&data, &frames, &flags, nullptr, nullptr))) {
            break;
          }
          pcm.resize((std::size_t)frames * stream_channels);
          auto* samples = (const std::int16_t*)data;
          for (std::size_t i = 0; i < pcm.size(); ++i) {
            pcm[i] = (flags & AUDCLNT_BUFFERFLAGS_SILENT) ? 0.f : samples[i] / 32768.f;
          }
          capture->ReleaseBuffer(frames);
          owner->on_data(pcm.data(), frames);
        }
      }

      if (client) {
        client->Stop();
      }
      if (capture) {
        capture->Release();
      }
      if (client) {
        client->Release();
      }
      if (event) {
        CloseHandle(event);
      }
      CoUninitialize();
    }
  #endif
  }

  std::vector<loopback_device_t> get_loopback_devices() {
    std::vector<loopback_device_t> devices = {{"", "Default output"}};
  #if defined(_WIN32) || defined(__linux__)
    ma_context context;
    if (init_loopback_context(&context) != MA_SUCCESS) {
      return devices;
    }
    ma_device_info* playback = nullptr;
    ma_uint32 playback_count = 0;
    if (ma_context_get_devices(&context, &playback, &playback_count, nullptr, nullptr) == MA_SUCCESS) {
      for (ma_uint32 i = 0; i < playback_count; ++i) {
      #if defined(_WIN32)
        std::string id = wide_to_utf8(playback[i].id.wasapi);
      #else
        std::string id = playback[i].id.pulse;
      #endif
        if (playback[i].isDefault) {
          devices[0].name = std::string("Default output (") + playback[i].name + ")";
        }
        devices.push_back({id, playback[i].name});
      }
    }
    ma_context_uninit(&context);
  #endif
    return devices;
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

  bool loopback_capture_t::can_exclude_processes() {
  #if defined(_WIN32)
    return true;
  #else
    return false;
  #endif
  }

  std::uint32_t loopback_capture_t::find_discord_pid() {
  #if defined(_WIN32)
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snapshot == INVALID_HANDLE_VALUE) {
      return 0;
    }
    struct entry_t { DWORD pid, parent; };
    std::vector<entry_t> discord;
    PROCESSENTRY32W entry = {};
    entry.dwSize = sizeof(entry);
    for (BOOL ok = Process32FirstW(snapshot, &entry); ok; ok = Process32NextW(snapshot, &entry)) {
      for (const wchar_t* name : {L"Discord.exe", L"DiscordPTB.exe", L"DiscordCanary.exe"}) {
        if (_wcsicmp(entry.szExeFile, name) == 0) {
          discord.push_back({entry.th32ProcessID, entry.th32ParentProcessID});
        }
      }
    }
    CloseHandle(snapshot);
    // the voice audio comes from a child process, so exclude the tree from its root
    for (const auto& e : discord) {
      bool parent_is_discord = std::any_of(discord.begin(), discord.end(), [&](const entry_t& o) { return o.pid == e.parent; });
      if (!parent_is_discord) {
        return e.pid;
      }
    }
    return discord.empty() ? 0 : discord.front().pid;
  #else
    return 0;
  #endif
  }

  bool loopback_capture_t::start(callback_t cb, const options_t& options) {
    stop();
    callback = std::move(cb);
    pending.clear();
    error.clear();
    set_gain(options.gain);
    if (options.only_pid && can_exclude_processes()) {
      if (start_process(options.only_pid, true)) {
        return true;
      }
      std::string reason = error;
      bool ok = start_device(options.device_id);
      error = "could not record only the shared app: " + reason;
      return ok;
    }
    if (options.exclude_discord && can_exclude_processes()) {
      if (std::uint32_t pid = find_discord_pid()) {
        if (start_process(pid, false)) {
          return true;
        }
        // older windows has no process loopback; record the device instead
        std::string reason = error;
        bool ok = start_device(options.device_id);
        error = "could not exclude discord: " + reason;
        return ok;
      }
    }
    return start_device(options.device_id);
  }

  bool loopback_capture_t::start_process(std::uint32_t pid, bool include) {
  #if defined(_WIN32)
    auto* state = new process_capture_t;
    std::promise<std::string> ready;
    auto result = ready.get_future();
    state->thread = std::thread(process_capture_thread, this, state, pid, include, &ready);
    error = result.get();
    if (!error.empty()) {
      state->thread.join();
      delete state;
      return false;
    }
    process_capture = state;
    (include ? included_pid : excluded_pid) = pid;
    return true;
  #else
    error = "excluding processes is not supported on this platform";
    return false;
  #endif
  }

  bool loopback_capture_t::start_device(const std::string& device_id) {
    auto* d = new device_t;
    device = d;
    auto proc = [](ma_device* dev, void*, const void* input, ma_uint32 frame_count) {
      ((loopback_capture_t*)dev->pUserData)->on_data((const f32_t*)input, frame_count);
    };
  #if defined(_WIN32)
    ma_device_config config = make_config(ma_device_type_loopback, proc, this);
  #elif defined(__linux__)
    ma_device_config config = make_config(ma_device_type_capture, proc, this);
  #else
    error = "loopback capture is not supported on this platform";
    destroy_device(device);
    return false;
  #endif
  #if defined(_WIN32) || defined(__linux__)
    if (init_loopback_context(&d->context) != MA_SUCCESS) {
      error = "audio backend is not available";
      destroy_device(device);
      return false;
    }
    d->has_context = true;
    ma_result result;
    ma_device_id id {};
  #if defined(__linux__)
    // pulse records an output through its monitor source
    std::string source = device_id.empty() ? "@DEFAULT_MONITOR@" : device_id + ".monitor";
    if (source.size() >= sizeof(id.pulse)) {
      error = "device name is too long";
      destroy_device(device);
      return false;
    }
    std::memcpy(id.pulse, source.c_str(), source.size());
    config.capture.pDeviceID = &id;
    result = ma_device_init(&d->context, &config, &d->device);
    if (result != MA_SUCCESS && device_id.empty() && default_monitor_name(&d->context, id)) {
      result = ma_device_init(&d->context, &config, &d->device);
    }
  #else
    if (!device_id.empty()) {
      // a playback device id: loopback records that output
      if (MultiByteToWideChar(CP_UTF8, 0, device_id.c_str(), -1, id.wasapi, (int)(sizeof(id.wasapi) / sizeof(id.wasapi[0]))) == 0) {
        error = "invalid device id";
        destroy_device(device);
        return false;
      }
      config.capture.pDeviceID = &id;
    }
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
  #if defined(_WIN32)
    if (auto* state = (process_capture_t*)process_capture) {
      state->stop = true;
      state->thread.join();
      delete state;
      process_capture = nullptr;
    }
  #endif
    excluded_pid = 0;
    included_pid = 0;
    pending.clear();
  }

  bool loopback_capture_t::is_running() const {
    return device != nullptr || process_capture != nullptr;
  }

  void loopback_capture_t::set_gain(f32_t value) {
    gain.store(std::clamp(value, 0.f, 4.f), std::memory_order_relaxed);
  }

  const std::string& loopback_capture_t::get_error() const {
    return error;
  }

  std::uint32_t loopback_capture_t::get_excluded_pid() const {
    return excluded_pid;
  }

  std::uint32_t loopback_capture_t::get_included_pid() const {
    return included_pid;
  }

  void loopback_capture_t::on_data(const f32_t* pcm, std::uint32_t frame_count) {
    if (!pcm || !callback) {
      return;
    }
    constexpr std::size_t packet_samples = stream_frame_size * stream_channels;
    std::size_t start = pending.size();
    pending.insert(pending.end(), pcm, pcm + frame_count * stream_channels);
    f32_t g = gain.load(std::memory_order_relaxed);
    if (g != 1.f) {
      for (std::size_t i = start; i < pending.size(); ++i) {
        pending[i] *= g;
      }
    }
    std::size_t offset = 0;
    for (; pending.size() - offset >= packet_samples; offset += packet_samples) {
      callback(pending.data() + offset, stream_frame_size);
    }
    pending.erase(pending.begin(), pending.begin() + offset);
  }

  std::vector<microphone_device_t> get_microphone_devices() {
    std::vector<microphone_device_t> devices = {{"", "Default microphone"}};
  #if defined(_WIN32) || defined(__linux__)
    ma_context context;
    if (init_loopback_context(&context) != MA_SUCCESS) {
      return devices;
    }
    ma_device_info* capture = nullptr;
    ma_uint32 capture_count = 0;
    if (ma_context_get_devices(&context, nullptr, nullptr, &capture, &capture_count) == MA_SUCCESS) {
      for (ma_uint32 i = 0; i < capture_count; ++i) {
      #if defined(_WIN32)
        std::string id = wide_to_utf8(capture[i].id.wasapi);
      #else
        std::string id = capture[i].id.pulse;
        if (id.ends_with(".monitor")) {
          continue;
        }
      #endif
        if (capture[i].isDefault) {
          devices[0].name = std::string("Default microphone (") + capture[i].name + ")";
        }
        devices.push_back({id, capture[i].name});
      }
    }
    ma_context_uninit(&context);
  #endif
    return devices;
  }

  microphone_capture_t::~microphone_capture_t() {
    stop();
  }

  bool microphone_capture_t::start(callback_t cb, const std::string& device_id) {
    stop();
    error.clear();
    callback = std::move(cb);
  #if defined(_WIN32) || defined(__linux__)
    auto* d = new device_t;
    device = d;
    auto proc = [](ma_device* dev, void*, const void* input, ma_uint32 frame_count) {
      ((microphone_capture_t*)dev->pUserData)->on_data((const f32_t*)input, frame_count);
    };
    ma_device_config config = make_config(ma_device_type_capture, proc, this);
    if (init_loopback_context(&d->context) != MA_SUCCESS &&
      (!device_id.empty() || ma_context_init(nullptr, 0, nullptr, &d->context) != MA_SUCCESS)) {
      error = "audio backend is not available";
      destroy_device(device);
      return false;
    }
    d->has_context = true;
    ma_device_id id {};
    if (!device_id.empty()) {
    #if defined(_WIN32)
      bool valid = MultiByteToWideChar(CP_UTF8, 0, device_id.c_str(), -1, id.wasapi, (int)(sizeof(id.wasapi) / sizeof(id.wasapi[0]))) != 0;
    #else
      bool valid = device_id.size() < sizeof(id.pulse);
      if (valid) {
        std::memcpy(id.pulse, device_id.c_str(), device_id.size());
      }
    #endif
      if (!valid) {
        error = "invalid device id";
        destroy_device(device);
        return false;
      }
      config.capture.pDeviceID = &id;
    }
    ma_result result = ma_device_init(&d->context, &config, &d->device);
    if (result != MA_SUCCESS) {
      error = std::string("failed to open the microphone: ") + ma_result_description(result);
      destroy_device(device);
      return false;
    }
    d->has_device = true;
    if (ma_device_start(&d->device) != MA_SUCCESS) {
      error = "failed to start the microphone";
      destroy_device(device);
      return false;
    }
    return true;
  #else
    error = "microphone capture is not supported on this platform";
    return false;
  #endif
  }

  void microphone_capture_t::stop() {
    destroy_device(device);
    pending.clear();
    level.store(0.f, std::memory_order_relaxed);
  }

  bool microphone_capture_t::is_running() const {
    return device != nullptr;
  }

  void microphone_capture_t::set_gain(f32_t value) {
    gain.store(std::clamp(value, 0.f, 4.f), std::memory_order_relaxed);
  }

  f32_t microphone_capture_t::get_level() const {
    return level.load(std::memory_order_relaxed);
  }

  const std::string& microphone_capture_t::get_error() const {
    return error;
  }

  void microphone_capture_t::on_data(const f32_t* pcm, std::uint32_t frame_count) {
    if (!pcm || !callback) {
      return;
    }
    constexpr std::size_t packet_samples = stream_frame_size * stream_channels;
    std::size_t start = pending.size();
    pending.insert(pending.end(), pcm, pcm + frame_count * stream_channels);
    f32_t g = gain.load(std::memory_order_relaxed);
    f32_t peak = 0.f;
    for (std::size_t i = start; i < pending.size(); ++i) {
      pending[i] *= g;
      peak = std::max(peak, std::abs(pending[i]));
    }
    level.store(peak, std::memory_order_relaxed);
    std::size_t offset = 0;
    for (; pending.size() - offset >= packet_samples; offset += packet_samples) {
      callback(pending.data() + offset, stream_frame_size);
    }
    pending.erase(pending.begin(), pending.begin() + offset);
  }

  live_mixer_t::~live_mixer_t() {
    stop();
  }

  bool live_mixer_t::start(const std::string& device_id) {
    if (device) {
      return true;
    }
    auto* d = new device_t;
    device = d;
    auto proc = [](ma_device* dev, void* output, const void*, ma_uint32 frame_count) {
      ((live_mixer_t*)dev->pUserData)->pull((f32_t*)output, frame_count);
    };
    ma_device_config config = make_config(ma_device_type_playback, proc, this);
    ma_device_id id {};
    if (!device_id.empty() && init_loopback_context(&d->context) == MA_SUCCESS) {
      d->has_context = true;
    #if defined(_WIN32)
      bool valid = MultiByteToWideChar(CP_UTF8, 0, device_id.c_str(), -1, id.wasapi, (int)(sizeof(id.wasapi) / sizeof(id.wasapi[0]))) != 0;
    #else
      bool valid = device_id.size() < sizeof(id.pulse);
      if (valid) {
        std::memcpy(id.pulse, device_id.c_str(), device_id.size());
      }
    #endif
      if (valid) {
        config.playback.pDeviceID = &id;
      }
    }
    if (!d->has_context) {
    #if defined(_WIN32)
      ma_backend backends[] = {ma_backend_wasapi, ma_backend_dsound, ma_backend_winmm};
    #elif defined(__linux__)
      ma_backend backends[] = {ma_backend_pulseaudio, ma_backend_alsa, ma_backend_jack};
    #else
      ma_backend backends[] = {ma_backend_coreaudio};
    #endif
      if (ma_context_init(backends, sizeof(backends) / sizeof(backends[0]), nullptr, &d->context) != MA_SUCCESS) {
        destroy_device(device);
        return false;
      }
      d->has_context = true;
    }
    ma_result result = ma_device_init(&d->context, &config, &d->device);
    if (result != MA_SUCCESS && config.playback.pDeviceID) {
      config.playback.pDeviceID = nullptr;
      result = ma_device_init(&d->context, &config, &d->device);
    }
    if (result != MA_SUCCESS) {
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

  void live_mixer_t::stop() {
    destroy_device(device);
    std::lock_guard<std::mutex> lock(mutex);
    sources.clear();
  }

  bool live_mixer_t::is_running() const {
    return device != nullptr;
  }

  void live_mixer_t::push(std::uint32_t source_id, const f32_t* pcm, std::uint32_t frame_count) {
    std::lock_guard<std::mutex> lock(mutex);
    source_t& source = sources[source_id];
    if (source.ring.empty()) {
      source.ring.resize((max_frames + stream_frame_size) * stream_channels);
    }
    std::size_t capacity = source.ring.size() / stream_channels;
    if (frame_count > capacity) {
      pcm += (frame_count - capacity) * stream_channels;
      frame_count = (std::uint32_t)capacity;
    }
    if (source.buffered_frames + frame_count > max_frames) {
      std::size_t skip = std::min<std::size_t>(source.buffered_frames + frame_count - target_frames, source.buffered_frames);
      source.read_frame = (source.read_frame + skip) % capacity;
      source.buffered_frames -= skip;
    }
    std::size_t write_frame = (source.read_frame + source.buffered_frames) % capacity;
    std::size_t first = std::min<std::size_t>(frame_count, capacity - write_frame);
    std::memcpy(&source.ring[write_frame * stream_channels], pcm, first * stream_channels * sizeof(f32_t));
    std::memcpy(source.ring.data(), pcm + first * stream_channels, (frame_count - first) * stream_channels * sizeof(f32_t));
    source.buffered_frames += frame_count;
  }

  void live_mixer_t::remove(std::uint32_t source_id) {
    std::lock_guard<std::mutex> lock(mutex);
    sources.erase(source_id);
  }

  void live_mixer_t::set_volume(f32_t value) {
    volume.store(std::clamp(value, 0.f, 4.f), std::memory_order_relaxed);
  }

  void live_mixer_t::pull(f32_t* out, std::uint32_t frame_count) {
    std::memset(out, 0, frame_count * stream_channels * sizeof(f32_t));
    {
      std::lock_guard<std::mutex> lock(mutex);
      f32_t gain = volume.load(std::memory_order_relaxed);
      for (auto& [id, source] : sources) {
        std::size_t capacity = source.ring.size() / stream_channels;
        source.primed = source.primed || source.buffered_frames >= target_frames;
        std::size_t count = source.primed ? std::min<std::size_t>(frame_count, source.buffered_frames) : 0;
        for (std::size_t i = 0; i < count; ++i) {
          std::size_t at = ((source.read_frame + i) % capacity) * stream_channels;
          for (std::uint32_t c = 0; c < stream_channels; ++c) {
            out[i * stream_channels + c] += source.ring[at + c] * gain;
          }
        }
        source.read_frame = (source.read_frame + count) % capacity;
        source.buffered_frames -= count;
        source.primed = source.primed && count == frame_count;
      }
    }
    for (std::size_t i = 0; i < (std::size_t)frame_count * stream_channels; ++i) {
      out[i] = std::clamp(out[i], -1.f, 1.f);
    }
    if (played) {
      played(out, frame_count);
    }
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
    // without an explicit list miniaudio falls back to its null backend, which plays silence
  #if defined(_WIN32)
    ma_backend backends[] = {ma_backend_wasapi, ma_backend_dsound, ma_backend_winmm};
  #elif defined(__linux__)
    ma_backend backends[] = {ma_backend_pulseaudio, ma_backend_alsa, ma_backend_jack};
  #else
    ma_backend backends[] = {ma_backend_coreaudio};
  #endif
    if (ma_context_init(backends, sizeof(backends) / sizeof(backends[0]), nullptr, &d->context) != MA_SUCCESS) {
      std::printf("audio playback: no audio backend available\n");
      destroy_device(device);
      return false;
    }
    d->has_context = true;
    if (ma_device_init(&d->context, &config, &d->device) != MA_SUCCESS) {
      destroy_device(device);
      return false;
    }
    d->has_device = true;
    if (ma_device_start(&d->device) != MA_SUCCESS) {
      destroy_device(device);
      return false;
    }
    std::printf("audio playback backend: %s\n", ma_get_backend_name(d->context.backend));
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

  bool opus_encoder_t::open(std::int32_t bitrate, bool voice) {
    close();
    int err = 0;
    auto* encoder = opus_encoder_create(stream_sample_rate, stream_channels, voice ? OPUS_APPLICATION_VOIP : OPUS_APPLICATION_AUDIO, &err);
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