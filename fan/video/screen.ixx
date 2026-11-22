module;

#if defined (FAN_WINDOW) && defined(FAN_VIDEO)

#include <cstring>

#if defined(FAN_2D)
#include <fan/utils/utility.h>
#include <WITCH/WITCH.h>
#include <WITCH/PR/PR.h>
#include <WITCH/MD/SCR/SCR.h>
#endif

#endif

export module fan.graphics.video.screen;

#if defined (FAN_WINDOW) && defined(FAN_VIDEO)

import std;
import fan.graphics.video.codec;

#if defined(FAN_2D)
import fan.types;
import fan.types.vector;
import fan.graphics.common_context;
import fan.graphics.shapes;

import fan.time;
import fan.print;
import fan.event;

export namespace fan {
  namespace graphics {

    struct screen_encode_t;

  #if defined(_WIN32)
    // desktop duplication that waits for the screen to change and keeps frames on the gpu.
    // a frame only reaches system memory when the encoder can't take the texture
    struct desktop_capture_t {
      ~desktop_capture_t() { close(); }

      bool open();
      void close();
      // true when texture holds a new frame, false after timeout_ms without one
      bool acquire(std::uint32_t timeout_ms);
      // the last acquired frame in system memory, one frame late so the copy never stalls on the gpu
      std::uint8_t* map_for_cpu(bool new_frame);

      // 0: the whole monitor. Otherwise the top level window (an HWND) that is captured instead, also when other
      // windows cover it. Set from any thread, it takes effect at the next frame. Goes back to 0 when the window closes
      std::atomic<std::uintptr_t> target_window {0};
      // the monitor captured while no window is, an HMONITOR; 0 is the main display. Set from any thread, a monitor
      // on the capture's own graphics card is switched to at the next frame
      std::atomic<std::uintptr_t> target_monitor {0};

      // true while the captured screen is in HDR and its frames are tone mapped. hdr_exposure darkens (< 1) or
      // brightens (> 1) them, set from any thread
      std::atomic<bool> hdr_active {false};
      std::atomic<f32_t> hdr_exposure {0.8f};

      MD_SCR_Geometry_t Geometry{};
      // where the captured monitor sits on the virtual desktop
      fan::vec2i desktop_origin = 0;
      ID3D11Device* device = nullptr;
      ID3D11DeviceContext* context = nullptr;
      IDXGIOutput1* output = nullptr;
      IDXGIOutputDuplication* duplication = nullptr;
      ID3D11Texture2D* texture = nullptr;
      ID3D11Texture2D* staging[2] = {nullptr, nullptr};
      std::uint32_t staging_index = 0;
      bool staging_ready = false;
      std::uint8_t* mapped = nullptr;
      std::uint32_t mapped_index = 0;
      bool has_frame = false;
      std::chrono::steady_clock::time_point next_reopen;
      // the target_monitor the output was picked for, and where that monitor sits on the virtual desktop
      std::uintptr_t opened_monitor = 0;
      fan::vec2i monitor_origin = 0;
      void switch_output(std::uintptr_t monitor);

    private:
      bool acquire_window(std::uint32_t timeout_ms);
      void release_window_buffer();
      void* window_dc = nullptr;
      void* window_bitmap = nullptr;
      void* window_old_bitmap = nullptr;
      std::uint8_t* window_bits = nullptr;
      std::int32_t window_buffer_width = 0;
      std::int32_t window_buffer_height = 0;
      bool create_textures(std::uint32_t width, std::uint32_t height);
      void release_textures();
      void release_duplication();

      // An HDR screen is captured in its own format (16 bit float, linear, 1.0 = 80 nits) and tone mapped on the gpu
      // to the 8 bit picture the rest takes. The old way had Windows do it, which came out washed out
      HRESULT duplicate();
      f32_t query_sdr_white_nits();
      bool create_tonemap();
      bool tonemap_frame(ID3D11Texture2D* desktop, std::uint32_t width, std::uint32_t height);
      void release_tonemap();
      ID3D11Texture2D* hdr_texture = nullptr;
      ID3D11ShaderResourceView* hdr_view = nullptr;
      ID3D11RenderTargetView* target_view = nullptr;
      ID3D11VertexShader* tonemap_vs = nullptr;
      ID3D11PixelShader* tonemap_ps = nullptr;
      ID3D11Buffer* tonemap_params = nullptr;
      f32_t sdr_white_nits = 200.f;
      f32_t peak_nits = 1000.f;
      bool hdr_reported = false;
    };
    using screen_capture_t = desktop_capture_t;
  #else
    using screen_capture_t = MD_SCR_t;
  #endif

    struct camera_device_t {
      std::string id;
      std::string name;
    };
    std::vector<camera_device_t> get_camera_devices();

    struct camera_capture_t {
      camera_capture_t() = default;
      camera_capture_t(const camera_capture_t&) = delete;
      camera_capture_t& operator=(const camera_capture_t&) = delete;
      ~camera_capture_t() {
        close();
      }
      bool open(const std::string& device_id = {}, std::uint32_t wanted_width = 1280, std::uint32_t wanted_height = 720, std::uint32_t wanted_fps = 30);
      void close();
      bool is_open() const {
        return impl != nullptr;
      }
      std::uint8_t* read(std::uint32_t timeout_ms);

      std::uint32_t width = 0;
      std::uint32_t height = 0;
      std::string error;
      std::vector<std::uint8_t> frame;
      void* impl = nullptr;
    };

    struct resolution_manager_t {
      struct detected_info_t {
        std::uint32_t screen_width, screen_height;

        f32_t screen_aspect;
        resolution_system_t::resolution_t optimal_resolution;
        std::vector<resolution_system_t::resolution_t> matching_aspect_resolutions;
      };

      detected_info_t detected_info;
      bool auto_detected = false;

      void detect_and_set_optimal(screen_encode_t& encoder);

      std::vector<std::pair<std::string, f32_t>> get_aspect_ratio_options() {
        return {
          {"16:9", 16.0f / 9.0f},
          {"16:10", 16.0f / 10.0f},
          {"4:3", 4.0f / 3.0f},
          {"21:9", 21.0f / 9.0f},
          {"Auto (Screen Native)", detected_info.screen_aspect}
        };

      }
    };

  #if !defined(_WIN32)
    // Linux: the compositor's frames are taken on a thread of their own. Read on the encoding thread, a frame cost
    // the wait for the compositor (about one refresh) and the encoding one after the other, 16 + 4 ms at 1440p,
    // and that capped the rate at about 50 fps. Here the reads go on while a frame is encoded, the encoder takes the
    // newest one. Three buffers: the one being written, the newest finished one and the one the encoder reads
    // PipeWire's loop (made by MD_SCR_open) only delivers frames to the thread it was made on: the capture is opened,
    // read and closed all on this thread
    struct capture_thread_t {
      ~capture_thread_t() { stop(); }

      // opens the capture on the thread (the portal may ask which screen first) and waits for that, then the thread
      // goes on reading. Returns MD_SCR_open's result, 0 is open
      int open(MD_SCR_t* scr) {
        stop();
        running = true;
        fresh = false;
        writing = 0;
        latest = -1;
        reading = -1;
        open_result = 1;
        opened = false;
        thread = std::thread([this, scr] { run(scr); });
        std::unique_lock<std::mutex> lock(mutex);
        ready.wait(lock, [this] { return opened; });
        int result = open_result;
        lock.unlock();
        if (result != 0) {
          thread.join();
        }
        return result;
      }

      // closes the capture: returns after the read in progress (at most about a second, the read's own timeout)
      void stop() {
        if (!thread.joinable()) {
          return;
        }
        running = false;
        ready.notify_all();
        thread.join();
      }

      bool active() const { return thread.joinable(); }

      // the newest frame not yet taken, waiting up to wait_ms for one. It stays valid until the next take
      std::uint8_t* take(std::uint32_t wait_ms) {
        std::unique_lock<std::mutex> lock(mutex);
        if (!fresh) {
          ready.wait_for(lock, std::chrono::milliseconds(wait_ms), [this] { return fresh || !running; });
        }
        if (!fresh || latest < 0) {
          return nullptr;
        }
        fresh = false;
        reading = latest;
        return slots[reading];
      }

      void run(MD_SCR_t* scr) {
        int result = MD_SCR_open(scr);
        {
          std::lock_guard<std::mutex> lock(mutex);
          open_result = result;
          opened = true;
        }
        ready.notify_all();
        if (result != 0) {
          running = false;
          return;
        }
        while (running) {
          auto start = std::chrono::steady_clock::now();
          std::uint8_t* frame = MD_SCR_read(scr);
          if (!frame) {
            // no frame within the read's timeout, or the capture failed: the encoder repeats the last one meanwhile
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
          }
          // No copy: the frame is in WITCH's buffer (malloc'd, freed by MD_SCR_close), which is swapped for one of
          // ours of the same size. WITCH fills that one next, the encoder reads the one that came out. A copy here cost
          // its time on every frame, GNOME only makes the next frame once this one is taken
          std::size_t size = scr->pw_ctx ? scr->pw_ctx->frame_size : 0;
          int target;
          {
            std::lock_guard<std::mutex> lock(mutex);
            target = writing;
          }
          if (size == 0 || frame != scr->pw_ctx->frame_data) {
            continue;
          }
          if (slots[target] == nullptr || slot_size[target] != size) {
            std::free(slots[target]);
            slots[target] = (std::uint8_t*)std::malloc(size);
            slot_size[target] = slots[target] ? size : 0;
            if (!slots[target]) {
              continue;
            }
          }
          std::uint8_t* spare = slots[target];
          slots[target] = frame;
          scr->pw_ctx->frame_data = spare;
          std::uint64_t took = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
          {
            std::lock_guard<std::mutex> lock(mutex);
            latest = target;
            fresh = true;
            // the next one goes where neither the newest nor the one being encoded is
            for (int i = 0; i < 3; ++i) {
              if (i != latest && i != reading) {
                writing = i;
                break;
              }
            }
          }
          frames.fetch_add(1, std::memory_order_relaxed);
          read_ns.fetch_add(took, std::memory_order_relaxed);
          last_read_ns.store(took, std::memory_order_relaxed);
          ready.notify_one();
        }
        // WITCH frees the buffer it holds, ours are freed here
        MD_SCR_close(scr);
        std::lock_guard<std::mutex> lock(mutex);
        for (int i = 0; i < 3; ++i) {
          std::free(slots[i]);
          slots[i] = nullptr;
          slot_size[i] = 0;
        }
        latest = reading = -1;
        fresh = false;
      }

      std::thread thread;
      std::atomic<bool> running = false;
      std::mutex mutex;
      std::condition_variable ready;
      // malloc'd, like WITCH's frame buffer they are swapped with
      std::uint8_t* slots[3] = {};
      std::size_t slot_size[3] = {};
      int writing = 0, latest = -1, reading = -1;
      bool fresh = false;
      bool opened = false;
      int open_result = 1;
      // frames the compositor gave and the time their reads took (waiting included), for the perf log
      std::atomic<std::uint64_t> frames = 0;
      std::atomic<std::uint64_t> read_ns = 0;
      std::atomic<std::uint64_t> last_read_ns = 0;
    };
  #endif

    struct screen_encode_t {
      bool update_scaling_quality(codec_config_t::scaling_quality_e new_quality) {
        config_.scaling_quality = new_quality;

        std::lock_guard<std::mutex> lock(mutex);
        update_flags |= codec_update_e::scaling_quality;
        return true;
      }

      // starts from the fixed defaults below, the capture's own size replaces them at the first read
      bool open() {
        mdscr_initialized_ = false;
        capture_open_failed = false;

        config_.frame_rate = 30;
        config_.codec = codec_config_t::H264;
        config_.rate_control = codec_config_t::VBR;
        config_.bitrate = 10000000;
        config_.width = 1280;
        config_.height = 720;

        if (!encoder_.open(config_, fan::vec2(config_.width, config_.height))) {
          fan::print_impl("Failed to open encoder");
          return false;

        }

        std::lock_guard<std::mutex> lock(mutex);
        update_flags = 0;
        return true;

      }

      bool ensure_mdscr_initialized() {
        if (mdscr_initialized_) return true;

        if (camera_active_) {
          auto now = std::chrono::steady_clock::now();
          if (now < camera_retry_at_) return false;
          if (!camera.open(camera_id_, 1280, 720, (std::uint32_t)std::max(config_.frame_rate, 1))) {
            fan::print_impl("failed to open the camera: " + camera.error);
            camera_retry_at_ = now + std::chrono::seconds(2);
            return false;
          }
        #if !defined(_WIN32)
          std::memset(&mdscr, 0, sizeof(mdscr));
        #endif
          mdscr.Geometry.Resolution.x = camera.width;
          mdscr.Geometry.Resolution.y = camera.height;
          mdscr.Geometry.LineSize = camera.width * 4;
        }
        else {
        #if defined(_WIN32)
          if (!screen_open_ && !mdscr.open()) {
            fan::print_impl("failed to open screen");
            capture_open_failed = true;
            return false;
          }
          screen_open_ = false;
          encoder_.gpu_device_ = mdscr.device;
        #else
          std::memset(&mdscr, 0, sizeof(mdscr));
        #if defined(MD_SCR_HAS_RESTORE_TOKEN)
          // a token from the share before (kept over an update) shares the same screen or window without the dialog.
          // It is used once: the next share after this one asks again, as usual
          MD_SCR_set_restore_token(take_resume_token().c_str());
        #endif
          if (int ret = capture_thread_.open(&mdscr); ret != 0) {
            fan::print_impl("failed to open screen:" + std::to_string(ret));
            // a cancelled portal dialog must not be opened again on the next read
            capture_open_failed = true;
            return false;
          }
        #if defined(MD_SCR_HAS_RESTORE_TOKEN)
          {
            std::lock_guard<std::mutex> lock(restore_token_mutex_);
            restore_token_ = MD_SCR_get_restore_token();
          }
        #endif
        #endif
        }

        resolution_manager.detect_and_set_optimal(*this);

        {
          // this open applies all of config_, a change queued before it would only reopen the same encoder again
          std::lock_guard<std::mutex> lock(mutex);
          update_flags = 0;
        }
        if (!encoder_.open(config_, fan::vec2(mdscr.Geometry.Resolution.x, mdscr.Geometry.Resolution.y))) {
          fan::print_impl("Failed to reopen encoder with screen resolution");

          return false;
        }

        mdscr_initialized_ = true;
        capture_open_failed = false;
        return true;

      }

      // ends the capture, the next screen_read opens it again. On Wayland that asks which screen or window to
      // share again, and picks up a changed resolution or rotation. Windows captures the desktop and has nothing to redo
      void close_screen() {
        if (camera_active_) {
          camera.close();
          screen_buffer = nullptr;
          mdscr_initialized_ = false;
          return;
        }
      #if !defined(_WIN32)
        if (!mdscr_initialized_) return;
        // closes the capture on its thread
        capture_thread_.stop();
        screen_buffer = nullptr;
        mdscr_initialized_ = false;
      #endif
      }

      // Linux, the portal's restore tokens. restore_token(): the one of the screen or window shared now (empty when the
      // portal gave none), for keeping over an update. set_resume_token(): given to the next opening of the capture,
      // once, which then shares that screen or window without the dialog. The gui thread reads and sets them
      std::string restore_token() {
        std::lock_guard<std::mutex> lock(restore_token_mutex_);
        return restore_token_;
      }
      void set_resume_token(const std::string& token) {
        std::lock_guard<std::mutex> lock(restore_token_mutex_);
        resume_token_ = token;
      }
      std::string take_resume_token() {
        std::lock_guard<std::mutex> lock(restore_token_mutex_);
        return std::exchange(resume_token_, std::string());
      }
      std::mutex restore_token_mutex_;
      std::string restore_token_;
      std::string resume_token_;

      ~screen_encode_t() {
      #if !defined(_WIN32)
        capture_thread_.stop();
      #endif
        encoder_.close();
      }
      bool is_initialized() const { return encoder_.is_initialized();

      }

      static std::vector<std::string> get_encoders() { return libav_encoder_t::get_available_encoders();

      }

      struct encode_data_t {
        std::vector<std::uint8_t> data;

        bool is_keyframe = false;
        std::int64_t timestamp = 0;
      };

      bool encode_write();

      std::size_t encode_read(std::uint8_t** data) {
        std::lock_guard<std::mutex> lock(data_mutex);

        if (encoded_packets_.empty()) return 0;

        current_packet_ = std::move(encoded_packets_.front());
        encoded_packets_.erase(encoded_packets_.begin());
        *data = current_packet_.data.data();
        return current_packet_.data.size();

      }

      void sleep_thread();

      void set_camera(bool enabled, const std::string& device_id) {
        std::lock_guard<std::mutex> lock(source_mutex_);
        if (enabled == want_camera_ && (!enabled || device_id == wanted_camera_id_)) return;
        want_camera_ = enabled;
        wanted_camera_id_ = device_id;
        source_changed_ = true;
      }
      bool is_camera() const {
        return camera_active_;
      }
      void poll_source() {
        if (!source_changed_.exchange(false)) return;
        bool enabled;
        std::string device_id;
        {
          std::lock_guard<std::mutex> lock(source_mutex_);
          enabled = want_camera_;
          device_id = wanted_camera_id_;
        }
        if (camera_active_) {
          camera.close();
        #if defined(_WIN32)
          if (screen_open_) mdscr.Geometry = screen_geometry_;
        #endif
        }
        else if (mdscr_initialized_) {
        #if defined(_WIN32)
          screen_geometry_ = mdscr.Geometry;
          screen_open_ = true;
        #else
          capture_thread_.stop();
        #endif
        }
      #if defined(_WIN32)
        encoder_.gpu_device_ = nullptr;
      #endif
        screen_buffer = nullptr;
        mdscr_initialized_ = false;
        camera_active_ = enabled;
        camera_id_ = device_id;
        camera_retry_at_ = {};
      }

      bool read_camera() {
        auto now = std::chrono::steady_clock::now();
        if (std::uint8_t* frame = camera.read((std::uint32_t)std::max(1000 / std::max(config_.frame_rate, 1), 1))) {
          screen_buffer = frame;
          last_read = now;
          return true;
        }
        if (!camera.is_open()) {
          screen_buffer = nullptr;
          mdscr_initialized_ = false;
          camera_retry_at_ = now + std::chrono::seconds(2);
          return false;
        }
        if (!screen_buffer || (!(encode_write_flags & codec_update_e::force_keyframe) && now - last_read < std::chrono::milliseconds(100))) {
          return false;
        }
        last_read = now;
        return true;
      }

      bool screen_read() {
        poll_source();
        if (!ensure_mdscr_initialized()) return false;
        if (camera_active_) return read_camera();

      #if defined(_WIN32)
        return read_desktop();
      #else
        auto now = std::chrono::steady_clock::now();
        // the capture thread's counts are the screen's, what this thread takes of them is the encoder's
        new_frames = capture_thread_.frames.load(std::memory_order_relaxed);
        screen_wait_ns = capture_thread_.read_ns.load(std::memory_order_relaxed);
        capture_time_ns = capture_thread_.last_read_ns.load(std::memory_order_relaxed);
        std::uint32_t wait_ms = (std::max)(1000 / (std::max)(config_.frame_rate, 1), 1);
        if (std::uint8_t* frame = capture_thread_.take(wait_ms)) {
          screen_buffer = frame;
          last_read = now;
          return true;
        }
        if (!screen_buffer || (!(encode_write_flags & codec_update_e::force_keyframe) && now - last_read < std::chrono::milliseconds(100))) {
          return false;
        }
        last_read = now;
        return true;
      #endif
      }

    #if defined(_WIN32)
      bool read_desktop();
    #endif

      void reset_to_auto_resolution() {
        user_set_resolution = false;

        user_width = 0;
        user_height = 0;
      }

      void set_user_resolution(std::uint32_t width, std::uint32_t height) {
        config_.width = width;

        config_.height = height;
        user_set_resolution = true;
        user_width = width;
        user_height = height;
        std::lock_guard<std::mutex> lock(mutex);
        update_flags |= codec_update_e::codec;

      }

      codec_config_t config_;
      std::string name = "libx264";
      std::uintptr_t new_codec = 0;
      // set by the gui and network threads, read and cleared by the encoding thread
      std::atomic<std::uint8_t> update_flags = 0;

      std::atomic<std::uint8_t> encode_write_flags = 0;
      bool user_set_resolution = false;
      std::uint32_t user_width = 0, user_height = 0;
      bool mdscr_initialized_ = false;
      std::atomic<bool> capture_open_failed{false};

      // read and clear by the cli after a failed screen_read: the portal was cancelled or the capture could not open
      bool take_capture_open_failed() { return capture_open_failed.exchange(false); }

      resolution_manager_t resolution_manager;

      libav_encoder_t encoder_;
      screen_capture_t mdscr;
      std::uint8_t* screen_buffer = nullptr;
    #if !defined(_WIN32)
      capture_thread_t capture_thread_;
    #endif
      camera_capture_t camera;
      bool camera_active_ = false;
      std::string camera_id_;
      std::chrono::steady_clock::time_point camera_retry_at_{};
      std::mutex source_mutex_;
      bool want_camera_ = false;
      std::string wanted_camera_id_;
      std::atomic<bool> source_changed_ = false;
    #if defined(_WIN32)
      bool screen_open_ = false;
      decltype(desktop_capture_t::Geometry) screen_geometry_{};
    #endif
      // time spent copying the last frame, not waiting for it
      std::uint64_t capture_time_ns = 0;
      // new frames the screen gave and the time spent getting them (waiting for the screen included), summed since
      // the start: the perf log reads the change per second, to tell a screen that gives few frames from a slow capture
      std::uint64_t new_frames = 0;
      std::uint64_t screen_wait_ns = 0;
      std::chrono::steady_clock::time_point last_read;

      std::vector<encode_data_t> encoded_packets_;
      encode_data_t current_packet_;
      std::mutex data_mutex;
      std::mutex mutex;

      std::int64_t frame_timestamp_ = 0;
      std::uint64_t encoder_start_time_ = fan::time::now();
      std::uint64_t frame_process_start_time_ = encoder_start_time_;
    };

    struct screen_decode_t {
      bool open() {
        if (!decoder_.open()) {
          fan::print_impl("Failed to open decoder");

          return false;
        }
        gpu_bridge_install();
        return true;

      }

      void close() { decoder_.close(); }
      ~screen_decode_t() { close();
        gpu_bridge_destroy();
      }

      bool is_initialized() const { return decoder_.initialized_;

      }

      void graphics_queue_callback(const std::function<void()>& f) {
        std::lock_guard<std::mutex> lock(mutex);

        graphics_queue.emplace_back(f);
      }

      static std::vector<std::string> get_decoders() {
        std::vector<std::string> decoders;

        auto hw_decoders = libav_decoder_t::get_available_hw_decoders();
        decoders.insert(decoders.end(), hw_decoders.begin(), hw_decoders.end());
        decoders.push_back("h264");
        decoders.push_back("hevc");
        decoders.push_back("av1");
        decoders.push_back("auto-detect");
        return decoders;

      }

      struct decode_data_t {
        std::array<std::vector<std::uint8_t>, 4> data;

        std::array<fan::vec2ui, 4> stride;
        fan::vec2ui image_size = { 0, 0 };
        std::uint8_t pixel_format = 0;
        std::uint8_t type = 0;
        // the picture stayed on the GPU (zero copy): it is in this slot, data is empty. Shown with show_gpu_frame
        int gpu_slot = -1;
        std::uint64_t gpu_frame_id = 0;

        // 0 = success, >250 = error codes
      };

      // ---- zero copy: an NVIDIA decoder's frames go from CUDA memory into memory Vulkan reads, both on the GPU,
      // instead of down into system memory and up again. Off by itself where it can't work (no NVIDIA, CUDA on
      // another GPU than the window, a driver without the Vulkan extension), the frames then download as before
      std::atomic<bool> zero_copy {true};
      // the last frames went this way
      bool zero_copy_active() const { return zero_copy_used_.load(std::memory_order_relaxed); }
      // on the render thread: the frame's planes into the renderer's images. False when the frame is gone
      // (its slot was written again meanwhile), nothing is changed then
      bool show_gpu_frame(const decode_data_t& frame, fan::graphics::shape_t& universal_image_renderer);
      void gpu_bridge_install();
      void gpu_bridge_destroy();
      void* gpu_bridge_ = nullptr;
      std::atomic<bool> zero_copy_used_ {false};

      decode_data_t decode(void* data, std::uintptr_t length, fan::graphics::shape_t& universal_image_renderer);

      // gives a frame's buffers back to the decoder, call from the thread that decodes
      void recycle(decode_data_t& frame) {
        decoder_.spare_planes_.assign(std::make_move_iterator(frame.data.begin()), std::make_move_iterator(frame.data.end()));
      }

      std::vector<std::uint8_t> last_sequence_header;
      std::string name = "auto-detect";
      std::uintptr_t new_codec = 0;
      std::atomic<std::uint8_t> update_flags = 0;

      fan::vec2ui frame_size_ = { 1, 1 };
      fan::vec2ui decoded_size;

      libav_decoder_t decoder_;
      std::mutex mutex;
      std::vector<std::function<void()>> graphics_queue;
      std::function<void()> reload_codec_cb = [] {};

    };

    void resolution_manager_t::detect_and_set_optimal(screen_encode_t& encoder) {
      if (auto_detected) return;
      detected_info.screen_width = encoder.mdscr.Geometry.Resolution.x;
      detected_info.screen_height = encoder.mdscr.Geometry.Resolution.y;

      detected_info.screen_aspect = static_cast<f32_t>(detected_info.screen_width) / detected_info.screen_height;
      detected_info.optimal_resolution = resolution_system_t::get_optimal_resolution(
        detected_info.screen_width, detected_info.screen_height
      );

      detected_info.matching_aspect_resolutions = resolution_system_t::get_resolutions_by_aspect(
        detected_info.screen_aspect, 0.02f
      );

      auto_detected = true;
    }

  }
}
#endif

#endif