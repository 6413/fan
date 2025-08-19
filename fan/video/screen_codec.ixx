module;

#include <WITCH/WITCH.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

#include <string>
#include <mutex>
#include <array>
#include <cstring>
#include <atomic>
#include <functional>
#include <thread>
#include <chrono>
#include <vector>
#include <memory>
#include <unordered_map>
#include <map>

#include <WITCH/PR/PR.h>
#include <WITCH/MD/SCR/SCR.h>

export module fan.graphics.video.screen_codec;

export import fan.graphics.loco;
import fan.time;
import fan.fmt;

export namespace fan {
  namespace graphics {

    struct codec_config_t {
      enum codec_type_e {
        H264 = 0,
        H265,
        AV1
      };

      enum rate_control_e {
        CBR = 0,  // Constant bitrate
        VBR,      // Variable bitrate
        CRF       // Constant rate factor
      };

      enum scaling_quality_e {
        SCALING_FASTEST = 0,     // SWS_POINT
        SCALING_FAST,            // SWS_FAST_BILINEAR  
        SCALING_BALANCED,        // SWS_BILINEAR
        SCALING_HIGH,            // SWS_BICUBIC
        SCALING_HIGHEST          // SWS_LANCZOS
      };

      scaling_quality_e scaling_quality = SCALING_FAST;

      codec_type_e codec = H264;
      rate_control_e rate_control = CBR;
      int bitrate = 2000000;
      int crf_value = 23;      // For CRF mode
      int frame_rate = 30;
      int width = 1280;
      int height = 720;
      int gop_size = 30;       // I-frame interval
      bool use_hardware = true; // Try hardware acceleration
    };

    struct codec_update_e {
      enum {
        codec = 1 << 0,
        rate_control = 1 << 1,
        frame_rate = 1 << 2,
        force_keyframe = 1 << 3,
        scaling_quality = 1 << 4
      };
    };

    int get_sws_flags(codec_config_t::scaling_quality_e quality) {
      switch (quality) {
      case codec_config_t::SCALING_FASTEST:
        return SWS_POINT;
      case codec_config_t::SCALING_FAST:
        return SWS_FAST_BILINEAR;
      case codec_config_t::SCALING_BALANCED:
        return SWS_BILINEAR | SWS_ACCURATE_RND;
      case codec_config_t::SCALING_HIGH:
        return SWS_BICUBIC | SWS_ACCURATE_RND;
      case codec_config_t::SCALING_HIGHEST:
        return SWS_LANCZOS | SWS_FULL_CHR_H_INT | SWS_ACCURATE_RND;
      default:
        return SWS_FAST_BILINEAR;
      }
    }

    const char* get_scaling_quality_name(codec_config_t::scaling_quality_e quality) {
      switch (quality) {
      case codec_config_t::SCALING_FASTEST: return "Fastest (Point)";
      case codec_config_t::SCALING_FAST: return "Fast (Bilinear)";
      case codec_config_t::SCALING_BALANCED: return "Balanced (Bilinear+)";
      case codec_config_t::SCALING_HIGH: return "High (Bicubic)";
      case codec_config_t::SCALING_HIGHEST: return "Highest (Lanczos)";
      default: return "Unknown";
      }
    }

#define __debug_prints 0

    struct libav_encoder_t {
      struct encoder_state_t {
        const AVCodec* codec = nullptr;
        AVCodecContext* codec_ctx = nullptr;
        AVFrame* frame = nullptr;
        AVPacket* packet = nullptr;
        SwsContext* sws_ctx = nullptr;
        uint8_t* converted_data[4] = { nullptr };
        int converted_linesize[4] = { 0 };
        std::string codec_name;
        bool initialized = false;
        AVBufferRef* hw_device_ctx = nullptr;
        AVHWDeviceType hw_type = AV_HWDEVICE_TYPE_NONE;
        fan::vec2 screencap_res;
        codec_config_t config;
      };

      std::atomic<encoder_state_t*> active_encoder_{ nullptr };
      std::atomic<encoder_state_t*> pending_cleanup_{ nullptr };
      encoder_state_t encoder_a_, encoder_b_;

      codec_config_t config_;
      std::string codec_name_;
      bool initialized_ = false;
      fan::vec2 screencap_res_;

      std::atomic<bool> needs_codec_reinit_{ false };
      codec_config_t pending_config_;

      static thread_local std::chrono::steady_clock::time_point last_cleanup_time_;

      static std::unordered_map<codec_config_t::codec_type_e, std::vector<std::string>> get_codec_names() {
        return {
          {codec_config_t::H264, {"h264_nvenc", "h264_amf", "h264_vaapi", "libx264"}},
          {codec_config_t::H265, {"hevc_nvenc", "hevc_amf", "hevc_vaapi", "libx265"}},
          {codec_config_t::AV1, {"av1_nvenc", "av1_amf", "av1_vaapi", "libaom-av1", "libsvtav1"}}
        };
      }

      bool update_scaling_quality(codec_config_t::scaling_quality_e new_quality) {
        encoder_state_t* encoder = active_encoder_.load();
        if (!encoder || !encoder->sws_ctx) return false;

        sws_freeContext(encoder->sws_ctx);
        encoder->sws_ctx = nullptr;

        int sws_flags = fan::graphics::get_sws_flags(new_quality);

        encoder->sws_ctx = sws_getContext(
          encoder->screencap_res.x, encoder->screencap_res.y, AV_PIX_FMT_BGRA,
          encoder->config.width, encoder->config.height, encoder->codec_ctx->pix_fmt,
          sws_flags,
          nullptr, nullptr, nullptr
        );

        if (!encoder->sws_ctx) {
#if __debug_prints
          fan::print_throttled("encoder_scaling_quality_update_failed", 2000);
#endif
          return false;
        }

        encoder->config.scaling_quality = new_quality;
        config_.scaling_quality = new_quality;
        return true;
      }

      bool init_hardware_acceleration(encoder_state_t* encoder, const codec_config_t& config) {
        if (!config.use_hardware) return false;

        std::vector<AVHWDeviceType> hw_types = {
          AV_HWDEVICE_TYPE_CUDA,
          AV_HWDEVICE_TYPE_D3D11VA,
          AV_HWDEVICE_TYPE_DXVA2,
          AV_HWDEVICE_TYPE_VAAPI,
          AV_HWDEVICE_TYPE_VIDEOTOOLBOX
        };

        for (auto type : hw_types) {
          if (av_hwdevice_ctx_create(&encoder->hw_device_ctx, type, nullptr, nullptr, 0) == 0) {
            encoder->hw_type = type;
#if __debug_prints
            const char* type_name = "unknown";
            switch (type) {
            case AV_HWDEVICE_TYPE_CUDA: type_name = "cuda"; break;
            case AV_HWDEVICE_TYPE_D3D11VA: type_name = "d3d11va"; break;
            case AV_HWDEVICE_TYPE_DXVA2: type_name = "dxva2"; break;
            case AV_HWDEVICE_TYPE_VAAPI: type_name = "vaapi"; break;
            case AV_HWDEVICE_TYPE_VIDEOTOOLBOX: type_name = "videotoolbox"; break;
            }
            fan::print_throttled("encoder_hw_accel_init: " + std::string(type_name), 5000);
#endif
            return true;
          }
        }
        return false;
      }

      AVPixelFormat choose_encoder_pixel_format(const std::string& encoder_name) {
        if (encoder_name.find("nvenc") != std::string::npos ||
          encoder_name.find("amf") != std::string::npos ||
          encoder_name.find("qsv") != std::string::npos ||
          encoder_name.find("vaapi") != std::string::npos ||
          encoder_name.find("videotoolbox") != std::string::npos) {
          return AV_PIX_FMT_NV12;
        }
        return AV_PIX_FMT_YUV420P;
      }

      bool find_and_init_codec(encoder_state_t* encoder, const codec_config_t& config) {
        auto codec_names = get_codec_names();
        auto& names = codec_names[config.codec];

        for (const auto& name : names) {
          encoder->codec = avcodec_find_encoder_by_name(name.c_str());
          if (!encoder->codec) continue;

          encoder->codec_name = name;
          encoder->codec_ctx = avcodec_alloc_context3(encoder->codec);
          if (!encoder->codec_ctx) continue;

          encoder->codec_ctx->width = config.width;
          encoder->codec_ctx->height = config.height;
          encoder->codec_ctx->time_base = { 1, config.frame_rate };
          encoder->codec_ctx->framerate = { config.frame_rate, 1 };
          encoder->codec_ctx->gop_size = config.frame_rate >= 144 ? config.frame_rate / 3 :
            config.frame_rate >= 120 ? config.frame_rate / 2 : config.gop_size;
          encoder->codec_ctx->max_b_frames = config.frame_rate >= 60 ? 0 : 1;
          encoder->codec_ctx->pix_fmt = choose_encoder_pixel_format(name);

          if (encoder->hw_device_ctx && name.find("nvenc") != std::string::npos) {
            encoder->codec_ctx->hw_device_ctx = av_buffer_ref(encoder->hw_device_ctx);
          }

          if (name.find("nvenc") != std::string::npos) {
            encoder->codec_ctx->bit_rate = config.bitrate;
            encoder->codec_ctx->rc_buffer_size = config.bitrate / 4;
            encoder->codec_ctx->rc_max_rate = config.bitrate;
            encoder->codec_ctx->rc_min_rate = config.bitrate;

            av_opt_set(encoder->codec_ctx->priv_data, "preset", "p1", 0);
            av_opt_set(encoder->codec_ctx->priv_data, "tune", "ll", 0);
            av_opt_set_int(encoder->codec_ctx->priv_data, "delay", 2, 0);
            av_opt_set_int(encoder->codec_ctx->priv_data, "async_depth", 8, 0);
            av_opt_set_int(encoder->codec_ctx->priv_data, "zerolatency", 0, 0);

            uint32_t surfaces = config.frame_rate >= 144 ? 48 :
              config.frame_rate >= 120 ? 40 : 32;
            av_opt_set_int(encoder->codec_ctx->priv_data, "surfaces", surfaces, 0);
            av_opt_set_int(encoder->codec_ctx->priv_data, "bf", 0, 0);

            av_opt_set(encoder->codec_ctx->priv_data, "spatial_aq", "0", 0);
            av_opt_set(encoder->codec_ctx->priv_data, "temporal_aq", "0", 0);
            av_opt_set_int(encoder->codec_ctx->priv_data, "lookahead_depth", 0, 0);
            av_opt_set_int(encoder->codec_ctx->priv_data, "rc_lookahead", 0, 0);
            av_opt_set(encoder->codec_ctx->priv_data, "gpu", 0, 0);
            av_opt_set_int(encoder->codec_ctx->priv_data, "strict_gop", 1, 0);
            av_opt_set_int(encoder->codec_ctx->priv_data, "forced-idr", 0, 0);
            av_opt_set_int(encoder->codec_ctx->priv_data, "no-scenecut", 1, 0);

#if __debug_prints
            fan::print("encoder_nvenc_init: " + name +
              " | " + std::to_string(config.width) + "x" + std::to_string(config.height) +
              " | " + std::to_string(config.frame_rate) + "fps" +
              " | " + std::to_string(config.bitrate / 1000) + "kbps" +
              " | " + std::to_string(surfaces) + " surfaces");
#endif

            switch (config.rate_control) {
            case codec_config_t::CBR:
              av_opt_set(encoder->codec_ctx->priv_data, "rc", "cbr", 0);
#if __debug_prints
              fan::print_throttled("encoder_rate_control: cbr", 2000);
#endif
              break;
            case codec_config_t::VBR:
              av_opt_set(encoder->codec_ctx->priv_data, "rc", "vbr", 0);
#if __debug_prints
              fan::print_throttled("encoder_rate_control: vbr", 2000);
#endif
              break;
            case codec_config_t::CRF:
              av_opt_set(encoder->codec_ctx->priv_data, "rc", "constqp", 0);
              av_opt_set_int(encoder->codec_ctx->priv_data, "qp", config.crf_value, 0);
#if __debug_prints
              fan::print_throttled("encoder_rate_control: crf_" + std::to_string(config.crf_value), 2000);
#endif
              break;
            }
          }
          else if (name.find("amf") != std::string::npos) {
            encoder->codec_ctx->bit_rate = config.bitrate;
            av_opt_set(encoder->codec_ctx->priv_data, "quality", "speed", 0);
            av_opt_set(encoder->codec_ctx->priv_data, "rc", "vbr_latency", 0);
            av_opt_set_int(encoder->codec_ctx->priv_data, "preanalysis", 0, 0);
#if __debug_prints
            fan::print("encoder_amf_init: " + name);
#endif
          }
          else if (name.find("vaapi") != std::string::npos) {
            encoder->codec_ctx->bit_rate = config.bitrate;
            av_opt_set(encoder->codec_ctx->priv_data, "low_power", "1", 0);
#if __debug_prints
            fan::print("encoder_vaapi_init: " + name);
#endif
          }
          else if (name == "libx264") {
            encoder->codec_ctx->bit_rate = config.bitrate;
            av_opt_set(encoder->codec_ctx->priv_data, "preset", "ultrafast", 0);
            av_opt_set(encoder->codec_ctx->priv_data, "tune", "zerolatency", 0);
#if __debug_prints
            fan::print("encoder_x264_init: software_fallback");
#endif
          }
          else if (name == "libx265") {
            encoder->codec_ctx->bit_rate = config.bitrate;
            av_opt_set(encoder->codec_ctx->priv_data, "preset", "ultrafast", 0);
            av_opt_set(encoder->codec_ctx->priv_data, "tune", "zerolatency", 0);
#if __debug_prints
            fan::print("encoder_x265_init: software_fallback");
#endif
          }
          else {
            encoder->codec_ctx->bit_rate = config.bitrate;
#if __debug_prints
            fan::print("encoder_generic_init: " + name);
#endif
          }

          if (avcodec_open2(encoder->codec_ctx, encoder->codec, nullptr) == 0) {
#if __debug_prints
            fan::print("encoder_success: " + name);
#endif
            return true;
          }

#if __debug_prints
          fan::print_throttled("encoder_fail: " + name, 1000);
#endif
          avcodec_free_context(&encoder->codec_ctx);
          encoder->codec = nullptr;
        }

#if __debug_prints
        fan::print("encoder_error: no_suitable_codec");
#endif
        return false;
      }

      void cleanup_encoder_state(encoder_state_t* encoder) {
        if (encoder->sws_ctx) {
          sws_freeContext(encoder->sws_ctx);
          encoder->sws_ctx = nullptr;
        }

        if (encoder->converted_data[0]) {
          av_freep(&encoder->converted_data[0]);
        }

        if (encoder->frame) {
          av_frame_free(&encoder->frame);
        }

        if (encoder->packet) {
          av_packet_free(&encoder->packet);
        }

        if (encoder->codec_ctx) {
          avcodec_free_context(&encoder->codec_ctx);
        }

        if (encoder->hw_device_ctx) {
          av_buffer_unref(&encoder->hw_device_ctx);
        }

        encoder->initialized = false;
        encoder->codec = nullptr;
        encoder->codec_name.clear();
        encoder->hw_type = AV_HWDEVICE_TYPE_NONE;
      }

      bool init_encoder_state(encoder_state_t* encoder, const codec_config_t& config, const fan::vec2& screencap_res) {
        cleanup_encoder_state(encoder);

        encoder->config = config;
        encoder->screencap_res = screencap_res.x > 0 ? screencap_res : fan::vec2{ config.width, config.height };

        init_hardware_acceleration(encoder, config);

        if (!find_and_init_codec(encoder, config)) {
          return false;
        }

        encoder->frame = av_frame_alloc();
        encoder->packet = av_packet_alloc();
        if (!encoder->frame || !encoder->packet) {
          cleanup_encoder_state(encoder);
          return false;
        }

        encoder->frame->format = encoder->codec_ctx->pix_fmt;
        encoder->frame->width = config.width;
        encoder->frame->height = config.height;

        if (av_frame_get_buffer(encoder->frame, 32) < 0) {
          cleanup_encoder_state(encoder);
          return false;
        }

        int sws_flags = fan::graphics::get_sws_flags(config.scaling_quality);
        encoder->sws_ctx = sws_getContext(
          encoder->screencap_res.x, encoder->screencap_res.y, AV_PIX_FMT_BGRA,
          config.width, config.height, encoder->codec_ctx->pix_fmt,
          sws_flags,
          nullptr, nullptr, nullptr
        );

        if (!encoder->sws_ctx) {
          cleanup_encoder_state(encoder);
          return false;
        }

        encoder->initialized = true;
        return true;
      }

      void maybe_cleanup_old_encoder() {
        auto now = std::chrono::steady_clock::now();
        if (now - last_cleanup_time_ < std::chrono::milliseconds(100)) {
          return;
        }
        last_cleanup_time_ = now;

        encoder_state_t* old = pending_cleanup_.exchange(nullptr);
        if (old) {
          cleanup_encoder_state(old);
        }
      }

    public:
      libav_encoder_t() {
        active_encoder_.store(&encoder_a_);
      }

      ~libav_encoder_t() {
        close();
      }

      bool open(const codec_config_t& config, const fan::vec2& screencap_res = { 0, 0 }) {
        config_ = config;
        screencap_res_ = screencap_res;

        encoder_state_t* encoder = active_encoder_.load();
        if (init_encoder_state(encoder, config, screencap_res)) {
          initialized_ = true;
          codec_name_ = encoder->codec_name;
          return true;
        }
        return false;
      }

      void close() {
        cleanup_encoder_state(&encoder_a_);
        cleanup_encoder_state(&encoder_b_);
        initialized_ = false;
      }

      struct encode_result_t {
        std::vector<uint8_t> data;
        bool is_keyframe = false;
        int64_t pts = 0;
        int64_t dts = 0;

        encode_result_t() = default;
        encode_result_t(const encode_result_t&) = delete;
        encode_result_t& operator=(const encode_result_t&) = delete;
        encode_result_t(encode_result_t&&) = default;
        encode_result_t& operator=(encode_result_t&&) = default;
      };

      std::vector<encode_result_t> encode_frame(uint8_t* bgra_data, int stride, int64_t pts, bool force_keyframe = false) {
        std::vector<encode_result_t> results;
        static uint64_t frame_count = 0;
        static uint64_t total_encode_time = 0;
        static uint64_t total_output_bytes = 0;
        static auto last_stats_time = std::chrono::steady_clock::now();

        auto encode_start = std::chrono::high_resolution_clock::now();

        if (needs_codec_reinit_.load()) {
          encoder_state_t* new_encoder = (active_encoder_.load() == &encoder_a_) ? &encoder_b_ : &encoder_a_;

          if (init_encoder_state(new_encoder, pending_config_, screencap_res_)) {
            encoder_state_t* old = active_encoder_.exchange(new_encoder);
            pending_cleanup_.store(old);
            needs_codec_reinit_.store(false);
            codec_name_ = new_encoder->codec_name;
#if __debug_prints
            fan::print_throttled("encoder_switched: " + codec_name_, 1000);
#endif
          }
        }

        encoder_state_t* encoder = active_encoder_.load();
        if (!encoder || !encoder->initialized) return results;

        if (!bgra_data || stride <= 0) return results;

        const uint8_t* src_data[4] = { bgra_data, nullptr, nullptr, nullptr };
        int src_linesize[4] = { stride, 0, 0, 0 };

        if (sws_scale(encoder->sws_ctx, src_data, src_linesize, 0, encoder->screencap_res.y,
          encoder->frame->data, encoder->frame->linesize) != encoder->config.height) {
          return results;
        }

        encoder->frame->pts = pts;
        encoder->frame->pict_type = force_keyframe ? AV_PICTURE_TYPE_I : AV_PICTURE_TYPE_NONE;

        static uint64_t last_forced_keyframe_pts = 0;
        static uint64_t keyframe_request_count = 0;

        if (force_keyframe) {
          uint64_t frames_since_last = pts - last_forced_keyframe_pts;
          keyframe_request_count++;
#if __debug_prints
          fan::print_throttled("encoder_forced_keyframe: #" + std::to_string(keyframe_request_count) +
            " pts_" + std::to_string(pts) +
            " gap_" + std::to_string(frames_since_last), 1000);
#endif
          last_forced_keyframe_pts = pts;

          if (frames_since_last < 30 && frames_since_last > 0) {
#if __debug_prints
            fan::print_throttled("encoder_keyframe_warning: too_soon_" + std::to_string(frames_since_last), 2000);
#endif
          }
        }

        if (avcodec_send_frame(encoder->codec_ctx, encoder->frame) < 0) {
#if __debug_prints
          fan::print_throttled("encoder_send_frame_failed", 1000);
#endif
          return results;
        }

        int ret;
        while ((ret = avcodec_receive_packet(encoder->codec_ctx, encoder->packet)) >= 0) {
          encode_result_t result;
          result.data.resize(encoder->packet->size);
          std::memcpy(result.data.data(), encoder->packet->data, encoder->packet->size);
          result.is_keyframe = (encoder->packet->flags & AV_PKT_FLAG_KEY) != 0;
          result.pts = encoder->packet->pts;
          result.dts = encoder->packet->dts;

          total_output_bytes += encoder->packet->size;

#if __debug_prints
          if (result.is_keyframe) {
            fan::print_throttled("encoder_keyframe_out: " + std::to_string(encoder->packet->size) + "_bytes", 2000);
          }
#endif

          results.push_back(std::move(result));
          av_packet_unref(encoder->packet);
        }

        auto encode_end = std::chrono::high_resolution_clock::now();
        auto encode_time = std::chrono::duration_cast<std::chrono::microseconds>(encode_end - encode_start).count();
        total_encode_time += encode_time;
        frame_count++;

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_stats_time).count() >= 2000) {
          double avg_encode_time = (double)total_encode_time / frame_count;
          double avg_fps = frame_count * 1000000.0 / total_encode_time;
          double avg_bitrate = (double)total_output_bytes * 8 * config_.frame_rate / frame_count;
          double target_bitrate = config_.bitrate;

          static uint64_t total_keyframes = 0;
          static uint64_t total_frames_counted = 0;
          for (const auto& result : results) {
            total_frames_counted++;
            if (result.is_keyframe) total_keyframes++;
          }

          double keyframe_percentage = total_frames_counted > 0 ?
            (double)total_keyframes * 100.0 / total_frames_counted : 0.0;

#if __debug_prints
          fan::print("encoder_stats: " + codec_name_ +
            " fps_" + std::to_string((int)avg_fps) +
            " time_" + std::to_string((int)avg_encode_time) + "us" +
            " bitrate_" + std::to_string((int)(avg_bitrate / 1000)) + "k/" + std::to_string((int)(target_bitrate / 1000)) + "k" +
            " frames_" + std::to_string(frame_count) +
            " keyframes_" + std::to_string((int)keyframe_percentage) + "%");
#endif

          total_encode_time = 0;
          total_output_bytes = 0;
          frame_count = 0;
          last_stats_time = now;
        }

        maybe_cleanup_old_encoder();
        return results;
      }

      bool update_config(const codec_config_t& new_config, uint8_t update_flags) {
        if (update_flags & codec_update_e::codec) {
          pending_config_ = new_config;
          needs_codec_reinit_.store(true);
          return true;
        }

        config_ = new_config;
        encoder_state_t* encoder = active_encoder_.load();
        if (!encoder) return false;

        encoder->config = new_config;

        if (update_flags & codec_update_e::rate_control) {
          encoder->codec_ctx->bit_rate = encoder->config.bitrate;

          if (encoder->codec_name.find("nvenc") != std::string::npos) {
            encoder->codec_ctx->rc_buffer_size = encoder->config.bitrate / 4;
            encoder->codec_ctx->rc_max_rate = encoder->config.bitrate;
            encoder->codec_ctx->rc_min_rate = encoder->config.bitrate;
          }
        }

        if (update_flags & codec_update_e::frame_rate) {
          encoder->codec_ctx->time_base = { 1, encoder->config.frame_rate };
          encoder->codec_ctx->framerate = { encoder->config.frame_rate, 1 };
          encoder->codec_ctx->gop_size = encoder->config.frame_rate >= 120 ? encoder->config.frame_rate / 2 : encoder->config.gop_size;
          encoder->codec_ctx->max_b_frames = encoder->config.frame_rate >= 60 ? 0 : 1;
        }

        return true;
      }

      const std::string& get_codec_name() const { return codec_name_; }
      bool is_initialized() const { return initialized_; }

      static std::vector<std::string> get_available_encoders() {
        std::vector<std::string> encoders;
        auto codec_names = get_codec_names();

        for (const auto& [type, names] : codec_names) {
          for (const auto& name : names) {
            if (avcodec_find_encoder_by_name(name.c_str())) {
              encoders.push_back(name);
            }
          }
        }
        return encoders;
      }
    };

    thread_local std::chrono::steady_clock::time_point libav_encoder_t::last_cleanup_time_ = std::chrono::steady_clock::now();

    struct libav_decoder_t {
      const AVCodec* codec_ = nullptr;
      AVCodecContext* codec_ctx_ = nullptr;
      AVFrame* frame_ = nullptr;
      AVFrame* hw_frame_ = nullptr;
      AVPacket* packet_ = nullptr;
      SwsContext* sws_ctx_ = nullptr;

      bool initialized_ = false;
      std::string codec_name_;
      bool using_hardware_ = false;
      enum AVPixelFormat hw_pixel_format_ = AV_PIX_FMT_NONE;

      AVBufferRef* hw_device_ctx_ = nullptr;
      AVHWDeviceType hw_type_ = AV_HWDEVICE_TYPE_NONE;

      static std::unordered_map<std::string, std::vector<std::string>> get_hw_decoder_names() {
        return {
          {"h264", {"h264_cuvid", "h264_qsv", "h264_videotoolbox", "h264_vaapi", "h264_dxva2", "h264_d3d11va"}},
          {"hevc", {"hevc_cuvid", "hevc_qsv", "hevc_videotoolbox", "hevc_vaapi", "hevc_dxva2", "hevc_d3d11va"}},
          {"av1", {"av1_cuvid", "av1_qsv", "av1_videotoolbox", "av1_vaapi"}}
        };
      }

      static enum AVPixelFormat get_hw_pixel_format(AVHWDeviceType type) {
        switch (type) {
        case AV_HWDEVICE_TYPE_CUDA: return AV_PIX_FMT_CUDA;
        case AV_HWDEVICE_TYPE_QSV: return AV_PIX_FMT_QSV;
        case AV_HWDEVICE_TYPE_VAAPI: return AV_PIX_FMT_VAAPI;
        case AV_HWDEVICE_TYPE_DXVA2: return AV_PIX_FMT_DXVA2_VLD;
        case AV_HWDEVICE_TYPE_D3D11VA: return AV_PIX_FMT_D3D11;
        case AV_HWDEVICE_TYPE_VIDEOTOOLBOX: return AV_PIX_FMT_VIDEOTOOLBOX;
        default: return AV_PIX_FMT_NONE;
        }
      }

      static enum AVPixelFormat get_hw_format(AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts) {
        libav_decoder_t* decoder = static_cast<libav_decoder_t*>(ctx->opaque);

        for (const enum AVPixelFormat* p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
          if (*p == decoder->hw_pixel_format_) {
            return *p;
          }
        }

#if __debug_prints
        fan::print("decoder_hw_surface_format_failed");
#endif
        return AV_PIX_FMT_NONE;
      }

      bool init_hardware_acceleration() {
        std::vector<std::pair<AVHWDeviceType, const char*>> hw_types = {
          {AV_HWDEVICE_TYPE_CUDA, "cuda"},
          {AV_HWDEVICE_TYPE_QSV, "qsv"},
          {AV_HWDEVICE_TYPE_VAAPI, "vaapi"},
          {AV_HWDEVICE_TYPE_DXVA2, "dxva2"},
          {AV_HWDEVICE_TYPE_D3D11VA, "d3d11va"},
          {AV_HWDEVICE_TYPE_VIDEOTOOLBOX, "videotoolbox"}
        };

        for (const auto& [type, name] : hw_types) {
          if (av_hwdevice_ctx_create(&hw_device_ctx_, type, nullptr, nullptr, 0) == 0) {
            hw_type_ = type;
            hw_pixel_format_ = get_hw_pixel_format(type);
#if __debug_prints
            fan::print("decoder_hw_device_init: " + std::string(name));
#endif
            return true;
          }
        }

#if __debug_prints
        fan::print("decoder_hw_device_unavailable");
#endif
        return false;
      }

      bool find_and_init_hw_decoder(const std::string& codec_type) {
        if (!hw_device_ctx_) return false;

        auto hw_decoders = get_hw_decoder_names();
        auto it = hw_decoders.find(codec_type);
        if (it == hw_decoders.end()) return false;

        for (const auto& decoder_name : it->second) {
          bool type_match = false;
          switch (hw_type_) {
          case AV_HWDEVICE_TYPE_CUDA:
            type_match = decoder_name.find("cuvid") != std::string::npos;
            break;
          case AV_HWDEVICE_TYPE_QSV:
            type_match = decoder_name.find("qsv") != std::string::npos;
            break;
          case AV_HWDEVICE_TYPE_VAAPI:
            type_match = decoder_name.find("vaapi") != std::string::npos;
            break;
          case AV_HWDEVICE_TYPE_DXVA2:
            type_match = decoder_name.find("dxva2") != std::string::npos;
            break;
          case AV_HWDEVICE_TYPE_D3D11VA:
            type_match = decoder_name.find("d3d11va") != std::string::npos;
            break;
          case AV_HWDEVICE_TYPE_VIDEOTOOLBOX:
            type_match = decoder_name.find("videotoolbox") != std::string::npos;
            break;
          default:
            continue;
          }

          if (!type_match) continue;

          codec_ = avcodec_find_decoder_by_name(decoder_name.c_str());
          if (!codec_) continue;

          codec_ctx_ = avcodec_alloc_context3(codec_);
          if (!codec_ctx_) continue;

          codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
          codec_ctx_->get_format = get_hw_format;
          codec_ctx_->opaque = this;

          if (decoder_name.find("cuvid") != std::string::npos) {
            av_opt_set_int(codec_ctx_->priv_data, "surfaces", 16, 0);
            av_opt_set_int(codec_ctx_->priv_data, "async_depth", 4, 0);
            av_opt_set_int(codec_ctx_->priv_data, "drop_second_field", 1, 0);
            av_opt_set(codec_ctx_->priv_data, "deint", "weave", 0);
            av_opt_set_int(codec_ctx_->priv_data, "crop", 0, 0);
            av_opt_set_int(codec_ctx_->priv_data, "resize", 0, 0);
            codec_ctx_->has_b_frames = 0;
            codec_ctx_->delay = 1;
          }
          else if (decoder_name.find("qsv") != std::string::npos) {
            av_opt_set_int(codec_ctx_->priv_data, "async_depth", 1, 0);
            codec_ctx_->delay = 0;
          }
          else if (decoder_name.find("vaapi") != std::string::npos) {
            av_opt_set_int(codec_ctx_->priv_data, "low_power", 1, 0);
            codec_ctx_->delay = 0;
          }

          std::atomic<bool> open_completed{ false };
          std::atomic<int> open_result{ -1 };

          std::thread open_thread([&]() {
            int result = avcodec_open2(codec_ctx_, codec_, nullptr);
            open_result.store(result);
            open_completed.store(true);
            });

          auto start_time = std::chrono::steady_clock::now();
          bool timed_out = false;

          while (!open_completed.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            if (elapsed > std::chrono::seconds(2)) {
              timed_out = true;
              break;
            }
          }

          if (timed_out) {
            if (open_thread.joinable()) {
              open_thread.detach();
            }
            if (codec_ctx_->hw_device_ctx) {
              av_buffer_unref(&codec_ctx_->hw_device_ctx);
            }
            avcodec_free_context(&codec_ctx_);
            codec_ = nullptr;
#if __debug_prints
            fan::print_throttled("decoder_hw_open_timeout: " + decoder_name, 2000);
#endif
            continue;
          }

          if (open_thread.joinable()) {
            open_thread.join();
          }

          if (open_result.load() == 0) {
            codec_name_ = decoder_name;
            using_hardware_ = true;
#if __debug_prints
            fan::print("decoder_hw_success: " + decoder_name);
#endif
            return true;
          }

          if (codec_ctx_->hw_device_ctx) {
            av_buffer_unref(&codec_ctx_->hw_device_ctx);
          }
          avcodec_free_context(&codec_ctx_);
          codec_ = nullptr;
#if __debug_prints
          fan::print_throttled("decoder_hw_fail: " + decoder_name, 1000);
#endif
        }

#if __debug_prints
        fan::print("decoder_hw_error: no_suitable_decoder");
#endif
        return false;
      }

      bool find_and_init_sw_decoder(const std::string& codec_type) {
        std::unordered_map<std::string, std::string> sw_decoders = {
          {"h264", "h264"},
          {"hevc", "hevc"},
          {"av1", "av1"}
        };

        auto it = sw_decoders.find(codec_type);
        if (it == sw_decoders.end()) return false;

        codec_ = avcodec_find_decoder_by_name(it->second.c_str());
        if (!codec_) return false;

        codec_ctx_ = avcodec_alloc_context3(codec_);
        if (!codec_ctx_) return false;

        if (avcodec_open2(codec_ctx_, codec_, nullptr) == 0) {
          codec_name_ = it->second;
          using_hardware_ = false;
#if __debug_prints
          fan::print("decoder_sw_success: " + it->second);
#endif
          return true;
        }

        avcodec_free_context(&codec_ctx_);
        codec_ = nullptr;
#if __debug_prints
        fan::print_throttled("decoder_sw_fail: " + it->second, 1000);
#endif
        return false;
      }

      std::string detect_codec_from_data(const uint8_t* data, size_t size) {
        if (size < 4) return "h264";

        if (data[0] == 0x00 && data[1] == 0x00 && data[2] == 0x00 && data[3] == 0x01) {
          if (size >= 5) {
            uint8_t nal_type = data[4] & 0x1F;
            if (nal_type <= 23) return "h264";
          }
          return "h264";
        }

        if (size >= 5 && data[0] == 0x00 && data[1] == 0x00 && data[2] == 0x00 && data[3] == 0x01) {
          uint8_t nal_type = (data[4] & 0x7E) >> 1;
          if (nal_type >= 32 && nal_type <= 63) return "hevc";
        }

        if (size >= 2) {
          uint8_t obu_type = (data[0] >> 3) & 0x0F;
          if (obu_type >= 1 && obu_type <= 8) return "av1";
        }

        return "h264";
      }

      libav_decoder_t() {
        packet_ = av_packet_alloc();
      }

      ~libav_decoder_t() {
        close();
        if (packet_) av_packet_free(&packet_);
      }

      bool open() {
        close();

        frame_ = av_frame_alloc();
        if (!frame_) {
#if __debug_prints
          fan::print_throttled("decoder_open_fail: frame_alloc", 1000);
#endif
          return false;
        }

        hw_frame_ = av_frame_alloc();
        if (!hw_frame_) {
          av_frame_free(&frame_);
#if __debug_prints
          fan::print_throttled("decoder_open_fail: hw_frame_alloc", 1000);
#endif
          return false;
        }

        init_hardware_acceleration();

        initialized_ = true;
#if __debug_prints
        fan::print("decoder_open_success");
#endif
        return true;
      }

      void close() {
        if (sws_ctx_) {
          sws_freeContext(sws_ctx_);
          sws_ctx_ = nullptr;
        }

        if (frame_) {
          av_frame_free(&frame_);
        }

        if (hw_frame_) {
          av_frame_free(&hw_frame_);
        }

        if (codec_ctx_) {
          if (codec_ctx_->hw_device_ctx) {
            av_buffer_unref(&codec_ctx_->hw_device_ctx);
          }
          avcodec_free_context(&codec_ctx_);
        }

        if (hw_device_ctx_) {
          av_buffer_unref(&hw_device_ctx_);
        }

        initialized_ = false;
        using_hardware_ = false;
        codec_ = nullptr;
        hw_type_ = AV_HWDEVICE_TYPE_NONE;
        hw_pixel_format_ = AV_PIX_FMT_NONE;
#if __debug_prints
        fan::print("decoder_closed");
#endif
      }

      struct decode_result_t {
        std::vector<std::vector<uint8_t>> plane_data;
        std::vector<int> linesize;
        fan::vec2ui image_size{ 0, 0 };
        AVPixelFormat pixel_format = AV_PIX_FMT_NONE;
        bool success = false;
        bool was_hardware_decoded = false;
        int64_t pts = 0;
      };

      std::vector<decode_result_t> decode_packet(const uint8_t* data, size_t size, const std::string& forced_codec) {
        std::vector<decode_result_t> results;

        if (!initialized_) {
          return results;
        }

        if (!codec_ctx_) {
          std::string detected_codec = forced_codec;
          if (detected_codec == "auto-detect") {
            detected_codec = detect_codec_from_data(data, size);
#if __debug_prints
            fan::print_throttled("decoder_detected_codec: " + detected_codec, 2000);
#endif
          }
          else {
#if __debug_prints
            fan::print_throttled("decoder_forced_codec: " + detected_codec, 2000);
#endif
          }

          bool hw_success = false;
          try {
            hw_success = find_and_init_hw_decoder(detected_codec);
          }
          catch (...) {
            hw_success = false;
#if __debug_prints
            fan::print_throttled("decoder_hw_init_exception", 2000);
#endif
          }

          if (!hw_success) {
            if (!find_and_init_sw_decoder(detected_codec)) {
#if __debug_prints
              fan::print_throttled("decoder_init_fail: no_decoder", 1000);
#endif
              return results;
            }
          }
        }

        static int packet_count = 0;
        static int eagain_count = 0;
        static bool has_seen_sps = false;
        static bool has_seen_pps = false;
        static bool fallback_triggered = false;

        packet_count++;

        if (size >= 5 && data[0] == 0x00 && data[1] == 0x00 && data[2] == 0x00 && data[3] == 0x01) {
          uint8_t nal_type = data[4] & 0x1F;

          if (nal_type == 7) {
            has_seen_sps = true;
          }
          else if (nal_type == 8) {
            has_seen_pps = true;
          }
        }

        packet_->data = const_cast<uint8_t*>(data);
        packet_->size = size;

        int ret = avcodec_send_packet(codec_ctx_, packet_);

        if (ret == AVERROR(EAGAIN)) {
          eagain_count++;

          if (using_hardware_ && !fallback_triggered &&
            has_seen_sps && !has_seen_pps && eagain_count > 30) {

            fallback_triggered = true;
#if __debug_prints
            fan::print_throttled("decoder_fallback_sw_due_to_eagain", 2000);
#endif
            close();
            if (open()) {
              std::string detected_codec = detect_codec_from_data(data, size);
              if (find_and_init_sw_decoder(detected_codec)) {
                eagain_count = 0;
                packet_->data = const_cast<uint8_t*>(data);
                packet_->size = size;
                ret = avcodec_send_packet(codec_ctx_, packet_);
              }
            }
          }
        }
        else if (ret < 0 && ret != AVERROR(EAGAIN)) {
#if __debug_prints
          fan::print_throttled("decoder_send_packet_failed", 1000);
#endif
          static int consecutive_errors = 0;
          if (++consecutive_errors > 5 && using_hardware_ && !fallback_triggered) {
            fallback_triggered = true;
#if __debug_prints
            fan::print_throttled("decoder_fallback_sw_due_to_errors", 2000);
#endif
            close();
            if (open()) {
              std::string detected_codec = detect_codec_from_data(data, size);
              find_and_init_sw_decoder(detected_codec);
              consecutive_errors = 0;
            }
          }
          return results;
        }

        while (true) {
          AVFrame* target_frame = using_hardware_ ? hw_frame_ : frame_;

          ret = avcodec_receive_frame(codec_ctx_, target_frame);
          if (ret == AVERROR(EAGAIN)) {
            break;
          }
          else if (ret == AVERROR_EOF) {
            break;
          }
          else if (ret < 0) {
#if __debug_prints
            fan::print_throttled("decoder_receive_frame_failed", 1000);
#endif
            break;
          }

          static bool first_success = true;
          if (first_success) {
            eagain_count = 0;
            first_success = false;
          }

          decode_result_t result;
          result.success = true;
          result.was_hardware_decoded = using_hardware_;
          result.pts = target_frame->pts;

          uint32_t actual_width = target_frame->width;
          uint32_t actual_height = target_frame->height;

          if (using_hardware_ && target_frame->format == hw_pixel_format_) {

            if (frame_->width != actual_width || frame_->height != actual_height) {
              av_frame_unref(frame_);

              frame_->width = actual_width;
              frame_->height = actual_height;

              if (av_frame_get_buffer(frame_, 32) < 0) {
#if __debug_prints
                fan::print_throttled("decoder_sw_frame_buffer_alloc_failed", 2000);
#endif
                continue;
              }
            }

            ret = av_hwframe_transfer_data(frame_, target_frame, 0);
            if (ret < 0) {
#if __debug_prints
              fan::print_throttled("decoder_hw_transfer_failed", 2000);
#endif
              continue;
            }

            frame_->width = actual_width;
            frame_->height = actual_height;

            target_frame = frame_;
          }

          result.image_size = { actual_width, actual_height };
          result.pixel_format = static_cast<AVPixelFormat>(target_frame->format);

          int num_planes = av_pix_fmt_count_planes(static_cast<AVPixelFormat>(target_frame->format));
          if (num_planes <= 0) continue;

          result.plane_data.resize(num_planes);
          result.linesize.resize(num_planes);

          for (int i = 0; i < num_planes; i++) {
            result.linesize[i] = target_frame->linesize[i];

            int plane_height;
            if (i == 0) {
              plane_height = target_frame->height;
            }
            else {
              const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(target_frame->format));
              if (desc) {
                plane_height = target_frame->height >> desc->log2_chroma_h;
              }
              else {
                plane_height = target_frame->height / 2;
              }
            }

            int plane_size = result.linesize[i] * plane_height;
            if (plane_size <= 0 || !target_frame->data[i]) continue;

            result.plane_data[i].resize(plane_size);
            std::memcpy(result.plane_data[i].data(), target_frame->data[i], plane_size);
          }

#if __debug_prints
          fan::print_throttled("decoder_frame_out: " + std::to_string(actual_width) + "x" + std::to_string(actual_height), 1000);
#endif
          results.push_back(std::move(result));
        }

        return results;
      }

      bool has_hardware_support() const {
        return hw_device_ctx_ != nullptr;
      }

      std::string get_hardware_type_string() const {
        if (!using_hardware_) return "Software";

        switch (hw_type_) {
        case AV_HWDEVICE_TYPE_CUDA: return "NVIDIA CUDA";
        case AV_HWDEVICE_TYPE_QSV: return "Intel QuickSync";
        case AV_HWDEVICE_TYPE_VAAPI: return "VAAPI";
        case AV_HWDEVICE_TYPE_DXVA2: return "DirectX VA 2.0";
        case AV_HWDEVICE_TYPE_D3D11VA: return "DirectX VA 11";
        case AV_HWDEVICE_TYPE_VIDEOTOOLBOX: return "Apple VideoToolbox";
        default: return "Unknown Hardware";
        }
      }

      bool force_software_decoding(const std::string& codec_type = "h264") {
        if (codec_ctx_) {
          avcodec_free_context(&codec_ctx_);
          codec_ = nullptr;
        }

        using_hardware_ = false;
#if __debug_prints
        fan::print("decoder_force_software");
#endif
        return find_and_init_sw_decoder(codec_type);
      }

      static std::vector<std::string> get_available_hw_decoders() {
        std::vector<std::string> available;
        auto hw_decoders = get_hw_decoder_names();

        for (const auto& [codec, decoders] : hw_decoders) {
          for (const auto& decoder : decoders) {
            if (avcodec_find_decoder_by_name(decoder.c_str())) {
              available.push_back(decoder);
            }
          }
        }

        return available;
      }
    };


    struct resolution_system_t {
      struct resolution_t {
        uint32_t width, height;
        std::string name;
        std::string category;
        f32_t aspect_ratio() const { return static_cast<f32_t>(width) / height; }
      };

      static const std::vector<resolution_t>& get_all_resolutions() {
        static const std::vector<resolution_t> resolutions = {
          // 16:9
          {3840, 2160, "3840x2160", "4K UHD"},
          {2560, 1440, "2560x1440", "QHD/1440p"},
          {1920, 1080, "1920x1080", "Full HD/1080p"},
          {1600, 900,  "1600x900",  "HD+"},
          {1366, 768,  "1366x768",  "WXGA"},
          {1280, 720,  "1280x720",  "HD/720p"},
          {854,  480,  "854x480",   "FWVGA"},

          // 16:10
          {2560, 1600, "2560x1600", "WQXGA"},
          {1920, 1200, "1920x1200", "WUXGA"},
          {1680, 1050, "1680x1050", "WSXGA+"},
          {1440, 900,  "1440x900",  "WXGA+"},
          {1280, 800,  "1280x800",  "WXGA"},

          // 4:3
          {1600, 1200, "1600x1200", "UXGA"},
          {1400, 1050, "1400x1050", "SXGA+"},
          {1280, 1024, "1280x1024", "SXGA"},
          {1024, 768,  "1024x768",  "XGA"},
          {800,  600,  "800x600",   "SVGA"},
          {640,  480,  "640x480",   "VGA"},

          // Ultrawide
          {3440, 1440, "3440x1440", "UWQHD"},
          {2560, 1080, "2560x1080", "UW-FHD"},
          {1920, 800,  "1920x800",  "UW-WXGA"},

          // Gaming/High Refresh
          {2048, 1152, "2048x1152", "QWXGA"},
          {1856, 1392, "1856x1392", "Custom"},


          // Streaming Optimized
          {1600, 1024, "1600x1024", "Custom 25:16"},
          {1536, 864,  "1536x864",  "Custom 16:9"},
          {1344, 756,  "1344x756",  "Custom 16:9"},
        };
        return resolutions;
      }

      static std::map<std::string, std::vector<resolution_t>> get_resolutions_by_category() {
        std::map<std::string, std::vector<resolution_t>> categorized;

        for (const auto& res : get_all_resolutions()) {
          categorized[res.category].push_back(res);
        }

        return categorized;
      }

      static std::vector<resolution_t> get_resolutions_by_aspect(f32_t target_aspect, f32_t tolerance = 0.01f) {
        std::vector<resolution_t> filtered;

        for (const auto& res : get_all_resolutions()) {
          if (std::abs(res.aspect_ratio() - target_aspect) <= tolerance) {
            filtered.push_back(res);
          }
        }

        std::sort(filtered.begin(), filtered.end(),
          [](const resolution_t& a, const resolution_t& b) {
            return (a.width * a.height) > (b.width * b.height);
          });

        return filtered;
      }

      static std::vector<std::string> get_display_strings(bool include_category = false) {
        std::vector<std::string> display_strings;

        if (include_category) {
          auto categorized = get_resolutions_by_category();
          for (const auto& [category, resolutions] : categorized) {
            for (const auto& res : resolutions) {
              display_strings.push_back(res.name + " (" + category + ")");
            }
          }
        }
        else {
          for (const auto& res : get_all_resolutions()) {
            display_strings.push_back(res.name);
          }
        }

        return display_strings;
      }

      static std::vector<resolution_t> get_popular_resolutions() {
        return {
          {3840, 2160, "3840x2160", "4K UHD"},
          {2560, 1440, "2560x1440", "QHD"},
          {1920, 1080, "1920x1080", "Full HD"},
          {1680, 1050, "1680x1050", "WSXGA+"},
          {1600, 900,  "1600x900",  "HD+"},
          {1440, 900,  "1440x900",  "WXGA+"},
          {1366, 768,  "1366x768",  "WXGA"},
          {1280, 720,  "1280x720",  "HD"},
          {1024, 768,  "1024x768",  "XGA"},
          {800,  600,  "800x600",   "SVGA"}
        };
      }

      static resolution_t get_optimal_resolution(uint32_t screen_width, uint32_t screen_height) {
        f32_t screen_aspect = static_cast<f32_t>(screen_width) / screen_height;

        auto matching_aspect = get_resolutions_by_aspect(screen_aspect, 0.02f);

        if (!matching_aspect.empty()) {
          for (const auto& res : matching_aspect) {
            if (res.width <= screen_width && res.height <= screen_height) {
              return res;
            }
          }
          return matching_aspect.back();
        }

        return { 1920, 1080, "1920x1080", "Full HD" };
      }
    };

    struct screen_encode_t;

    struct resolution_manager_t {
      struct detected_info_t {
        uint32_t screen_width, screen_height;
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

    struct screen_encode_t {
      bool update_scaling_quality(codec_config_t::scaling_quality_e new_quality) {
        config_.scaling_quality = new_quality;

        std::lock_guard<std::mutex> lock(mutex);
        update_flags |= codec_update_e::scaling_quality;

        return true;
      }

      bool open(const codec_config_t& default_config = codec_config_t()) {
        mdscr_initialized_ = false;

        config_.frame_rate = 30;
        config_.codec = codec_config_t::H264;
        config_.rate_control = codec_config_t::VBR;
        config_.bitrate = 10000000;
        config_.width = 1280;
        config_.height = 720;

        if (!encoder_.open(config_, fan::vec2(config_.width, config_.height))) {
          fan::print("Failed to open encoder");
          return false;
        }

        {
          std::lock_guard<std::mutex> lock(mutex);
          update_flags = 0;
        }

        return true;
      }

      bool ensure_mdscr_initialized() {
        if (mdscr_initialized_) {
          return true;
        }

        std::memset(&mdscr, 0, sizeof(mdscr));
        if (int ret = MD_SCR_open(&mdscr); ret != 0) {
          fan::print("failed to open screen:" + std::to_string(ret));
          return false;
        }

        resolution_manager.detect_and_set_optimal(*this);

        if (!encoder_.open(config_, fan::vec2(mdscr.Geometry.Resolution.x, mdscr.Geometry.Resolution.y))) {
          fan::print("Failed to reopen encoder with screen resolution");
          return false;
        }

        mdscr_initialized_ = true;
        return true;
      }
      ~screen_encode_t() {
        encoder_.close();
      }

      bool is_initialized() const {
        return encoder_.initialized_;
      }

      static std::vector<std::string> get_encoders() {
        return libav_encoder_t::get_available_encoders();
      }

      struct encode_data_t {
        std::vector<uint8_t> data;
        bool is_keyframe = false;
        int64_t timestamp = 0;
      };

      bool encode_write() {
        if (!ensure_mdscr_initialized()) {
          return false;
        }

        if (!user_set_resolution &&
          (config_.width != mdscr.Geometry.Resolution.x || config_.height != mdscr.Geometry.Resolution.y)) {

          resolution_manager.detect_and_set_optimal(*this);

          config_.width = mdscr.Geometry.Resolution.x;
          config_.height = mdscr.Geometry.Resolution.y;

          {
            std::lock_guard<std::mutex> lock(mutex);
            update_flags &= ~codec_update_e::codec;
          }

          if (!encoder_.open(config_, fan::vec2(mdscr.Geometry.Resolution.x, mdscr.Geometry.Resolution.y))) {
            return false;
          }
        }

        if (update_flags != 0) {
          std::lock_guard<std::mutex> lock(mutex);

          if (update_flags & codec_update_e::scaling_quality) {
            if (!encoder_.update_scaling_quality(config_.scaling_quality)) {
              fan::print("Failed to update scaling quality");
            }
          }
          if (update_flags & ~codec_update_e::scaling_quality) {
            encoder_.update_config(config_, update_flags & ~codec_update_e::scaling_quality);
          }

          update_flags = 0;
        }

        if (!screen_buffer) {
          fan::print("Error: screen_buffer is null");
          return false;
        }

        if (mdscr.Geometry.Resolution.x <= 0 || mdscr.Geometry.Resolution.y <= 0) {
          fan::print("Error: invalid screen resolution: " +
            std::to_string(mdscr.Geometry.Resolution.x) + "x" +
            std::to_string(mdscr.Geometry.Resolution.y));
          return false;
        }

        if (mdscr.Geometry.LineSize <= 0) {
          fan::print("Error: invalid line size: " + std::to_string(mdscr.Geometry.LineSize));
          return false;
        }

        size_t expected_buffer_size = mdscr.Geometry.LineSize * mdscr.Geometry.Resolution.y;

        int min_line_size = mdscr.Geometry.Resolution.x * 4;
        if (mdscr.Geometry.LineSize < min_line_size) {
          fan::print("Error: LineSize too small for BGRA. Expected at least: " +
            std::to_string(min_line_size) + ", got: " +
            std::to_string(mdscr.Geometry.LineSize));
          return false;
        }

        bool force_keyframe = (encode_write_flags & codec_update_e::force_keyframe) != 0;

        auto results = encoder_.encode_frame(
          screen_buffer,
          mdscr.Geometry.LineSize,
          frame_timestamp_,
          force_keyframe
        );

        if (encode_write_flags & codec_update_e::force_keyframe) {
          encode_write_flags &= ~codec_update_e::force_keyframe;
        }

        std::lock_guard<std::mutex> lock(data_mutex);

        for (const auto& result : results) {
          encode_data_t packet;
          packet.data = std::move(result.data);
          packet.is_keyframe = result.is_keyframe;
          packet.timestamp = result.pts;
          encoded_packets_.push_back(std::move(packet));
        }

        frame_timestamp_++;
        return !results.empty();
      }

      size_t encode_read(uint8_t** data) {
        std::lock_guard<std::mutex> lock(data_mutex);

        if (encoded_packets_.empty()) {
          return 0;
        }

        current_packet_ = std::move(encoded_packets_.front());
        encoded_packets_.erase(encoded_packets_.begin());

        *data = current_packet_.data.data();
        return current_packet_.data.size();
      }

      void sleep_thread() {
        uint64_t target_frame_time_ns = 1000000000ULL / config_.frame_rate;
        uint64_t current_time = fan::time::clock::now();
        uint64_t time_diff = current_time - frame_process_start_time_;

        if (time_diff > target_frame_time_ns) {
          frame_process_start_time_ = current_time;
        }
        else {
          uint64_t sleep_time = target_frame_time_ns - time_diff;
          frame_process_start_time_ = current_time + sleep_time;

          if (config_.frame_rate >= 120) {
            if (sleep_time > 2000000) {
              fan::event::sleep((sleep_time - 1000000) / 1000000);
            }
            while (fan::time::clock::now() < frame_process_start_time_) {
              std::this_thread::yield();
            }
          }
          else if (config_.frame_rate >= 60) {
            if (sleep_time > 1000000) {
              fan::event::sleep((sleep_time - 500000) / 1000000);
            }
            while (fan::time::clock::now() < frame_process_start_time_) {
              std::this_thread::yield();
            }
          }
          else {
            fan::event::sleep(sleep_time / 1000000);
          }
        }
      }

      bool screen_read() {
        if (!ensure_mdscr_initialized()) {
          return false;
        }

        screen_buffer = MD_SCR_read(&mdscr);
        return screen_buffer != nullptr;
      }

      void reset_to_auto_resolution() {
        user_set_resolution = false;
        user_width = 0;
        user_height = 0;
      }

      void set_user_resolution(uint32_t width, uint32_t height) {
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
      uintptr_t new_codec = 0;
      uint8_t update_flags = 0;
      uint8_t encode_write_flags = 0;
      bool user_set_resolution = false;
      uint32_t user_width = 0, user_height = 0;
      bool mdscr_initialized_ = false;

      resolution_manager_t resolution_manager;

      libav_encoder_t encoder_;
      MD_SCR_t mdscr;
      uint8_t* screen_buffer = nullptr;

      std::vector<encode_data_t> encoded_packets_;
      encode_data_t current_packet_;
      std::mutex data_mutex;
      std::mutex mutex;

      int64_t frame_timestamp_ = 0;
      uint64_t encoder_start_time_ = fan::time::clock::now();
      uint64_t frame_process_start_time_ = encoder_start_time_;
    };

    struct screen_decode_t {
      bool open() {
        if (!decoder_.open()) {
          fan::print("Failed to open decoder");
          return false;
        }
        return true;
      }
      void close() {
        decoder_.close();
      }
      ~screen_decode_t() {
        close();
      }

      bool is_initialized() const {
        return decoder_.initialized_;
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
        std::array<std::vector<uint8_t>, 4> data;
        std::array<fan::vec2ui, 4> stride;
        fan::vec2ui image_size = { 0, 0 };
        uint8_t pixel_format = 0;
        uint8_t type = 0;  // 0 = success, >250 = error codes
      };

      decode_data_t decode(void* data, uintptr_t length, loco_t::shape_t& universal_image_renderer) {
        decode_data_t ret;

        if (update_flags & codec_update_e::codec) {
          std::lock_guard<std::mutex> lock(mutex);
          decoder_.close();
          if (!decoder_.open()) {
            ret.type = 254;
            return ret;
          }

          update_flags = 0;

          reload_codec_cb();
        }

        auto results = decoder_.decode_packet(static_cast<const uint8_t*>(data), length, name);

        if (results.empty()) {
          ret.type = 252;
          return ret;
        }

        auto& frame = results[0];
        if (!frame.success) {
          ret.type = 251;
          return ret;
        }

        ret.image_size = frame.image_size;
        ret.type = 1;

        switch (frame.pixel_format) {
        case AV_PIX_FMT_YUV420P:
          ret.pixel_format = fan::graphics::image_format::yuv420p;
          break;
        case AV_PIX_FMT_NV12:
          ret.pixel_format = fan::graphics::image_format::nv12;
          break;
        default:
          ret.type = 249;
          return ret;
        }

        for (size_t i = 0; i < frame.plane_data.size() && i < 4; i++) {
          ret.data[i] = frame.plane_data[i];
          ret.stride[i] = { static_cast<uint32_t>(frame.linesize[i]), 0 };
        }

        frame_size_ = frame.image_size;
        decoded_size = frame.image_size;

        return ret;
      }


      std::string name = "auto-detect";
      uintptr_t new_codec = 0;
      uint8_t update_flags = 0;
      fan::vec2ui frame_size_ = { 1, 1 };
      fan::vec2ui decoded_size;

      libav_decoder_t decoder_;
      std::mutex mutex;
      std::vector<std::function<void()>> graphics_queue;
      std::function<void()> reload_codec_cb = [] {};
    };

    uint8_t convert_pixel_format(::AVPixelFormat fmt) {
      switch (fmt) {
      case AV_PIX_FMT_YUV420P: return fan::graphics::image_format::yuv420p;
      case AV_PIX_FMT_NV12: return fan::graphics::image_format::nv12;
      case AV_PIX_FMT_BGRA: return fan::graphics::image_format::b8g8r8a8_unorm;
      default: return 0;
      }
    }

  } // namespace graphics
} // namespace fan


void fan::graphics::resolution_manager_t::detect_and_set_optimal(screen_encode_t& encoder) {
  if (auto_detected) return;

  detected_info.screen_width = encoder.mdscr.Geometry.Resolution.x;
  detected_info.screen_height = encoder.mdscr.Geometry.Resolution.y;
  detected_info.screen_aspect = static_cast<f32_t>(detected_info.screen_width) / detected_info.screen_height;

  detected_info.optimal_resolution = resolution_system_t::get_optimal_resolution(
    detected_info.screen_width, detected_info.screen_height
  );

  detected_info.matching_aspect_resolutions = resolution_system_t::get_resolutions_by_aspect(
    detected_info.screen_aspect, 0.02f // 2% tolerance
  );

  //encoder.config_.width = detected_info.optimal_resolution.width;
  //encoder.config_.height = detected_info.optimal_resolution.height;

  auto_detected = true;
}