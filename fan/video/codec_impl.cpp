module;

#if defined (FAN_WINDOW) && defined(FAN_VIDEO)

#if defined(FAN_2D)
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/log.h>
#include <libswscale/swscale.h>
}
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d11.h>
#include <wrl/client.h>
extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_d3d11va.h>
}
#endif
#endif

#endif

module fan.graphics.video.codec;

#if defined (FAN_WINDOW) && defined(FAN_VIDEO)

import std;

#if defined(FAN_2D)
import fan.print.error;
import fan.types.vector;
import fan.graphics.common_context;
import fan.graphics;
import fan.time;
import fan.print;
import fan.event;

namespace fan {
  namespace graphics {

#if defined(_WIN32)
    using Microsoft::WRL::ComPtr;

    // turns the bgra desktop texture into the encoder's nv12 surface on the gpu,
    // nothing of the frame reaches system memory
    struct libav_encoder_t::gpu_state_t {
      ~gpu_state_t() {
        av_buffer_unref(&frames);
        av_buffer_unref(&device);
      }

      AVBufferRef* device = nullptr;
      AVBufferRef* frames = nullptr;
      ComPtr<ID3D11VideoDevice> video_device;
      ComPtr<ID3D11VideoContext> video_context;
      ComPtr<ID3D11VideoProcessorEnumerator> enumerator;
      ComPtr<ID3D11VideoProcessor> processor;
      ID3D11Texture2D* input_texture = nullptr;
      ComPtr<ID3D11VideoProcessorInputView> input;
      std::vector<ComPtr<ID3D11VideoProcessorOutputView>> outputs;
    };

    void libav_encoder_t::destroy_gpu_state(encoder_state_t* encoder) {
      delete encoder->gpu;
      encoder->gpu = nullptr;
    }

    bool libav_encoder_t::open_gpu_state(encoder_state_t* encoder, const codec_config_t& config) {
      // the encoder can hold a few frames, a full pool only skips a frame
      constexpr int pool_size = 12;

      auto gpu = std::make_unique<gpu_state_t>();
      gpu->device = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_D3D11VA);
      if (!gpu->device) return false;

      auto* device_ctx = static_cast<AVD3D11VADeviceContext*>(reinterpret_cast<AVHWDeviceContext*>(gpu->device->data)->hwctx);
      gpu_device_->AddRef();
      device_ctx->device = gpu_device_;
      if (av_hwdevice_ctx_init(gpu->device) < 0) return false;

      gpu->frames = av_hwframe_ctx_alloc(gpu->device);
      if (!gpu->frames) return false;

      auto* frames_ctx = reinterpret_cast<AVHWFramesContext*>(gpu->frames->data);
      frames_ctx->format = AV_PIX_FMT_D3D11;
      frames_ctx->sw_format = AV_PIX_FMT_NV12;
      frames_ctx->width = config.width;
      frames_ctx->height = config.height;
      frames_ctx->initial_pool_size = pool_size;
      static_cast<AVD3D11VAFramesContext*>(frames_ctx->hwctx)->BindFlags = D3D11_BIND_RENDER_TARGET;
      if (av_hwframe_ctx_init(gpu->frames) < 0) return false;

      if (FAILED(gpu_device_->QueryInterface(IID_PPV_ARGS(&gpu->video_device)))) return false;
      if (FAILED(device_ctx->device_context->QueryInterface(IID_PPV_ARGS(&gpu->video_context)))) return false;

      D3D11_VIDEO_PROCESSOR_CONTENT_DESC content {};
      content.InputFrameFormat = D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE;
      content.InputWidth = (UINT)encoder->screencap_res.x;
      content.InputHeight = (UINT)encoder->screencap_res.y;
      content.OutputWidth = config.width;
      content.OutputHeight = config.height;
      content.Usage = D3D11_VIDEO_USAGE_OPTIMAL_SPEED;
      if (FAILED(gpu->video_device->CreateVideoProcessorEnumerator(&content, &gpu->enumerator))) return false;
      if (FAILED(gpu->video_device->CreateVideoProcessor(gpu->enumerator.Get(), 0, &gpu->processor))) return false;

      // bt.601 limited range, what nv12.frag decodes and what swscale produces
      D3D11_VIDEO_PROCESSOR_COLOR_SPACE input_space {};
      D3D11_VIDEO_PROCESSOR_COLOR_SPACE output_space {};
      output_space.Nominal_Range = D3D11_VIDEO_PROCESSOR_NOMINAL_RANGE_16_235;
      gpu->video_context->VideoProcessorSetStreamColorSpace(gpu->processor.Get(), 0, &input_space);
      gpu->video_context->VideoProcessorSetOutputColorSpace(gpu->processor.Get(), &output_space);
      gpu->video_context->VideoProcessorSetStreamFrameFormat(gpu->processor.Get(), 0, D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE);
      gpu->video_context->VideoProcessorSetStreamAutoProcessingMode(gpu->processor.Get(), 0, FALSE);

      gpu->outputs.resize(pool_size);
      encoder->gpu = gpu.release();
      return true;
    }

    bool libav_encoder_t::convert_texture(encoder_state_t* encoder, ID3D11Texture2D* texture, AVFrame* frame) {
      auto* gpu = encoder->gpu;

      if (gpu->input_texture != texture) {
        D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC desc {};
        desc.ViewDimension = D3D11_VPIV_DIMENSION_TEXTURE2D;
        gpu->input.Reset();
        if (FAILED(gpu->video_device->CreateVideoProcessorInputView(texture, gpu->enumerator.Get(), &desc, &gpu->input))) {
          gpu->input_texture = nullptr;
          return false;
        }
        gpu->input_texture = texture;
      }

      auto slice = reinterpret_cast<std::intptr_t>(frame->data[1]);
      if (slice < 0 || slice >= (std::intptr_t)gpu->outputs.size()) return false;

      auto& output = gpu->outputs[slice];
      if (!output) {
        D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC desc {};
        desc.ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2DARRAY;
        desc.Texture2DArray.FirstArraySlice = (UINT)slice;
        desc.Texture2DArray.ArraySize = 1;
        auto* surface = reinterpret_cast<ID3D11Texture2D*>(frame->data[0]);
        if (FAILED(gpu->video_device->CreateVideoProcessorOutputView(surface, gpu->enumerator.Get(), &desc, &output))) {
          return false;
        }
      }

      D3D11_VIDEO_PROCESSOR_STREAM stream {};
      stream.Enable = TRUE;
      stream.pInputSurface = gpu->input.Get();
      return SUCCEEDED(gpu->video_context->VideoProcessorBlt(gpu->processor.Get(), output.Get(), 0, 1, &stream));
    }

    void libav_encoder_t::fall_back_to_cpu() {
      fan::print_impl("gpu encoding failed, using the cpu path");
      gpu_device_ = nullptr;
      if (!needs_codec_reinit_.load()) {
        pending_config_ = config_;
        needs_codec_reinit_.store(true);
      }
    }

    std::vector<libav_encoder_t::encode_result_t> libav_encoder_t::encode_texture(ID3D11Texture2D* texture, std::int64_t pts, bool force_keyframe) {
      std::vector<encode_result_t> results;

      apply_pending_reinit();

      encoder_state_t* encoder = active_encoder_.load();
      if (!encoder || !encoder->initialized || !encoder->gpu || !texture) return results;

      AVFrame* frame = av_frame_alloc();
      if (!frame) return results;

      if (av_hwframe_get_buffer(encoder->gpu->frames, frame, 0) >= 0) {
        if (!convert_texture(encoder, texture, frame) || !submit_frame(encoder, frame, pts, force_keyframe, results)) {
          fall_back_to_cpu();
        }
      }
      av_frame_free(&frame);
      return results;
    }
#else
    void libav_encoder_t::destroy_gpu_state(encoder_state_t* encoder) {}
#endif

    bool libav_encoder_t::find_and_init_codec(encoder_state_t* encoder, const codec_config_t& config) {
      auto codec_names = get_codec_names();

      for (const auto& name : codec_names[config.codec]) {
        encoder->codec = avcodec_find_encoder_by_name(name.c_str());

        if (!encoder->codec) continue;

        encoder->codec_name = name;

      #if defined(_WIN32)
        if (gpu_device_ && name.find("nvenc") != std::string::npos && open_codec(encoder, config, name, true)) {
          return true;
        }
      #endif
        if (open_codec(encoder, config, name, false)) {
          return true;
        }

        encoder->codec = nullptr;
      }

      return false;

    }

    bool libav_encoder_t::open_codec(encoder_state_t* encoder, const codec_config_t& config, const std::string& name, bool use_gpu) {
      encoder->codec_ctx = avcodec_alloc_context3(encoder->codec);
      if (!encoder->codec_ctx) return false;

      encoder->codec_ctx->width = config.width;
      encoder->codec_ctx->height = config.height;

      encoder->codec_ctx->time_base = {1, config.frame_rate};
      encoder->codec_ctx->framerate = {config.frame_rate, 1};
      encoder->codec_ctx->gop_size = gop_size_for(config);
      encoder->codec_ctx->max_b_frames = config.frame_rate >= 60 ? 0 : 1;
      encoder->codec_ctx->bit_rate = config.bitrate;
      encoder->codec_ctx->pix_fmt = choose_encoder_pixel_format(name);

    #if defined(_WIN32)
      if (use_gpu) {
        if (!open_gpu_state(encoder, config)) {
          avcodec_free_context(&encoder->codec_ctx);
          destroy_gpu_state(encoder);
          return false;
        }
        encoder->codec_ctx->pix_fmt = AV_PIX_FMT_D3D11;
        encoder->codec_ctx->hw_frames_ctx = av_buffer_ref(encoder->gpu->frames);
      }
    #endif

      if (!use_gpu && encoder->hw_device_ctx && name.find("nvenc") != std::string::npos) {
        encoder->codec_ctx->hw_device_ctx = av_buffer_ref(encoder->hw_device_ctx);

      }

      void* options = encoder->codec_ctx->priv_data;

      if (name.find("nvenc") != std::string::npos) {
        encoder->codec_ctx->rc_buffer_size = config.bitrate / 4;
        encoder->codec_ctx->rc_max_rate = config.bitrate;
        encoder->codec_ctx->rc_min_rate = config.bitrate;

        av_opt_set(options, "preset", "p1", 0);
        av_opt_set(options, "tune", "ll", 0);

        av_opt_set_int(options, "delay", 2, 0);
        av_opt_set_int(options, "async_depth", 8, 0);
        av_opt_set_int(options, "zerolatency", 0, 0);

        av_opt_set_int(options, "surfaces", config.frame_rate >= 144 ? 48 : config.frame_rate >= 120 ? 40 : 32, 0);
        av_opt_set_int(options, "bf", 0, 0);

        av_opt_set(options, "spatial_aq", "0", 0);
        av_opt_set(options, "temporal_aq", "0", 0);
        av_opt_set_int(options, "lookahead_depth", 0, 0);
        av_opt_set_int(options, "rc_lookahead", 0, 0);
        av_opt_set(options, "gpu", 0, 0);

        av_opt_set_int(options, "strict_gop", 1, 0);
        // without this a requested keyframe is an intra frame that a viewer can't start from
        av_opt_set_int(options, "forced-idr", 1, 0);
        av_opt_set_int(options, "no-scenecut", 1, 0);

        switch (config.rate_control) {
        case codec_config_t::CBR: av_opt_set(options, "rc", "cbr", 0); break;
        case codec_config_t::VBR: av_opt_set(options, "rc", "vbr", 0); break;
        case codec_config_t::CRF:
          av_opt_set(options, "rc", "constqp", 0);
          av_opt_set_int(options, "qp", config.crf_value, 0);
          break;
        }
      }
      else if (name.find("amf") != std::string::npos) {
        av_opt_set(options, "quality", "speed", 0);
        av_opt_set(options, "rc", "vbr_latency", 0);
        av_opt_set_int(options, "preanalysis", 0, 0);
      }
      else if (name.find("vaapi") != std::string::npos) {
        av_opt_set(options, "low_power", "1", 0);
      }
      else if (name == "libx264" || name == "libx265") {
        av_opt_set(options, "preset", "ultrafast", 0);
        av_opt_set(options, "tune", "zerolatency", 0);
      }

      if (avcodec_open2(encoder->codec_ctx, encoder->codec, nullptr) == 0) {
        return true;

      }

      avcodec_free_context(&encoder->codec_ctx);
      destroy_gpu_state(encoder);
      return false;

    }

    bool libav_encoder_t::init_encoder_state(encoder_state_t* encoder, const codec_config_t& config, const fan::vec2& screencap_res) {
      cleanup_encoder_state(encoder);

      encoder->config = config;
      encoder->screencap_res = screencap_res.x > 0 ? screencap_res : fan::vec2 {config.width, config.height};

      init_hardware_acceleration(encoder, config);

      if (!find_and_init_codec(encoder, config)) return false;

      encoder->packet = av_packet_alloc();
      if (!encoder->packet) {
        cleanup_encoder_state(encoder);
        return false;

      }

      if (encoder->gpu) {
        encoder->initialized = true;
        return true;
      }

      encoder->frame = av_frame_alloc();
      if (!encoder->frame) {
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

      encoder->sws_ctx = create_sws_context(encoder, config, config.scaling_quality);

      if (!encoder->sws_ctx) {
        cleanup_encoder_state(encoder);
        return false;

      }

      encoder->initialized = true;
      return true;

    }

    SwsContext* libav_encoder_t::create_sws_context(encoder_state_t* encoder, const codec_config_t& config, codec_config_t::scaling_quality_e quality) {
      SwsContext* ctx = sws_alloc_context();
      if (!ctx) return nullptr;

      // a few threads make conversion several times faster, all of them would take cores from a game
      int threads = std::clamp((int)std::thread::hardware_concurrency() / 4, 1, 4);

      av_opt_set_int(ctx, "srcw", encoder->screencap_res.x, 0);
      av_opt_set_int(ctx, "srch", encoder->screencap_res.y, 0);
      av_opt_set_pixel_fmt(ctx, "src_format", AV_PIX_FMT_BGRA, 0);
      av_opt_set_int(ctx, "dstw", config.width, 0);
      av_opt_set_int(ctx, "dsth", config.height, 0);
      av_opt_set_pixel_fmt(ctx, "dst_format", encoder->codec_ctx->pix_fmt, 0);
      av_opt_set_int(ctx, "sws_flags", fan::graphics::get_sws_flags(quality), 0);
      av_opt_set_int(ctx, "threads", threads, 0);

      if (sws_init_context(ctx, nullptr, nullptr) < 0) {
        sws_freeContext(ctx);
        return nullptr;
      }
      return ctx;
    }

    bool libav_encoder_t::convert_frame(encoder_state_t* encoder, std::uint8_t* bgra_data, int stride) {
      int height = encoder->screencap_res.y;

      // only sws_scale_frame runs threaded. it refs the source, a frame without a
      // buffer would be copied first, so the capture gets wrapped without copying
      AVBufferRef* buffer = av_buffer_create(bgra_data, stride * height, [](void*, std::uint8_t*) {}, nullptr, AV_BUFFER_FLAG_READONLY);
      AVFrame* src = buffer ? av_frame_alloc() : nullptr;
      int result = -1;
      if (src) {
        src->format = AV_PIX_FMT_BGRA;
        src->width = encoder->screencap_res.x;
        src->height = height;
        src->data[0] = bgra_data;
        src->linesize[0] = stride;
        src->buf[0] = buffer;
        buffer = nullptr;
        result = sws_scale_frame(encoder->sws_ctx, encoder->frame, src);
        av_frame_free(&src);
      }
      av_buffer_unref(&buffer);
      if (result >= 0) {
        return true;
      }

      const std::uint8_t* src_data[4] = {bgra_data, nullptr, nullptr, nullptr};
      int src_linesize[4] = {stride, 0, 0, 0};
      return sws_scale(encoder->sws_ctx, src_data, src_linesize, 0, height,
        encoder->frame->data, encoder->frame->linesize) == encoder->config.height;
    }

    void libav_encoder_t::apply_pending_reinit() {
      if (!needs_codec_reinit_.load()) return;

      encoder_state_t* new_encoder = (active_encoder_.load() == &encoder_a_) ?

        &encoder_b_ : &encoder_a_;
      if (init_encoder_state(new_encoder, pending_config_, screencap_res_)) {
        encoder_state_t* old = active_encoder_.exchange(new_encoder);

        pending_cleanup_.store(old);
        needs_codec_reinit_.store(false);
        codec_name_ = new_encoder->codec_name;
      }
    }

    std::vector<libav_encoder_t::encode_result_t> libav_encoder_t::encode_frame(std::uint8_t* bgra_data, int stride, std::int64_t pts, bool force_keyframe) {
      std::vector<encode_result_t> results;

      apply_pending_reinit();

      encoder_state_t* encoder = active_encoder_.load();

      if (!encoder || !encoder->initialized || !encoder->frame || !bgra_data || stride <= 0) return results;

      if (convert_frame(encoder, bgra_data, stride)) {
        submit_frame(encoder, encoder->frame, pts, force_keyframe, results);
      }
      return results;

    }

    bool libav_encoder_t::submit_frame(encoder_state_t* encoder, AVFrame* frame, std::int64_t pts, bool force_keyframe, std::vector<encode_result_t>& results) {
      frame->pts = pts;
      frame->pict_type = force_keyframe ? AV_PICTURE_TYPE_I : AV_PICTURE_TYPE_NONE;

      if (avcodec_send_frame(encoder->codec_ctx, frame) < 0) return false;

      while (avcodec_receive_packet(encoder->codec_ctx, encoder->packet) >= 0) {
        encode_result_t result;

        result.data.resize(encoder->packet->size);
        std::memcpy(result.data.data(), encoder->packet->data, encoder->packet->size);
        result.is_keyframe = (encoder->packet->flags & AV_PKT_FLAG_KEY) != 0;
        result.pts = encoder->packet->pts;
        result.dts = encoder->packet->dts;

        results.push_back(std::move(result));
        av_packet_unref(encoder->packet);

      }

      maybe_cleanup_old_encoder();
      return true;

    }

    bool libav_encoder_t::update_config(const codec_config_t& new_config, std::uint8_t update_flags) {
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
        encoder->codec_ctx->time_base = {1, encoder->config.frame_rate};

        encoder->codec_ctx->framerate = {encoder->config.frame_rate, 1};
        encoder->codec_ctx->gop_size = gop_size_for(encoder->config);

        encoder->codec_ctx->max_b_frames = encoder->config.frame_rate >= 60 ? 0 : 1;
      }

      return true;

    }

    bool libav_decoder_t::find_and_init_hw_decoder(const std::string& codec_type) {
      if (!hw_device_ctx_) return false;

      auto hw_decoders = get_hw_decoder_names();
      auto it = hw_decoders.find(codec_type);
      if (it == hw_decoders.end()) return false;

      for (const auto& decoder_name : it->second) {
        bool type_match = false;

        switch (hw_type_) {
        case AV_HWDEVICE_TYPE_CUDA: type_match = decoder_name.find("cuvid") != std::string::npos; break;

        case AV_HWDEVICE_TYPE_QSV: type_match = decoder_name.find("qsv") != std::string::npos; break;
        case AV_HWDEVICE_TYPE_VAAPI: type_match = decoder_name.find("vaapi") != std::string::npos; break;

        case AV_HWDEVICE_TYPE_DXVA2: type_match = decoder_name.find("dxva2") != std::string::npos; break;
        case AV_HWDEVICE_TYPE_D3D11VA: type_match = decoder_name.find("d3d11va") != std::string::npos; break;

        case AV_HWDEVICE_TYPE_VIDEOTOOLBOX: type_match = decoder_name.find("videotoolbox") != std::string::npos; break;
        default: continue;

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
          //  av_opt_set_int(codec_ctx_->priv_data, "surfaces", 16, 0);

          av_opt_set_int(codec_ctx_->priv_data, "async_depth", 4, 0);
          av_opt_set_int(codec_ctx_->priv_data, "drop_second_field", 1, 0);
          av_opt_set(codec_ctx_->priv_data, "deint", "weave", 0);
          av_opt_set_int(codec_ctx_->priv_data, "crop", 0, 0);
          av_opt_set_int(codec_ctx_->priv_data, "resize", 0, 0);

          codec_ctx_->has_b_frames = 0;
          codec_ctx_->delay = 1;
          // otherwise cuvid holds back 4 decoded frames
          codec_ctx_->flags |= AV_CODEC_FLAG_LOW_DELAY;
        }
        else if (decoder_name.find("qsv") != std::string::npos) {
          av_opt_set_int(codec_ctx_->priv_data, "async_depth", 1, 0);

          codec_ctx_->delay = 0;
        }
        else if (decoder_name.find("vaapi") != std::string::npos) {
          av_opt_set_int(codec_ctx_->priv_data, "low_power", 1, 0);

          codec_ctx_->delay = 0;
        }

        std::atomic<bool> open_completed {false};

        std::atomic<int> open_result {-1};

        std::thread open_thread([&]() {
          int result = avcodec_open2(codec_ctx_, codec_, nullptr);
          open_result.store(result);
          open_completed.store(true);
        });

        auto start_time = std::chrono::steady_clock::now();
        bool timed_out = false;

        while (!open_completed.load()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));

          if (std::chrono::steady_clock::now() - start_time > std::chrono::seconds(2)) {
            timed_out = true;

            break;
          }
        }

        if (timed_out) {
          if (open_thread.joinable()) open_thread.detach();

          if (codec_ctx_->hw_device_ctx) av_buffer_unref(&codec_ctx_->hw_device_ctx);
          avcodec_free_context(&codec_ctx_);
          codec_ = nullptr;
          continue;
        }

        if (open_thread.joinable()) open_thread.join();

        if (open_result.load() == 0) {
          codec_name_ = decoder_name;

          using_hardware_ = true;
          return true;
        }

        if (codec_ctx_->hw_device_ctx) av_buffer_unref(&codec_ctx_->hw_device_ctx);
        avcodec_free_context(&codec_ctx_);

        codec_ = nullptr;
      }

      return false;

    }

    std::vector<libav_decoder_t::decode_result_t> libav_decoder_t::decode_packet(
      const std::uint8_t* data,
      std::size_t size,
      const std::string& forced_codec,
      bool to_rgba,
      fan::vec2ui target_size,
      codec_config_t::scaling_quality_e quality) {
      std::vector<decode_result_t> results;
      if (!initialized_) return results;

      if (!codec_ctx_) {
        std::string detected_codec = forced_codec;
        if (detected_codec == "auto-detect") {
          detected_codec = detect_codec_from_data(data, size);
        }

        bool hw_success = false;
        try {
          hw_success = find_and_init_hw_decoder(detected_codec);
        }
        catch (...) {
          hw_success = false;
        }

        if (!hw_success && !find_and_init_sw_decoder(detected_codec)) {
          return results;
        }
      }

      packet_->data = const_cast<std::uint8_t*>(data);
      packet_->size = size;

      int ret = avcodec_send_packet(codec_ctx_, packet_);
      if (ret < 0 && ret != AVERROR(EAGAIN)) {
        ++decode_failures_;
        return results;
      }

      while (true) {
        AVFrame* target_frame = using_hardware_ ? hw_frame_ : frame_;
        ret = avcodec_receive_frame(codec_ctx_, target_frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
        if (ret < 0) {
          ++decode_failures_;
          break;
        }
        decode_failures_ = 0;

        decode_result_t result;
        result.success = true;
        result.was_hardware_decoded = using_hardware_;
        result.pts = target_frame->pts;

        std::uint32_t actual_width = target_frame->width;
        std::uint32_t actual_height = target_frame->height;

        if (using_hardware_ && target_frame->format == hw_pixel_format_) {
          av_frame_unref(frame_);
          frame_->format = AV_PIX_FMT_NONE;

          ret = av_hwframe_transfer_data(frame_, target_frame, 0);
          if (ret < 0) {
            av_frame_unref(target_frame);
            continue;
          }

          actual_width = frame_->width;
          actual_height = frame_->height;
          target_frame = frame_;
        }

        result.image_size = {actual_width, actual_height};
        int dst_w = (target_size.x > 0) ? target_size.x : actual_width;
        int dst_h = (target_size.y > 0) ? target_size.y : actual_height;
        int sws_flags = get_sws_flags(quality);

        if (to_rgba) {
          sws_ctx_ = sws_getCachedContext(sws_ctx_,
            actual_width, actual_height, static_cast<AVPixelFormat>(target_frame->format),
            dst_w, dst_h, AV_PIX_FMT_RGBA,
            sws_flags, nullptr, nullptr, nullptr);
          if (sws_ctx_) {
            result.image_size = fan::vec2ui(dst_w, dst_h);
            result.linesize.resize(1);
            result.plane_data.resize(1);
            result.linesize[0] = dst_w * 4;
            result.plane_data[0].resize(dst_h * result.linesize[0]);

            std::uint8_t* dst[4] = {result.plane_data[0].data(), nullptr, nullptr, nullptr};
            int dst_stride[4] = {result.linesize[0], 0, 0, 0};
            sws_scale(sws_ctx_, target_frame->data, target_frame->linesize, 0, actual_height, dst, dst_stride);
            result.pixel_format = AV_PIX_FMT_RGBA;
          }
        }
        else {
          result.pixel_format = static_cast<AVPixelFormat>(target_frame->format);
          int num_planes = av_pix_fmt_count_planes(static_cast<AVPixelFormat>(target_frame->format));
          if (num_planes > 0) {
            result.plane_data = std::move(spare_planes_);
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
                if (desc) plane_height = target_frame->height >> desc->log2_chroma_h;
                else plane_height = target_frame->height / 2;
              }

              int plane_size = result.linesize[i] * plane_height;
              if (plane_size > 0 && target_frame->data[i]) {
                result.plane_data[i].resize(plane_size);
                std::memcpy(result.plane_data[i].data(), target_frame->data[i], plane_size);
              }
            }
          }
        }

        av_frame_unref(frame_);
        av_frame_unref(hw_frame_);

        results.push_back(std::move(result));
      }

      return results;
    }

  }

  void mp4_demuxer_t::extract_extradata_() {
    sps_pps_.clear();

    if (!fmt || video_stream < 0) {
      return;
    }

    AVCodecParameters* par = fmt->streams[video_stream]->codecpar;

    if (!par || !par->extradata || par->extradata_size < 7) {
      return;
    }

    const uint8_t* data = par->extradata;
    size_t size = par->extradata_size;

    if (data[0] != 1) {
      return;
    }

    size_t pos = 5;

    uint8_t num_sps = data[pos++] & 0x1F;

    for (uint8_t i = 0; i < num_sps; ++i) {
      if (pos + 2 > size) return;

      uint16_t sps_size =
        (uint16_t(data[pos]) << 8) |
        uint16_t(data[pos + 1]);

      pos += 2;

      if (pos + sps_size > size) return;

      sps_pps_.insert(sps_pps_.end(), {0x00,0x00,0x00,0x01});
      sps_pps_.insert(
        sps_pps_.end(),
        data + pos,
        data + pos + sps_size
      );

      pos += sps_size;
    }

    if (pos + 1 > size) return;

    uint8_t num_pps = data[pos++];

    for (uint8_t i = 0; i < num_pps; ++i) {
      if (pos + 2 > size) return;

      uint16_t pps_size =
        (uint16_t(data[pos]) << 8) |
        uint16_t(data[pos + 1]);

      pos += 2;

      if (pos + pps_size > size) return;

      sps_pps_.insert(sps_pps_.end(), {0x00,0x00,0x00,0x01});
      sps_pps_.insert(
        sps_pps_.end(),
        data + pos,
        data + pos + pps_size
      );

      pos += pps_size;
    }
  }

}
#endif
#endif
