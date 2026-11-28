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
#include <libavutil/hwcontext.h>
}
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d11.h>
#include <d3d11_4.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#pragma comment(lib, "d3dcompiler.lib")
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

    static constexpr const char* fullscreen_vs = R"(
      struct output_t {
        float4 position : SV_Position;
        float2 uv : TEXCOORD0;
      };
      output_t main(uint id : SV_VertexID) {
        output_t output;
        output.uv = float2((id << 1) & 2, id & 2);
        output.position = float4(output.uv * float2(2, -2) + float2(-1, 1), 0, 1);
        return output;
      }
    )";

    // bt.601 limited range, the inverse of nv12.frag. the d3d11 video processor is
    // avoided on purpose, driver video settings change its output
    static constexpr const char* luma_ps = R"(
      Texture2D image : register(t0);
      SamplerState image_sampler : register(s0);
      float main(float4 position : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
        float3 rgb = image.Sample(image_sampler, uv).rgb;
        return dot(rgb, float3(0.2568, 0.5041, 0.0979)) + 0.0625;
      }
    )";

    static constexpr const char* chroma_ps = R"(
      Texture2D image : register(t0);
      SamplerState image_sampler : register(s0);
      float2 main(float4 position : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
        float3 rgb = image.Sample(image_sampler, uv).rgb;
        return float2(
          dot(rgb, float3(-0.1482, -0.2910, 0.4392)),
          dot(rgb, float3(0.4392, -0.3678, -0.0714))
        ) + 0.5;
      }
    )";

    // 4:4:4: the picture copied as it is (scaled when the sizes differ), NVENC turns rgb into 4:4:4 itself with the
    // same bt.601 limited range
    static constexpr const char* copy_ps = R"(
      Texture2D image : register(t0);
      SamplerState image_sampler : register(s0);
      float4 main(float4 position : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
        return float4(image.Sample(image_sampler, uv).rgb, 1);
      }
    )";

    static ComPtr<ID3DBlob> compile_shader(std::string_view source, const char* target) {
      ComPtr<ID3DBlob> code;
      ComPtr<ID3DBlob> errors;
      if (FAILED(D3DCompile(source.data(), source.size(), nullptr, nullptr, nullptr, "main", target,
        D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors))) {
        if (errors) {
          fan::print_impl(static_cast<const char*>(errors->GetBufferPointer()));
        }
        return nullptr;
      }
      return code;
    }

    // turns the bgra desktop texture into the encoder's nv12 surface on the gpu,
    // nothing of the frame reaches system memory
    struct libav_encoder_t::gpu_state_t {
      ~gpu_state_t() {
        if (fence_event) {
          CloseHandle(fence_event);
        }
        av_buffer_unref(&frames);
        av_buffer_unref(&device);
      }
      // the wait for the conversion to finish: a fence the GPU sets, the thread asleep on its event. Without one
      // (older Windows) the query below is polled
      ComPtr<ID3D11Fence> fence;
      ComPtr<ID3D11DeviceContext4> context4;
      HANDLE fence_event = nullptr;
      std::uint64_t fence_value = 0;

      struct surface_views_t {
        ComPtr<ID3D11RenderTargetView> luma;
        ComPtr<ID3D11RenderTargetView> chroma;
        ComPtr<ID3D11RenderTargetView> rgb;
      };
      // the surfaces are BGRA, for 4:4:4
      bool rgb = false;
      ComPtr<ID3D11PixelShader> copy_shader;

      AVBufferRef* device = nullptr;
      AVBufferRef* frames = nullptr;
      ComPtr<ID3D11Device> d3d_device;
      ComPtr<ID3D11DeviceContext> context;
      ComPtr<ID3D11VertexShader> vertex_shader;
      ComPtr<ID3D11PixelShader> luma_shader;
      ComPtr<ID3D11PixelShader> chroma_shader;
      ComPtr<ID3D11SamplerState> sampler;
      ID3D11Texture2D* input_texture = nullptr;
      ComPtr<ID3D11ShaderResourceView> input;
      // by surface and slice, the pool can hand out separate textures that are all slice 0
      std::map<std::pair<ID3D11Texture2D*, std::intptr_t>, surface_views_t> outputs;
      ComPtr<ID3D11Query> blit_done;
    };

    void libav_encoder_t::destroy_gpu_state(encoder_state_t* encoder) {
      delete encoder->gpu;
      encoder->gpu = nullptr;
    }

    bool libav_encoder_t::open_gpu_state(encoder_state_t* encoder, const codec_config_t& config) {
      // the encoder can hold a few frames, a full pool only skips a frame
      constexpr int pool_size = 12;

      auto fail = [](const char* step, int result = 0) {
        fan::print_impl("gpu input: " + std::string(step) + (result ? " (" + std::to_string(result) + ")" : ""));
        return false;
      };
      auto gpu = std::make_unique<gpu_state_t>();
      gpu->device = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_D3D11VA);
      if (!gpu->device) return fail("no d3d11 device context");

      auto* device_ctx = static_cast<AVD3D11VADeviceContext*>(reinterpret_cast<AVHWDeviceContext*>(gpu->device->data)->hwctx);
      gpu_device_->AddRef();
      device_ctx->device = gpu_device_;
      int result = av_hwdevice_ctx_init(gpu->device);
      if (result < 0) return fail("d3d11 device init failed", result);

      gpu->frames = nullptr;
      gpu->rgb = config.chroma_444 && config.codec != codec_config_t::AV1;
      UINT nv12_support = 0, bgra_support = 0;
      gpu_device_->CheckFormatSupport(DXGI_FORMAT_NV12, &nv12_support);
      gpu_device_->CheckFormatSupport(DXGI_FORMAT_B8G8R8A8_UNORM, &bgra_support);
      fan::print_impl("gpu input: format support NV12 " + std::to_string(nv12_support) +
        ", BGRA " + std::to_string(bgra_support));
      // FFmpeg's D3D11 frame pool makes one array texture for all the surfaces. NV12 as a render target is what
      // the video processor writes into, but some drivers only create it with other bind flags: try the combinations
      auto try_frames = [&](UINT bind_flags, int pool_size, int& result) {
        av_buffer_unref(&gpu->frames);
        gpu->frames = av_hwframe_ctx_alloc(gpu->device);
        if (!gpu->frames) {
          result = -1;
          return false;
        }
        auto* frames_ctx = reinterpret_cast<AVHWFramesContext*>(gpu->frames->data);
        frames_ctx->format = AV_PIX_FMT_D3D11;
        frames_ctx->sw_format = gpu->rgb ? AV_PIX_FMT_BGRA : AV_PIX_FMT_NV12;
        frames_ctx->width = config.width;
        frames_ctx->height = config.height;
        frames_ctx->initial_pool_size = pool_size;
        static_cast<AVD3D11VAFramesContext*>(frames_ctx->hwctx)->BindFlags = bind_flags;
        result = av_hwframe_ctx_init(gpu->frames);
        return result >= 0;
      };
      const UINT bind_options[] = {
        D3D11_BIND_RENDER_TARGET | D3D11_BIND_DECODER,
        D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,
        D3D11_BIND_RENDER_TARGET,
        0
      };
      bool frames_ok = false;
      for (int pool : {pool_size, 8, 4}) {
        for (UINT bind_flags : bind_options) {
          if (try_frames(bind_flags, pool, result)) {
            frames_ok = true;
            break;
          }
        }
        if (frames_ok) {
          break;
        }
      }
      // Some NVIDIA drivers refuse an NV12 texture array as a render target (E_INVALIDARG) while one texture of it is
      // fine. A pool size of 0 makes FFmpeg skip the array and make a separate texture for each surface it hands out,
      // the encoder registers each the same way
      if (!frames_ok) {
        for (UINT bind_flags : {(UINT)D3D11_BIND_RENDER_TARGET, (UINT)(D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE)}) {
          if (try_frames(bind_flags, 0, result)) {
            // the first surface is made on demand: one now shows whether the driver takes it
            AVFrame* probe = av_frame_alloc();
            bool made = probe && av_hwframe_get_buffer(gpu->frames, probe, 0) >= 0;
            av_frame_free(&probe);
            if (made) {
              frames_ok = true;
              fan::print_impl("gpu input: separate textures per frame (the driver refuses texture arrays)");
              break;
            }
          }
        }
      }
      if (!frames_ok) {
        // FFmpeg only says "unknown": the driver's own answer for the same texture, and the size, say why
        D3D11_TEXTURE2D_DESC desc {};
        desc.Width = (UINT)config.width;
        desc.Height = (UINT)config.height;
        desc.MipLevels = 1;
        desc.ArraySize = 4;
        desc.Format = gpu->rgb ? DXGI_FORMAT_B8G8R8A8_UNORM : DXGI_FORMAT_NV12;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_RENDER_TARGET;
        ComPtr<ID3D11Texture2D> test;
        HRESULT array_hr = gpu_device_->CreateTexture2D(&desc, nullptr, &test);
        test.Reset();
        desc.ArraySize = 1;
        HRESULT single_hr = gpu_device_->CreateTexture2D(&desc, nullptr, &test);
        char text[160];
        std::snprintf(text, sizeof(text), "gpu input: no frame pool for %dx%d (array 0x%08lx, single 0x%08lx), using cpu frames",
          (int)config.width, (int)config.height, (unsigned long)array_hr, (unsigned long)single_hr);
        fan::print_impl(text);
        return false;
      }

      gpu->d3d_device = gpu_device_;
      gpu->context = device_ctx->device_context;
      D3D11_QUERY_DESC query_desc {};
      query_desc.Query = D3D11_QUERY_EVENT;
      if (FAILED(gpu_device_->CreateQuery(&query_desc, &gpu->blit_done))) return fail("d3d11 query failed");
      {
        ComPtr<ID3D11Device5> device5;
        if (SUCCEEDED(gpu_device_->QueryInterface(IID_PPV_ARGS(&device5))) &&
          SUCCEEDED(device5->CreateFence(0, D3D11_FENCE_FLAG_NONE, IID_PPV_ARGS(&gpu->fence))) &&
          SUCCEEDED(gpu->context->QueryInterface(IID_PPV_ARGS(&gpu->context4)))) {
          gpu->fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        }
        if (!gpu->fence_event) {
          gpu->fence.Reset();
          gpu->context4.Reset();
        }
      }

      auto vertex_code = compile_shader(fullscreen_vs, "vs_5_0");
      auto luma_code = compile_shader(luma_ps, "ps_5_0");
      auto chroma_code = compile_shader(chroma_ps, "ps_5_0");
      if (!vertex_code || !luma_code || !chroma_code) return fail("shader compile failed");

      if (FAILED(gpu_device_->CreateVertexShader(vertex_code->GetBufferPointer(), vertex_code->GetBufferSize(), nullptr, &gpu->vertex_shader))) return fail("vertex shader create failed");
      if (FAILED(gpu_device_->CreatePixelShader(luma_code->GetBufferPointer(), luma_code->GetBufferSize(), nullptr, &gpu->luma_shader))) return fail("luma shader create failed");
      if (FAILED(gpu_device_->CreatePixelShader(chroma_code->GetBufferPointer(), chroma_code->GetBufferSize(), nullptr, &gpu->chroma_shader))) return fail("chroma shader create failed");
      if (gpu->rgb) {
        auto copy_code = compile_shader(copy_ps, "ps_5_0");
        if (!copy_code || FAILED(gpu_device_->CreatePixelShader(copy_code->GetBufferPointer(), copy_code->GetBufferSize(), nullptr, &gpu->copy_shader))) return fail("copy shader create failed");
      }

      D3D11_SAMPLER_DESC sampler_desc {};
      sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
      sampler_desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
      sampler_desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
      sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
      sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;
      if (FAILED(gpu_device_->CreateSamplerState(&sampler_desc, &gpu->sampler))) return fail("sampler create failed");

      encoder->gpu = gpu.release();
      return true;
    }

    bool libav_encoder_t::convert_texture(encoder_state_t* encoder, ID3D11Texture2D* texture, AVFrame* frame) {
      auto* gpu = encoder->gpu;

      if (gpu->input_texture != texture) {
        gpu->input.Reset();
        if (FAILED(gpu->d3d_device->CreateShaderResourceView(texture, nullptr, &gpu->input))) {
          gpu->input_texture = nullptr;
          return false;
        }
        gpu->input_texture = texture;
      }

      auto* surface = reinterpret_cast<ID3D11Texture2D*>(frame->data[0]);
      auto slice = reinterpret_cast<std::intptr_t>(frame->data[1]);

      auto& views = gpu->outputs[{surface, slice}];
      if (gpu->rgb && !views.rgb) {
        D3D11_TEXTURE2D_DESC surface_desc;
        surface->GetDesc(&surface_desc);
        D3D11_RENDER_TARGET_VIEW_DESC desc {};
        if (surface_desc.ArraySize > 1) {
          desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
          desc.Texture2DArray.FirstArraySlice = (UINT)slice;
          desc.Texture2DArray.ArraySize = 1;
        }
        else {
          desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
        }
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        if (FAILED(gpu->d3d_device->CreateRenderTargetView(surface, &desc, &views.rgb))) {
          return false;
        }
      }
      if (!gpu->rgb && !views.luma) {
        D3D11_TEXTURE2D_DESC surface_desc;
        surface->GetDesc(&surface_desc);

        D3D11_RENDER_TARGET_VIEW_DESC desc {};
        if (surface_desc.ArraySize > 1) {
          desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
          desc.Texture2DArray.FirstArraySlice = (UINT)slice;
          desc.Texture2DArray.ArraySize = 1;
        }
        else {
          desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
        }
        desc.Format = DXGI_FORMAT_R8_UNORM;
        if (FAILED(gpu->d3d_device->CreateRenderTargetView(surface, &desc, &views.luma))) {
          return false;
        }
        desc.Format = DXGI_FORMAT_R8G8_UNORM;
        if (FAILED(gpu->d3d_device->CreateRenderTargetView(surface, &desc, &views.chroma))) {
          views.luma.Reset();
          return false;
        }
      }

      auto* context = gpu->context.Get();
      context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      context->IASetInputLayout(nullptr);
      context->VSSetShader(gpu->vertex_shader.Get(), nullptr, 0);
      context->PSSetShaderResources(0, 1, gpu->input.GetAddressOf());
      context->PSSetSamplers(0, 1, gpu->sampler.GetAddressOf());

      auto draw_plane = [&](ID3D11RenderTargetView* target, ID3D11PixelShader* shader, int width, int height) {
        D3D11_VIEWPORT viewport {0.f, 0.f, (float)width, (float)height, 0.f, 1.f};
        context->OMSetRenderTargets(1, &target, nullptr);
        context->RSSetViewports(1, &viewport);
        context->PSSetShader(shader, nullptr, 0);
        context->Draw(3, 0);
      };
      if (gpu->rgb) {
        draw_plane(views.rgb.Get(), gpu->copy_shader.Get(), frame->width, frame->height);
      }
      else {
        draw_plane(views.luma.Get(), gpu->luma_shader.Get(), frame->width, frame->height);
        draw_plane(views.chroma.Get(), gpu->chroma_shader.Get(), (frame->width + 1) / 2, (frame->height + 1) / 2);
      }

      // the capture copies into the input and nvenc reads the surface next
      ID3D11RenderTargetView* no_target = nullptr;
      ID3D11ShaderResourceView* no_input = nullptr;
      context->OMSetRenderTargets(1, &no_target, nullptr);
      context->PSSetShaderResources(0, 1, &no_input);

      // the draws are only queued and nvenc doesn't wait for d3d11. With a game on the GPU they finish late (several
      // ms): polling for that in a loop kept a CPU core busy the whole time, now the thread sleeps until the GPU says
      if (gpu->fence) {
        ++gpu->fence_value;
        if (SUCCEEDED(gpu->context4->Signal(gpu->fence.Get(), gpu->fence_value)) &&
          SUCCEEDED(gpu->fence->SetEventOnCompletion(gpu->fence_value, gpu->fence_event))) {
          gpu->context->Flush();
          WaitForSingleObject(gpu->fence_event, 100);
          return true;
        }
      }
      gpu->context->End(gpu->blit_done.Get());
      gpu->context->Flush();
      auto give_up = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
      while (gpu->context->GetData(gpu->blit_done.Get(), nullptr, 0, 0) == S_FALSE &&
        std::chrono::steady_clock::now() < give_up) {
        std::this_thread::yield();
      }
      return true;
    }

    void libav_encoder_t::forget_input_texture() {
      encoder_state_t* encoder = active_encoder_.load();
      if (encoder && encoder->gpu) {
        encoder->gpu->input.Reset();
        encoder->gpu->input_texture = nullptr;
      }
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

    // True when an AV1 packet holds a key frame: an OBU_FRAME or OBU_FRAME_HEADER that is shown and whose frame_type
    // is KEY_FRAME (0). OBUs: a header byte (type in bits 6-3, an extension byte after it when bit 2, a size when bit
    // 1), the size as leb128
    static bool av1_has_key_frame(const std::uint8_t* data, std::size_t size) {
      std::size_t at = 0;
      while (at < size) {
        std::uint8_t header = data[at++];
        int type = (header >> 3) & 0x0F;
        bool extension = (header & 0x04) != 0;
        bool has_size = (header & 0x02) != 0;
        if (extension) {
          ++at;
        }
        std::size_t length = size - std::min(at, size);
        if (has_size) {
          length = 0;
          for (int i = 0; i < 8 && at < size; ++i) {
            std::uint8_t byte = data[at++];
            length |= (std::size_t)(byte & 0x7F) << (i * 7);
            if (!(byte & 0x80)) {
              break;
            }
          }
        }
        if (at >= size) {
          break;
        }
        // 3 frame header, 6 frame: show_existing_frame is the first bit, frame_type the next two
        if (type == 3 || type == 6) {
          std::uint8_t first = data[at];
          if (!(first & 0x80) && ((first >> 5) & 0x03) == 0) {
            return true;
          }
        }
        at += length;
      }
      return false;
    }

    bool libav_encoder_t::find_and_init_codec(encoder_state_t* encoder, const codec_config_t& config) {
      auto codec_names = get_codec_names();

      // nvenc without an nvidia gpu or amf without its runtime fails the same way every time
      static std::mutex unavailable_mutex;
      static std::unordered_set<std::string> unavailable;
      auto is_vendor_encoder = [](const std::string& name) {
        return name.find("nvenc") != std::string::npos || name.find("amf") != std::string::npos;
      };

      for (const auto& name : codec_names[config.codec]) {
        if (config.software_only && (name.find("nvenc") != std::string::npos || name.find("amf") != std::string::npos ||
          name.find("vaapi") != std::string::npos || name.find("qsv") != std::string::npos)) {
          continue;
        }
        {
          std::lock_guard lock(unavailable_mutex);
          if (unavailable.contains(name)) continue;
        }
        encoder->codec = avcodec_find_encoder_by_name(name.c_str());

        if (!encoder->codec) continue;

        encoder->codec_name = name;

      #if defined(_WIN32)
        if (gpu_device_ && name.find("nvenc") != std::string::npos) {
          if (open_codec(encoder, config, name, true)) {
            return true;
          }
          // the capture's texture stays on the gpu, which is several ms cheaper per frame than a readback. A driver
          // that refuses one of the tuning options still takes the frames with the basic set
          fan::print_impl("nvenc gpu input failed for " + name + ", trying the basic options");
          if (open_codec(encoder, config, name, true, false, false, true)) {
            return true;
          }
          fan::print_impl("nvenc gpu input failed for " + name + ", using cpu input");
        }
      #endif
        if (name.find("vaapi") != std::string::npos) {
          // vaapi encodes gpu surfaces, it needs the vaapi device that init_hardware_acceleration opened
          if (encoder->hw_type != AV_HWDEVICE_TYPE_VAAPI) continue;
          // amd only has the normal entrypoint, some intel chips only the low power one
          if (open_codec(encoder, config, name, false, false) || open_codec(encoder, config, name, false, true)) {
            return true;
          }
          encoder->codec = nullptr;
          continue;
        }
        if (open_codec(encoder, config, name, false)) {
          return true;
        }
        if (is_vendor_encoder(name)) {
          std::lock_guard lock(unavailable_mutex);
          unavailable.insert(name);
        }

        encoder->codec = nullptr;
      }

      // No encoder of this codec opened: AV1 or HEVC on a card without it, and this FFmpeg has no CPU encoder of
      // either. H.264, which every card and x264 have, instead of no stream at all
      if (config.codec != codec_config_t::H264) {
        fan::print_impl("no encoder for the chosen codec opened, using H.264");
        codec_config_t fallback = config;
        fallback.codec = codec_config_t::H264;
        return find_and_init_codec(encoder, fallback);
      }
      return false;

    }

    bool libav_encoder_t::open_vaapi_frames(encoder_state_t* encoder, const codec_config_t& config) {
      AVBufferRef* frames = av_hwframe_ctx_alloc(encoder->hw_device_ctx);
      if (!frames) return false;
      auto* frames_ctx = (AVHWFramesContext*)frames->data;
      frames_ctx->format = AV_PIX_FMT_VAAPI;
      frames_ctx->sw_format = AV_PIX_FMT_NV12;
      frames_ctx->width = config.width;
      frames_ctx->height = config.height;
      frames_ctx->initial_pool_size = 16;
      if (av_hwframe_ctx_init(frames) < 0) {
        av_buffer_unref(&frames);
        return false;
      }
      encoder->hw_frames_ctx = frames;
      encoder->codec_ctx->pix_fmt = AV_PIX_FMT_VAAPI;
      encoder->codec_ctx->hw_frames_ctx = av_buffer_ref(frames);
      return encoder->codec_ctx->hw_frames_ctx != nullptr;
    }

    bool libav_encoder_t::open_codec(encoder_state_t* encoder, const codec_config_t& config, const std::string& name, bool use_gpu, bool low_power, bool intra_refresh, bool basic) {
      encoder->codec_ctx = avcodec_alloc_context3(encoder->codec);
      if (!encoder->codec_ctx) return false;
      encoder->intra_refresh = false;

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

      if (name.find("vaapi") != std::string::npos && !open_vaapi_frames(encoder, config)) {
        avcodec_free_context(&encoder->codec_ctx);
        if (encoder->hw_frames_ctx) av_buffer_unref(&encoder->hw_frames_ctx);
        return false;
      }

      void* options = encoder->codec_ctx->priv_data;

      if (name.find("nvenc") != std::string::npos) {
        encoder->codec_ctx->rc_buffer_size = config.bitrate / 4;
        encoder->codec_ctx->rc_max_rate = config.bitrate;
        encoder->codec_ctx->rc_min_rate = config.bitrate;

        // p1 is the fastest and the blockiest. The encoder is its own chip with room to spare at these rates, so a
        // slower preset costs no frames: p4 up to 60 fps, p3 up to 120, p2 above (4k at 144 still keeps up).
        // basic is the retry after the full option set was refused: p2 and nothing that a driver might not know
        av_opt_set(options, "preset", basic ? "p2" : config.frame_rate <= 60 ? "p4" : config.frame_rate <= 120 ? "p3" : "p2", 0);
        av_opt_set(options, "tune", "ll", 0);
        // 4:4:4 from the BGRA surfaces (open_gpu_state): NVENC converts them itself, with the profile that carries it
        if (config.chroma_444 && config.codec != codec_config_t::AV1 && use_gpu) {
          av_opt_set(options, "rgb_mode", "yuv444", 0);
          av_opt_set(options, "profile", config.codec == codec_config_t::H265 ? "rext" : "high444p", 0);
        }

        // frames come out as they are encoded. A delay of n holds n frames back, at 60 fps 2 of them were 33 ms
        av_opt_set_int(options, "delay", 0, 0);
        // several slices let a software decoder use its threads inside one frame, see find_and_init_sw_decoder
        if (!basic && config.codec != codec_config_t::AV1) {
          encoder->codec_ctx->slices = 4;
        }
        av_opt_set_int(options, "async_depth", basic ? 4 : 8, 0);
        av_opt_set_int(options, "zerolatency", 0, 0);

        av_opt_set_int(options, "surfaces", basic ? 16 : config.frame_rate >= 144 ? 48 : config.frame_rate >= 120 ? 40 : 32, 0);
        av_opt_set_int(options, "bf", 0, 0);

        if (!basic) {
          // spatial AQ gives dark and flat parts (shade, sky, fog) more bits; without it they broke into squares first
          av_opt_set(options, "spatial_aq", "1", 0);
          // HEVC and AV1: a quick look at the frame at quarter size first spreads the bits better (sharper text and
          // edges for the same bitrate). On the encoder chip, a frame's encode takes a little longer, no frame later
          if (config.codec != codec_config_t::H264) {
            av_opt_set(options, "multipass", "qres", 0);
          }
          av_opt_set_int(options, "aq-strength", 8, 0);
          av_opt_set(options, "temporal_aq", "0", 0);
          av_opt_set_int(options, "lookahead_depth", 0, 0);
          av_opt_set_int(options, "rc_lookahead", 0, 0);
          av_opt_set(options, "gpu", 0, 0);
        }

        av_opt_set_int(options, "strict_gop", 1, 0);
        // without this a requested keyframe is an intra frame that a viewer can't start from
        av_opt_set_int(options, "forced-idr", 1, 0);
        av_opt_set_int(options, "no-scenecut", 1, 0);
        // A keyframe every gop is several times a normal frame and comes out at once: a burst the upload paces out
        // over the next frames, a short hitch every two seconds. Intra refresh renews the picture a part at a time
        // over the gop instead, every frame about the same size. Keyframes then come only when asked for (a viewer
        // starting, or one that lost a frame). Not every GPU has it, the open is tried again without
        if (intra_refresh && !basic) {
          av_opt_set_int(options, "intra-refresh", 1, 0);
          encoder->intra_refresh = true;
        }

        switch (config.rate_control) {
        case codec_config_t::CBR: av_opt_set(options, "rc", "cbr", 0); break;
        case codec_config_t::VBR: av_opt_set(options, "rc", "vbr", 0); break;
        case codec_config_t::CRF:
          av_opt_set(options, "rc", "constqp", 0);
          av_opt_set_int(options, "qp", config.crf_value, 0);
          // qp 0 is the lossless mode, but nvenc only honours it with the lossless tuning
          if (config.crf_value == 0) {
            av_opt_set(options, "tune", "lossless", 0);
          }
          break;
        }
      }
      else if (name.find("amf") != std::string::npos) {
        av_opt_set(options, "quality", "speed", 0);
        av_opt_set(options, "rc", "vbr_latency", 0);
        av_opt_set_int(options, "preanalysis", 0, 0);
      }
      else if (name.find("vaapi") != std::string::npos) {
        av_opt_set_int(options, "low_power", low_power ? 1 : 0, 0);
      }
      else if (name == "libx264" || name == "libx265") {
        av_opt_set(options, "preset", "ultrafast", 0);
        av_opt_set(options, "tune", "zerolatency", 0);
        if (config.rate_control == codec_config_t::CRF) {
          // crf 0 is lossless in x264 and x265
          av_opt_set_int(options, "crf", config.crf_value, 0);
        }
        else {
          // without a buffer limit one busy frame can be many times the average, and the viewer waits for all of it
          encoder->codec_ctx->rc_max_rate = config.bitrate;
          encoder->codec_ctx->rc_buffer_size = config.bitrate / 2;
        }
      }

      // the gpu converter writes bt.601 limited range (nv12.frag), and so does swscale on the cpu path. Saying so in
      // the bitstream keeps a decoder from guessing bt.709 for 1080p, which came out darker and yellow
      encoder->codec_ctx->colorspace = AVCOL_SPC_BT470BG;
      encoder->codec_ctx->color_primaries = AVCOL_PRI_BT709;
      encoder->codec_ctx->color_trc = AVCOL_TRC_BT709;
      encoder->codec_ctx->color_range = AVCOL_RANGE_MPEG;

      int open_result = avcodec_open2(encoder->codec_ctx, encoder->codec, nullptr);
      if (open_result == 0) {
        if (name.find("nvenc") != std::string::npos) {
          fan::print_impl("encoder: " + name + (encoder->gpu ? " (gpu frames)" : " (cpu frames)"));
        }
        return true;

      }
      fan::print_impl("encoder open failed: " + name + (use_gpu ? " gpu" : " cpu") + " (" + std::to_string(open_result) + ")");

      avcodec_free_context(&encoder->codec_ctx);
      if (encoder->hw_frames_ctx) av_buffer_unref(&encoder->hw_frames_ctx);
      destroy_gpu_state(encoder);
      encoder->intra_refresh = false;
      // a GPU without intra refresh refuses the whole open: the same encoder again, with keyframes as before
      if (intra_refresh && name.find("nvenc") != std::string::npos) {
        return open_codec(encoder, config, name, use_gpu, low_power, false);
      }
      return false;

    }

    bool libav_encoder_t::init_encoder_state(encoder_state_t* encoder, const codec_config_t& config_in, const fan::vec2& screencap_res) {
      cleanup_encoder_state(encoder);
      // 4:2:0 (NV12) is made of 2x2 blocks: an odd size, which a shared window easily has (1919x1037), can't be a GPU
      // input texture at all, every frame init failed and the frames went through the CPU. The picture is scaled to one pixel less instead
      codec_config_t config = config_in;
      config.width = std::max<decltype(config.width)>(2, config.width & ~(decltype(config.width))1);
      config.height = std::max<decltype(config.height)>(2, config.height & ~(decltype(config.height))1);

      encoder->config = config;
      encoder->requested = config_in;
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

      encoder->frame->format = cpu_pixel_format(encoder);
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
      int threads = std::clamp((int)std::thread::hardware_concurrency() / 4, 1, 8);

      av_opt_set_int(ctx, "srcw", encoder->screencap_res.x, 0);
      av_opt_set_int(ctx, "srch", encoder->screencap_res.y, 0);
      av_opt_set_pixel_fmt(ctx, "src_format", AV_PIX_FMT_BGRA, 0);
      av_opt_set_int(ctx, "dstw", config.width, 0);
      av_opt_set_int(ctx, "dsth", config.height, 0);
      av_opt_set_pixel_fmt(ctx, "dst_format", cpu_pixel_format(encoder), 0);
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
      fan::print_impl("encoder: opened again for a changed setting (codec, encoder or GPU/CPU choice)");
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

      if (!convert_frame(encoder, bgra_data, stride)) return results;

      if (encoder->hw_frames_ctx) {
        AVFrame* surface = av_frame_alloc();
        if (surface && av_hwframe_get_buffer(encoder->hw_frames_ctx, surface, 0) >= 0 &&
          av_hwframe_transfer_data(surface, encoder->frame, 0) >= 0) {
          submit_frame(encoder, surface, pts, force_keyframe, results);
        }
        av_frame_free(&surface);
        return results;
      }

      submit_frame(encoder, encoder->frame, pts, force_keyframe, results);
      return results;

    }

    bool libav_encoder_t::submit_frame(encoder_state_t* encoder, AVFrame* frame, std::int64_t pts, bool force_keyframe, std::vector<encode_result_t>& results) {
      frame->pts = pts;
      frame->pict_type = force_keyframe ? AV_PICTURE_TYPE_I : AV_PICTURE_TYPE_NONE;

      if (avcodec_send_frame(encoder->codec_ctx, frame) < 0) return false;

      while (avcodec_receive_packet(encoder->codec_ctx, encoder->packet) >= 0) {
        encode_result_t result;

        result.data.assign(encoder->packet->data, encoder->packet->data + encoder->packet->size);
        result.is_keyframe = (encoder->packet->flags & AV_PKT_FLAG_KEY) != 0;
        // AV1 from NVENC: its key frames didn't all come out marked, viewers then waited 3 s for a keyframe that had
        // come, gave up and decoded frames whose reference was missing (an old or black picture). The frame header
        // says it for sure
        if (!result.is_keyframe && encoder->codec_name.find("av1") != std::string::npos) {
          result.is_keyframe = av1_has_key_frame(encoder->packet->data, (std::size_t)encoder->packet->size);
        }
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
        else if (encoder->codec_name == "libx264" || encoder->codec_name == "libx265") {
          encoder->codec_ctx->rc_max_rate = encoder->config.bitrate;
          encoder->codec_ctx->rc_buffer_size = encoder->config.bitrate / 2;
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

    // slice NAL units (types 1 and 5) in one annex b access unit
    static int count_h264_slices(const std::uint8_t* data, std::size_t size) {
      int slices = 0;
      for (std::size_t i = 0; i + 3 < size; ++i) {
        if (data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1) {
          std::uint8_t type = data[i + 3] & 0x1f;
          slices += type == 1 || type == 5;
          i += 3;
        }
      }
      return slices;
    }

    bool libav_decoder_t::find_and_init_hw_decoder(const std::string& codec_type) {
      if (!hw_device_ctx_) return false;

      auto hw_decoders = get_hw_decoder_names();
      auto it = hw_decoders.find(codec_type);
      if (it == hw_decoders.end()) return false;
      // the card failed on this codec before: the CPU this time, if it has a decoder for it (AV1 has none here)
      if (codec_type == hw_failed_codec_ && avcodec_find_decoder_by_name(codec_type.c_str()) &&
        (codec_type != "av1")) {
        return false;
      }

      // D3D11VA, DXVA2, VAAPI (and NVDEC after cuvid) decode through FFmpeg's own decoder of the codec with the card
      // attached. There are no decoders named h264_d3d11va: AMD and Intel cards never decoded on the GPU, and AV1,
      // which has no CPU decoder in this FFmpeg, didn't show at all
      std::vector<std::string> names = it->second;
      if (hw_type_ == AV_HWDEVICE_TYPE_D3D11VA || hw_type_ == AV_HWDEVICE_TYPE_DXVA2 || hw_type_ == AV_HWDEVICE_TYPE_VAAPI ||
        hw_type_ == AV_HWDEVICE_TYPE_CUDA) {
        names.push_back(codec_type);
      }

      for (const auto& decoder_name : names) {
        bool native = decoder_name == codec_type;
        bool type_match = native;

        if (!native) {
          switch (hw_type_) {
          case AV_HWDEVICE_TYPE_CUDA: type_match = decoder_name.find("cuvid") != std::string::npos; break;

          case AV_HWDEVICE_TYPE_QSV: type_match = decoder_name.find("qsv") != std::string::npos; break;
          case AV_HWDEVICE_TYPE_VAAPI: type_match = decoder_name.find("vaapi") != std::string::npos; break;

          case AV_HWDEVICE_TYPE_DXVA2: type_match = decoder_name.find("dxva2") != std::string::npos; break;
          case AV_HWDEVICE_TYPE_D3D11VA: type_match = decoder_name.find("d3d11va") != std::string::npos; break;

          case AV_HWDEVICE_TYPE_VIDEOTOOLBOX: type_match = decoder_name.find("videotoolbox") != std::string::npos; break;
          default: break;
          }
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
        else if (native) {
          // a frame out as soon as it is decoded: frame threads would each hold one back
          codec_ctx_->flags |= AV_CODEC_FLAG_LOW_DELAY;
          codec_ctx_->thread_count = 1;
        }

        // the packets carry no real clock, only an order: said here, the cuvid decoders stop warning
        // "Invalid pkt_timebase, passing timestamps as-is" on every open
        codec_ctx_->pkt_timebase = AVRational {1, 1000};

        // a driver that hangs in the open leaves this thread running after the timeout below,
        // so everything it touches has to outlive this scope
        struct open_state_t {
          std::atomic<bool> completed {false};
          std::atomic<int> result {-1};
        };
        auto open_state = std::make_shared<open_state_t>();
        auto& open_completed = open_state->completed;
        auto& open_result = open_state->result;

        std::thread open_thread([open_state, ctx = codec_ctx_, codec = codec_]() {
          int result = avcodec_open2(ctx, codec, nullptr);
          open_state->result.store(result);
          open_state->completed.store(true);
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

          // the open is still running inside this context. It is left to leak, freeing it would
          // take it away from under the driver call
          codec_ctx_ = nullptr;
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

      // The sharer changed codec (H.264, HEVC, AV1): the open decoder gets the new one's packets. An H.264 decoder
      // given AV1 finds nothing it knows in them and says nothing, so the picture froze for good. Only what a packet
      // shows for sure switches: AV1 has no start code, HEVC is known by its keyframe's first NAL, H.264 by its SPS
      if (codec_ctx_ && forced_codec == "auto-detect" && size >= 5) {
        bool start_code = (data[0] == 0 && data[1] == 0 && data[2] == 1) || (data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 1);
        std::string seen = detect_codec_from_data(data, size);
        std::size_t nal = data[2] == 1 ? 3 : 4;
        bool h264_sps = start_code && (data[nal] & 0x1F) == 7 && (data[nal] & 0x80) == 0;
        bool sure = (seen == "av1" && !start_code) || seen == "hevc" || (seen == "h264" && (current_codec_ == "av1" ? start_code : h264_sps));
        if (sure && seen != current_codec_) {
          fan::print_impl("the stream changed codec: " + current_codec_ + " -> " + seen);
          if (codec_ctx_->hw_device_ctx) av_buffer_unref(&codec_ctx_->hw_device_ctx);
          avcodec_free_context(&codec_ctx_);
          codec_ = nullptr;
          using_hardware_ = false;
          decode_failures_ = 0;
          codec_switched_ = true;
          if (frame_) av_frame_unref(frame_);
          if (hw_frame_) av_frame_unref(hw_frame_);
        }
      }

      if (!codec_ctx_) {
        std::string detected_codec = forced_codec;
        if (detected_codec == "auto-detect") {
          detected_codec = detect_codec_from_data(data, size);
        }

        sliced_stream_ = detected_codec == "h264" && count_h264_slices(data, size) > 1;
        current_codec_ = detected_codec;

        bool hw_success = false;
        if (allow_hardware_) {
          try {
            hw_success = find_and_init_hw_decoder(detected_codec);
          }
          catch (...) {
            hw_success = false;
          }
        }

        if (!hw_success && !find_and_init_sw_decoder(detected_codec)) {
          return results;
        }
      }

      packet_->data = const_cast<std::uint8_t*>(data);
      packet_->size = size;

      int ret = avcodec_send_packet(codec_ctx_, packet_);
      if (ret < 0 && ret != AVERROR(EAGAIN)) {
        // the card keeps failing on this codec (no support for it): remembered, the reopen after 5 goes to the CPU
        if (++decode_failures_ >= 5 && using_hardware_) {
          hw_failed_codec_ = current_codec_;
        }
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

        // A CUDA frame stays on the GPU when someone takes it as it is (copied there into memory the renderer can
        // read): no download into system memory and no upload back. 8-bit 4:2:0 only, the rest downloads below
        if (using_hardware_ && target_frame->format == AV_PIX_FMT_CUDA && gpu_frame_hook && target_frame->hw_frames_ctx) {
          auto* frames = reinterpret_cast<AVHWFramesContext*>(target_frame->hw_frames_ctx->data);
          if (frames->sw_format == AV_PIX_FMT_NV12) {
            // AVCUDADeviceContext starts with the CUcontext the decoder's memory belongs to
            void* cuda_context = *reinterpret_cast<void**>(frames->device_ctx->hwctx);
            std::array<const std::uint8_t*, 2> planes {target_frame->data[0], target_frame->data[1]};
            std::array<int, 2> pitches {target_frame->linesize[0], target_frame->linesize[1]};
            int slot = gpu_frame_hook(cuda_context, planes, pitches, fan::vec2ui(target_frame->width, target_frame->height));
            if (slot >= 0) {
              result.image_size = {actual_width, actual_height};
              result.pixel_format = AV_PIX_FMT_NV12;
              result.gpu_slot = slot;
              av_frame_unref(frame_);
              av_frame_unref(hw_frame_);
              results.push_back(std::move(result));
              continue;
            }
          }
        }

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

    // HEVC keeps its VPS, SPS and PPS in an hvcC box, laid out differently from H.264's avcC below. Read as avcC
    // they came out as garbage, and a file that has them only there (MP4 wallpapers often do: the first frame starts
    // with the picture itself) could not be decoded at all
    if (par->codec_id == AV_CODEC_ID_HEVC) {
      if (size < 23) {
        return;
      }
      nal_length_size_ = (data[21] & 0x03) + 1;
      size_t pos = 22;
      uint8_t arrays = data[pos++];
      for (uint8_t a = 0; a < arrays; ++a) {
        if (pos + 3 > size) return;
        pos += 1;   // completeness and the NAL type, the NALs carry their type themselves
        uint16_t count = (uint16_t(data[pos]) << 8) | uint16_t(data[pos + 1]);
        pos += 2;
        for (uint16_t n = 0; n < count; ++n) {
          if (pos + 2 > size) return;
          uint16_t nal_size = (uint16_t(data[pos]) << 8) | uint16_t(data[pos + 1]);
          pos += 2;
          if (pos + nal_size > size) return;
          sps_pps_.insert(sps_pps_.end(), {0x00,0x00,0x00,0x01});
          sps_pps_.insert(sps_pps_.end(), data + pos, data + pos + nal_size);
          pos += nal_size;
        }
      }
      return;
    }

    nal_length_size_ = (data[4] & 0x03) + 1;

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
