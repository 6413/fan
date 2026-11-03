module;

#if defined (FAN_WINDOW) && defined(FAN_VIDEO)

#include <cstring>
#include <libavformat/avformat.h>
extern "C" {
#include <libavutil/pixdesc.h>
}

#if defined(FAN_2D)
#include <fan/utils/utility.h>
#include <WITCH/WITCH.h>
#include <WITCH/PR/PR.h>
#endif

#endif

module fan.graphics.video.screen;

#if defined (FAN_WINDOW) && defined(FAN_VIDEO)

#if defined(FAN_2D)

#if defined(_WIN32)

template <typename T>
static void release_com(T*& p) {
  if (p) {
    p->Release();
    p = nullptr;
  }
}

bool fan::graphics::desktop_capture_t::create_textures(std::uint32_t width, std::uint32_t height) {
  release_textures();

  D3D11_TEXTURE2D_DESC desc {};
  desc.Width = width;
  desc.Height = height;
  desc.MipLevels = 1;
  desc.ArraySize = 1;
  desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
  desc.SampleDesc.Count = 1;
  desc.Usage = D3D11_USAGE_DEFAULT;
  desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
  if (device->CreateTexture2D(&desc, nullptr, &texture) != S_OK) return false;

  desc.Usage = D3D11_USAGE_STAGING;
  desc.BindFlags = 0;
  desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
  for (auto& s : staging) {
    if (device->CreateTexture2D(&desc, nullptr, &s) != S_OK) return false;
  }

  Geometry.Resolution.x = width;
  Geometry.Resolution.y = height;
  return true;
}

void fan::graphics::desktop_capture_t::release_textures() {
  if (mapped) {
    context->Unmap(staging[mapped_index], 0);
    mapped = nullptr;
  }
  release_com(texture);
  release_com(staging[0]);
  release_com(staging[1]);
  staging_ready = false;
  has_frame = false;
}

void fan::graphics::desktop_capture_t::release_duplication() {
  release_com(duplication);
  next_reopen = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
}

bool fan::graphics::desktop_capture_t::open() {
  IDXGIFactory1* factory = nullptr;
  if (CreateDXGIFactory1(IID_PPV_ARGS(&factory)) != S_OK) return false;

  // the first output, on the adapter that drives it
  IDXGIAdapter1* adapter = nullptr;
  IDXGIOutput* first_output = nullptr;
  for (UINT i = 0; !first_output && factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
    if (adapter->EnumOutputs(0, &first_output) != S_OK) {
      first_output = nullptr;
      release_com(adapter);
    }
  }
  factory->Release();
  if (!first_output) return false;

  const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
  HRESULT result = D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
    levels, 2, D3D11_SDK_VERSION, &device, nullptr, &context);
  if (result != S_OK) {
    result = D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0,
      nullptr, 0, D3D11_SDK_VERSION, &device, nullptr, &context);
  }
  adapter->Release();

  DXGI_OUTPUT_DESC output_desc {};
  bool ok = result == S_OK &&
    first_output->GetDesc(&output_desc) == S_OK &&
    first_output->QueryInterface(IID_PPV_ARGS(&output)) == S_OK;
  first_output->Release();
  if (!ok) return false;

  auto& rect = output_desc.DesktopCoordinates;
  if (!create_textures(rect.right - rect.left, rect.bottom - rect.top)) return false;

  if (output->DuplicateOutput(device, &duplication) != S_OK) {
    release_duplication();
    return false;
  }
  return true;
}

void fan::graphics::desktop_capture_t::close() {
  if (context) {
    release_textures();
  }
  release_com(duplication);
  release_com(output);
  release_com(context);
  release_com(device);
}

bool fan::graphics::desktop_capture_t::acquire(std::uint32_t timeout_ms) {
  if (!duplication) {
    if (std::chrono::steady_clock::now() < next_reopen || output->DuplicateOutput(device, &duplication) != S_OK) {
      release_duplication();
      std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms));
      return false;
    }
  }

  DXGI_OUTDUPL_FRAME_INFO info;
  IDXGIResource* resource = nullptr;
  HRESULT result = duplication->AcquireNextFrame(timeout_ms, &info, &resource);
  if (result == DXGI_ERROR_WAIT_TIMEOUT) {
    return false;
  }
  if (result != S_OK) {
    // mode changes, fullscreen switches and the secure desktop end up here
    release_duplication();
    return false;
  }

  bool new_frame = false;
  ID3D11Texture2D* desktop = nullptr;
  // without a present time only the mouse moved, the image is unchanged
  if (info.LastPresentTime.QuadPart != 0 && resource->QueryInterface(IID_PPV_ARGS(&desktop)) == S_OK) {
    D3D11_TEXTURE2D_DESC desc;
    desktop->GetDesc(&desc);
    if ((desc.Width == Geometry.Resolution.x && desc.Height == Geometry.Resolution.y) || create_textures(desc.Width, desc.Height)) {
      context->CopyResource(texture, desktop);
      new_frame = true;
      has_frame = true;
    }
    desktop->Release();
  }
  resource->Release();
  duplication->ReleaseFrame();
  return new_frame;
}

std::uint8_t* fan::graphics::desktop_capture_t::map_for_cpu(bool new_frame) {
  if (mapped && !new_frame) {
    return mapped;
  }
  if (mapped) {
    context->Unmap(staging[mapped_index], 0);
    mapped = nullptr;
  }

  std::uint32_t written = staging_index;
  context->CopyResource(staging[written], texture);
  mapped_index = staging_ready ? written ^ 1 : written;
  staging_index = written ^ 1;
  staging_ready = true;

  D3D11_MAPPED_SUBRESOURCE map;
  if (context->Map(staging[mapped_index], 0, D3D11_MAP_READ, 0, &map) != S_OK) {
    return nullptr;
  }
  Geometry.LineSize = map.RowPitch;
  mapped = static_cast<std::uint8_t*>(map.pData);
  return mapped;
}

bool fan::graphics::screen_encode_t::read_desktop() {
  std::uint32_t wait_ms = (std::max)(1000 / (std::max)(config_.frame_rate, 1), 1);
  bool new_frame = mdscr.acquire(wait_ms);

  auto now = std::chrono::steady_clock::now();
  if (!new_frame) {
    // a still screen must still answer keyframe requests and show viewers it's alive
    bool keyframe_wanted = (encode_write_flags & codec_update_e::force_keyframe) != 0;
    if (!mdscr.has_frame || (!keyframe_wanted && now - last_read < std::chrono::milliseconds(100))) {
      return false;
    }
  }
  last_read = now;

  if (encoder_.uses_gpu_frames()) {
    screen_buffer = nullptr;
  }
  else {
    screen_buffer = mdscr.map_for_cpu(new_frame);
    if (!screen_buffer) {
      return false;
    }
  }
  capture_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - now).count();
  return true;
}

#endif

bool fan::graphics::screen_encode_t::encode_write() {
  if (!ensure_mdscr_initialized()) return false;

  // the desktop's size can change while streaming
  if (encoder_.screencap_res_.x != mdscr.Geometry.Resolution.x || encoder_.screencap_res_.y != mdscr.Geometry.Resolution.y) {
    if (!encoder_.open(config_, fan::vec2(mdscr.Geometry.Resolution.x, mdscr.Geometry.Resolution.y))) return false;
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

    if (!encoder_.open(config_, fan::vec2(mdscr.Geometry.Resolution.x, mdscr.Geometry.Resolution.y))) return false;

  }

  if (update_flags != 0) {
    std::lock_guard<std::mutex> lock(mutex);

    if (update_flags & codec_update_e::scaling_quality) {
      if (!encoder_.update_scaling_quality(config_.scaling_quality)) {
        fan::print_impl("Failed to update scaling quality");

      }
    }
    if (update_flags & ~codec_update_e::scaling_quality) {
      encoder_.update_config(config_, update_flags & ~codec_update_e::scaling_quality);

    }
    update_flags = 0;

  }

  bool force_keyframe = (encode_write_flags & codec_update_e::force_keyframe) != 0;

  std::vector<libav_encoder_t::encode_result_t> results;

#if defined(_WIN32)
  if (encoder_.uses_gpu_frames()) {
    results = encoder_.encode_texture(mdscr.texture, frame_timestamp_, force_keyframe);
  }
  else
#endif
  {
    if (!screen_buffer) {
      return false;
    }

    if (mdscr.Geometry.Resolution.x <= 0 || mdscr.Geometry.Resolution.y <= 0) {
      fan::print_impl("Error: invalid screen resolution");

      return false;
    }

    if (mdscr.Geometry.LineSize < mdscr.Geometry.Resolution.x * 4) {
      fan::print_impl("Error: LineSize too small for BGRA.");

      return false;
    }

    results = encoder_.encode_frame(screen_buffer, mdscr.Geometry.LineSize, frame_timestamp_, force_keyframe);
  }

  if (encode_write_flags & codec_update_e::force_keyframe) encode_write_flags &= ~codec_update_e::force_keyframe;

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

// a real wait instead of spinning, so pacing doesn't hold a cpu core
static void precise_sleep(std::uint64_t ns) {
#if defined(_WIN32)
  thread_local HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
  if (timer) {
    LARGE_INTEGER due;
    due.QuadPart = -static_cast<LONGLONG>(ns / 100);
    SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE);
    WaitForSingleObject(timer, INFINITE);
    return;
  }
#endif
  std::this_thread::sleep_for(std::chrono::nanoseconds(ns));
}

void fan::graphics::screen_encode_t::sleep_thread() {
  std::uint64_t target_frame_time_ns = 1000000000ULL / config_.frame_rate;

  std::uint64_t current_time = fan::time::now();
  std::uint64_t time_diff = current_time - frame_process_start_time_;

  if (time_diff > target_frame_time_ns) {
    frame_process_start_time_ = current_time;

  }
  else {
    std::uint64_t sleep_time = target_frame_time_ns - time_diff;

    frame_process_start_time_ = current_time + sleep_time;
    precise_sleep(sleep_time);
  }
}

fan::graphics::screen_decode_t::decode_data_t fan::graphics::screen_decode_t::decode(void* data, std::uintptr_t length, fan::graphics::shape_t& universal_image_renderer) {
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

  // keep the decoder's native planes (yuv420p / nv12); the default converts to rgba,
  // which the switch below doesn't accept
  auto results = decoder_.decode_packet(static_cast<const std::uint8_t*>(data), length, name, false);

  if (results.empty()) {
    // hw decoders can't always recover from a resolution change, reopen on next packet
    if (decoder_.decode_failures_ >= 5) {
      std::lock_guard<std::mutex> lock(mutex);
      update_flags |= codec_update_e::codec;
    }
    ret.type = 252;
    return ret;

  }

  // the newest one, the rest would only be shown late
  auto& frame = results.back();

  if (!frame.success) {
    ret.type = 251;
    return ret;

  }

  ret.image_size = frame.image_size;
  ret.type = 1;

  switch (frame.pixel_format) {
  case AV_PIX_FMT_YUV420P: ret.pixel_format = fan::graphics::image_format_e::yuv420p; break;

  case AV_PIX_FMT_NV12: ret.pixel_format = fan::graphics::image_format_e::nv12; break;
  case AV_PIX_FMT_BGRA: ret.pixel_format = fan::graphics::image_format_e::bgra; break;
  // full-range ("JPEG") 4:2:0 has the same plane layout as yuv420p
  case AV_PIX_FMT_YUVJ420P: ret.pixel_format = fan::graphics::image_format_e::yuv420p; break;
  default: {
    static std::vector<int> reported_formats;
    if (std::find(reported_formats.begin(), reported_formats.end(), (int)frame.pixel_format) == reported_formats.end()) {
      reported_formats.push_back((int)frame.pixel_format);
      const char* format_name = av_get_pix_fmt_name(static_cast<AVPixelFormat>(frame.pixel_format));
      fan::print_impl(std::string("decoder output pixel format not supported: ") + (format_name ? format_name : "unknown"));
    }
    ret.type = 249;
    return ret;
  }

  }

  for (std::size_t i = 0; i < frame.plane_data.size() && i < 4; i++) {
    ret.data[i] = std::move(frame.plane_data[i]);

    ret.stride[i] = { static_cast<std::uint32_t>(frame.linesize[i]), 0 };
  }

  frame_size_ = frame.image_size;
  decoded_size = frame.image_size;

  return ret;
}

#endif

#endif
