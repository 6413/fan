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
#if defined(_WIN32)
  #include <WITCH/MD/SCR/SCR.h>
  #include <dxgi1_6.h>
  #include <d3dcompiler.h>
  #pragma comment(lib, "d3dcompiler.lib")
#endif
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
  release_com(target_view);
  release_com(hdr_view);
  release_com(hdr_texture);
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

  if (duplicate() != S_OK) {
    release_duplication();
    return false;
  }
  return true;
}

void fan::graphics::desktop_capture_t::close() {
  if (context) {
    release_textures();
  }
  release_window_buffer();
  release_tonemap();
  release_com(duplication);
  release_com(output);
  release_com(context);
  release_com(device);
}

void fan::graphics::desktop_capture_t::release_window_buffer() {
  if (window_dc) {
    SelectObject((HDC)window_dc, (HGDIOBJ)window_old_bitmap);
    if (window_bitmap) DeleteObject((HGDIOBJ)window_bitmap);
    DeleteDC((HDC)window_dc);
  }
  window_dc = nullptr;
  window_bitmap = nullptr;
  window_old_bitmap = nullptr;
  window_bits = nullptr;
  window_buffer_width = window_buffer_height = 0;
}

// One window instead of the monitor: PrintWindow asks the window itself to paint into a bitmap, so what covers it
// doesn't show. The bitmap goes to the same texture the desktop frames use, the rest of the pipeline is the same
bool fan::graphics::desktop_capture_t::acquire_window(std::uint32_t timeout_ms) {
  HWND hwnd = (HWND)target_window.load();
  if (!IsWindow(hwnd)) {
    // closed: back to the monitor
    target_window = 0;
    release_window_buffer();
    return false;
  }
  // no wait here: the encode loop paces the frames, a wait on top of it halved the rate
  (void)timeout_ms;
  RECT rect;
  if (IsIconic(hwnd) || !GetWindowRect(hwnd, &rect)) {
    return false;
  }
  std::int32_t width = rect.right - rect.left;
  std::int32_t height = rect.bottom - rect.top;
  if (width < 16 || height < 16) {
    return false;
  }
  desktop_origin = fan::vec2i(rect.left, rect.top);

  if (!window_dc || width != window_buffer_width || height != window_buffer_height) {
    release_window_buffer();
    HDC screen_dc = GetDC(nullptr);
    HDC dc = CreateCompatibleDC(screen_dc);
    ReleaseDC(nullptr, screen_dc);
    BITMAPINFO info {};
    info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    info.bmiHeader.biWidth = width;
    info.bmiHeader.biHeight = -height; // top down
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = BI_RGB;
    void* bits = nullptr;
    HBITMAP bitmap = CreateDIBSection(dc, &info, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (!dc || !bitmap) {
      if (bitmap) DeleteObject(bitmap);
      if (dc) DeleteDC(dc);
      return false;
    }
    window_old_bitmap = SelectObject(dc, bitmap);
    window_dc = dc;
    window_bitmap = bitmap;
    window_bits = (std::uint8_t*)bits;
    window_buffer_width = width;
    window_buffer_height = height;
  }

  // 2 is PW_RENDERFULLCONTENT: windows drawn by the compositor (browsers, most games in a window) too
  if (!PrintWindow(hwnd, (HDC)window_dc, 2)) {
    return false;
  }
  std::uint32_t* pixels = (std::uint32_t*)window_bits;
  for (std::size_t i = 0, count = (std::size_t)width * height; i < count; ++i) {
    pixels[i] |= 0xff000000u;
  }
  if ((std::int32_t)Geometry.Resolution.x != width || (std::int32_t)Geometry.Resolution.y != height) {
    if (!create_textures(width, height)) {
      return false;
    }
  }
  context->UpdateSubresource(texture, 0, nullptr, window_bits, width * 4, 0);
  has_frame = true;
  return true;
}

// Starts duplicating the screen. Asking for the 16 bit float format first: an HDR screen then comes in its own
// format, an SDR screen still in 8 bit. The old call is the fallback (Windows before 10 1703)
HRESULT fan::graphics::desktop_capture_t::duplicate() {
  IDXGIOutput5* output5 = nullptr;
  if (output->QueryInterface(IID_PPV_ARGS(&output5)) == S_OK) {
    const DXGI_FORMAT formats[] = {DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_B8G8R8A8_UNORM};
    // this call wants a caller that is per monitor DPI aware (v2), for this thread only while it runs
    using set_context_t = HANDLE(WINAPI*)(HANDLE);
    auto set_context = (set_context_t)GetProcAddress(GetModuleHandleW(L"user32.dll"), "SetThreadDpiAwarenessContext");
    HANDLE previous = set_context ? set_context((HANDLE)-4) : nullptr; // DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
    HRESULT result = output5->DuplicateOutput1(device, 0, 2, formats, &duplication);
    if (set_context && previous) {
      set_context(previous);
    }
    output5->Release();
    if (result == S_OK) {
      sdr_white_nits = query_sdr_white_nits();
      // the brightest the monitor shows, the tone mapping reaches full white there
      IDXGIOutput6* output6 = nullptr;
      DXGI_OUTPUT_DESC1 desc1 {};
      peak_nits = 1000.f;
      if (output->QueryInterface(IID_PPV_ARGS(&output6)) == S_OK) {
        if (output6->GetDesc1(&desc1) == S_OK && desc1.MaxLuminance > 100.f) {
          peak_nits = desc1.MaxLuminance;
        }
        output6->Release();
      }
      return S_OK;
    }
  }
  return output->DuplicateOutput(device, &duplication);
}

// How bright SDR white is on this screen while it is in HDR (Settings > Display > HDR > SDR content brightness).
// The tone mapping puts that brightness at full white, so the picture looks like it does on the screen
f32_t fan::graphics::desktop_capture_t::query_sdr_white_nits() {
  DXGI_OUTPUT_DESC output_desc {};
  if (output->GetDesc(&output_desc) != S_OK) {
    return 200.f;
  }
  UINT32 path_count = 0, mode_count = 0;
  if (GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &path_count, &mode_count) != ERROR_SUCCESS) {
    return 200.f;
  }
  std::vector<DISPLAYCONFIG_PATH_INFO> paths(path_count);
  std::vector<DISPLAYCONFIG_MODE_INFO> modes(mode_count);
  if (QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &path_count, paths.data(), &mode_count, modes.data(), nullptr) != ERROR_SUCCESS) {
    return 200.f;
  }
  // spelled out here, older SDK headers leave the SDR white level request out
  struct sdr_white_level_t {
    DISPLAYCONFIG_DEVICE_INFO_HEADER header;
    ULONG level; // 1000 is 80 nits
  };
  for (UINT32 i = 0; i < path_count; ++i) {
    DISPLAYCONFIG_SOURCE_DEVICE_NAME source {};
    source.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
    source.header.size = sizeof(source);
    source.header.adapterId = paths[i].sourceInfo.adapterId;
    source.header.id = paths[i].sourceInfo.id;
    if (DisplayConfigGetDeviceInfo(&source.header) != ERROR_SUCCESS || std::wcscmp(source.viewGdiDeviceName, output_desc.DeviceName) != 0) {
      continue;
    }
    sdr_white_level_t white {};
    white.header.type = (DISPLAYCONFIG_DEVICE_INFO_TYPE)11; // DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL
    white.header.size = sizeof(white);
    white.header.adapterId = paths[i].targetInfo.adapterId;
    white.header.id = paths[i].targetInfo.id;
    if (DisplayConfigGetDeviceInfo(&white.header) == ERROR_SUCCESS && white.level > 0) {
      return white.level / 1000.f * 80.f;
    }
  }
  return 200.f;
}

// one triangle over the whole picture. Each pixel: scaled so SDR white is 1.0, brighter parts rolled off softly
// instead of clipped (by luminance, so colours keep their hue), then put into sRGB like any 8 bit screen picture
static const char tonemap_hlsl[] = R"(
cbuffer params : register(b0) { float white_scale; float peak; float knee; float unused; };
Texture2D<float4> source : register(t0);

float4 vs(uint id : SV_VertexID) : SV_Position {
  float2 uv = float2((id << 1) & 2, id & 2);
  return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}

float3 to_srgb(float3 c) {
  return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

float4 ps(float4 position : SV_Position) : SV_Target {
  float3 c = max(source.Load(int3(position.xy, 0)).rgb * white_scale, 0.0);
  float luminance = dot(c, float3(0.2126, 0.7152, 0.0722));
  if (luminance > knee) {
    float room = 1.0 - knee;
    float x = (luminance - knee) / room;
    float w = max((peak - knee) / room, 1.0);
    float r = x * (1.0 + x / (w * w)) / (1.0 + x);
    c *= (knee + room * min(r, 1.0)) / luminance;
  }
  float brightest = max(c.r, max(c.g, c.b));
  if (brightest > 1.0) {
    c /= brightest;
  }
  return float4(to_srgb(c), 1.0);
}
)";

bool fan::graphics::desktop_capture_t::create_tonemap() {
  ID3DBlob* vs_code = nullptr;
  ID3DBlob* ps_code = nullptr;
  ID3DBlob* errors = nullptr;
  bool ok =
    D3DCompile(tonemap_hlsl, sizeof(tonemap_hlsl) - 1, "tonemap", nullptr, nullptr, "vs", "vs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &vs_code, &errors) == S_OK &&
    D3DCompile(tonemap_hlsl, sizeof(tonemap_hlsl) - 1, "tonemap", nullptr, nullptr, "ps", "ps_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &ps_code, &errors) == S_OK &&
    device->CreateVertexShader(vs_code->GetBufferPointer(), vs_code->GetBufferSize(), nullptr, &tonemap_vs) == S_OK &&
    device->CreatePixelShader(ps_code->GetBufferPointer(), ps_code->GetBufferSize(), nullptr, &tonemap_ps) == S_OK;
  if (!ok && errors) {
    fan::print_impl(std::string("HDR tone mapping shader: ") + (const char*)errors->GetBufferPointer());
  }
  release_com(vs_code);
  release_com(ps_code);
  release_com(errors);
  if (ok) {
    D3D11_BUFFER_DESC desc {};
    desc.ByteWidth = 16;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    ok = device->CreateBuffer(&desc, nullptr, &tonemap_params) == S_OK;
  }
  if (!ok) {
    release_tonemap();
  }
  return ok;
}

void fan::graphics::desktop_capture_t::release_tonemap() {
  release_com(tonemap_vs);
  release_com(tonemap_ps);
  release_com(tonemap_params);
}

bool fan::graphics::desktop_capture_t::tonemap_frame(ID3D11Texture2D* desktop, std::uint32_t width, std::uint32_t height) {
  if (!tonemap_ps && !create_tonemap()) {
    return false;
  }
  if (!hdr_texture) {
    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    if (device->CreateTexture2D(&desc, nullptr, &hdr_texture) != S_OK ||
      device->CreateShaderResourceView(hdr_texture, nullptr, &hdr_view) != S_OK) {
      release_com(hdr_view);
      release_com(hdr_texture);
      return false;
    }
  }
  if (!target_view && device->CreateRenderTargetView(texture, nullptr, &target_view) != S_OK) {
    return false;
  }

  // the duplicated frame can't be read by a shader, it is copied to one that can
  context->CopyResource(hdr_texture, desktop);
  f32_t exposure = std::clamp(hdr_exposure.load(), 0.2f, 2.f);
  f32_t white = std::max(sdr_white_nits, 1.f);
  const f32_t params[4] = {
    80.f / white * exposure,                          // 1.0 = SDR white, times the exposure
    std::max(peak_nits / white * exposure, 1.5f),     // the monitor's peak on that scale
    0.5f,                                             // below this nothing changes
    0
  };
  context->UpdateSubresource(tonemap_params, 0, nullptr, params, 0, 0);

  D3D11_VIEWPORT viewport {0, 0, (f32_t)width, (f32_t)height, 0, 1};
  context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  context->IASetInputLayout(nullptr);
  context->VSSetShader(tonemap_vs, nullptr, 0);
  context->PSSetShader(tonemap_ps, nullptr, 0);
  context->PSSetConstantBuffers(0, 1, &tonemap_params);
  context->PSSetShaderResources(0, 1, &hdr_view);
  context->RSSetState(nullptr);
  context->RSSetViewports(1, &viewport);
  context->OMSetBlendState(nullptr, nullptr, 0xffffffff);
  context->OMSetDepthStencilState(nullptr, 0);
  context->OMSetRenderTargets(1, &target_view, nullptr);
  context->Draw(3, 0);

  // nothing stays bound, the encoder reads the texture next
  ID3D11ShaderResourceView* no_view = nullptr;
  ID3D11RenderTargetView* no_target = nullptr;
  context->PSSetShaderResources(0, 1, &no_view);
  context->OMSetRenderTargets(1, &no_target, nullptr);
  return true;
}

bool fan::graphics::desktop_capture_t::acquire(std::uint32_t timeout_ms) {
  if (target_window.load() != 0) {
    release_com(duplication);
    return acquire_window(timeout_ms);
  }
  if (!duplication) {
    if (std::chrono::steady_clock::now() < next_reopen || duplicate() != S_OK) {
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
      hdr_active = desc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT;
      if (desc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT) {
        new_frame = tonemap_frame(desktop, desc.Width, desc.Height);
      }
      else {
        context->CopyResource(texture, desktop);
        new_frame = true;
      }
      has_frame |= new_frame;
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

  bool follow_capture_size = !user_set_resolution &&
    (config_.width != mdscr.Geometry.Resolution.x || config_.height != mdscr.Geometry.Resolution.y);

  // the desktop's size can change while streaming. When the output follows it the encoder is opened
  // below with the new config, opening it here as well would do the whole thing twice
  if (!follow_capture_size &&
    (encoder_.screencap_res_.x != mdscr.Geometry.Resolution.x || encoder_.screencap_res_.y != mdscr.Geometry.Resolution.y)) {
    if (!encoder_.open(config_, fan::vec2(mdscr.Geometry.Resolution.x, mdscr.Geometry.Resolution.y))) return false;
  }

  if (follow_capture_size) {
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

  // taken in one step, a request that arrives during the encode stays set for the next frame
  bool force_keyframe = (encode_write_flags.fetch_and((std::uint8_t)~codec_update_e::force_keyframe) & codec_update_e::force_keyframe) != 0;
  // a frame that never reaches the encoder must not use the request up
  auto keep_keyframe_request = [&] {
    if (force_keyframe) {
      encode_write_flags |= codec_update_e::force_keyframe;
    }
  };

  std::vector<libav_encoder_t::encode_result_t> results;

#if defined(_WIN32)
  if (encoder_.uses_gpu_frames()) {
    results = encoder_.encode_texture(mdscr.texture, frame_timestamp_, force_keyframe);
  }
  else
#endif
  {
    if (!screen_buffer) {
      keep_keyframe_request();
      return false;
    }

    if (mdscr.Geometry.Resolution.x <= 0 || mdscr.Geometry.Resolution.y <= 0) {
      fan::print_impl("Error: invalid screen resolution");

      keep_keyframe_request();
      return false;
    }

    if (mdscr.Geometry.LineSize < mdscr.Geometry.Resolution.x * 4) {
      fan::print_impl("Error: LineSize too small for BGRA.");

      keep_keyframe_request();
      return false;
    }

    results = encoder_.encode_frame(screen_buffer, mdscr.Geometry.LineSize, frame_timestamp_, force_keyframe);
  }

  std::lock_guard<std::mutex> lock(data_mutex);

  for (auto& result : results) {
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
  std::uint64_t interval = 1000000000ULL / (std::max)(config_.frame_rate, 1);
  std::uint64_t now = fan::time::now();

  // The next frame is due one interval after the last one was due, not after it arrived: a late frame makes the
  // next wait shorter, so the rate stays at frame_rate on average. Waiting a full interval from the arrival, and
  // then for the screen's next refresh on top, lost a refresh every frame (60 fps on a 60 Hz screen came out ~30).
  // The capture waits for the screen's next frame after this, so the wait ends a little early to not miss it
  std::uint64_t due = frame_process_start_time_ + interval;
  if (due < now || due > now + interval * 2) {
    due = now; // far behind, or the clock was reset: start over from now
  }
  frame_process_start_time_ = due;
  std::uint64_t slack = (std::min)(interval / 4, (std::uint64_t)4000000);
  if (due > now + slack) {
    precise_sleep(due - slack - now);
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
