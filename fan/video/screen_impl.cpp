module;

#if defined (FAN_WINDOW) && defined(FAN_VIDEO)

#include <cstring>
extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/pixdesc.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}

#if defined(FAN_2D)
#include <fan/utils/utility.h>
#include <WITCH/WITCH.h>
#include <WITCH/PR/PR.h>
#if defined(_WIN32)
  #include <WITCH/MD/SCR/SCR.h>
  #include <dxgi1_6.h>
  #include "wgc_capture.h"
  #include <d3dcompiler.h>
  #pragma comment(lib, "d3dcompiler.lib")
  #define VK_USE_PLATFORM_WIN32_KHR
#else
  #include <dlfcn.h>
  #include <unistd.h>
#endif
#include <vulkan/vulkan.h>
#endif

#if defined(_WIN32)
  #include <mfapi.h>
  #include <mfidl.h>
  #include <mfreadwrite.h>
#elif defined(__linux__)
  #include <cerrno>
  #include <linux/videodev2.h>
  #include <sys/ioctl.h>
  #include <sys/mman.h>
  #include <fcntl.h>
  #include <unistd.h>
  #include <poll.h>
#endif

#endif

module fan.graphics.video.screen;

#if defined (FAN_WINDOW) && defined(FAN_VIDEO) && defined(FAN_2D)
import fan.graphics.vulkan.core;
#endif

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
  ++texture_generation;

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

// how well an output fits: 2 the monitor asked for, 1 the main display, 0 another one, -1 not on the desktop
static int capture_output_score(IDXGIOutput* output, std::uintptr_t want, DXGI_OUTPUT_DESC& desc) {
  if (output->GetDesc(&desc) != S_OK || !desc.AttachedToDesktop) {
    return -1;
  }
  if (want && desc.Monitor == (HMONITOR)want) {
    return 2;
  }
  MONITORINFO info {};
  info.cbSize = sizeof(info);
  return GetMonitorInfoW(desc.Monitor, &info) && (info.dwFlags & MONITORINFOF_PRIMARY) ? 1 : 0;
}

bool fan::graphics::desktop_capture_t::open() {
  IDXGIFactory1* factory = nullptr;
  if (CreateDXGIFactory1(IID_PPV_ARGS(&factory)) != S_OK) return false;

  // the monitor asked for (target_monitor), else the main display, else any, on the adapter that drives it. The first
  // output of the first adapter was taken before, often not the main display
  std::uintptr_t want = target_monitor.load();
  IDXGIAdapter1* adapter = nullptr;
  IDXGIOutput* first_output = nullptr;
  int best = -1;
  IDXGIAdapter1* candidate = nullptr;
  for (UINT i = 0; best < 2 && factory->EnumAdapters1(i, &candidate) != DXGI_ERROR_NOT_FOUND; ++i) {
    IDXGIOutput* out = nullptr;
    for (UINT j = 0; best < 2 && candidate->EnumOutputs(j, &out) != DXGI_ERROR_NOT_FOUND; ++j) {
      DXGI_OUTPUT_DESC desc;
      int score = capture_output_score(out, want, desc);
      if (score > best) {
        best = score;
        release_com(first_output);
        release_com(adapter);
        first_output = out;
        adapter = candidate;
        adapter->AddRef();
      }
      else {
        out->Release();
      }
    }
    candidate->Release();
  }
  factory->Release();
  if (!first_output) {
    release_com(adapter);
    return false;
  }

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

  ++device_generation;
  opened_monitor = (std::uintptr_t)output_desc.Monitor;
  {
    MONITORINFO info {};
    info.cbSize = sizeof(info);
    captured_primary = GetMonitorInfoW(output_desc.Monitor, &info) && (info.dwFlags & MONITORINFOF_PRIMARY) != 0;
  }

  auto& rect = output_desc.DesktopCoordinates;
  monitor_origin = fan::vec2i(rect.left, rect.top);
  if (!create_textures(rect.right - rect.left, rect.bottom - rect.top)) return false;

  if (duplicate() != S_OK) {
    release_duplication();
    return false;
  }
  return true;
}

void fan::graphics::desktop_capture_t::close_wgc() {
  if (wgc_session) {
    fan::graphics::wgc::close((fan::graphics::wgc::session_t*)wgc_session);
    wgc_session = nullptr;
  }
  wgc_window = 0;
}

void fan::graphics::desktop_capture_t::close() {
  close_wgc();
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
    // closed: back to the monitor, and tell the app so a share of that window stops instead of showing the desktop
    target_window_closed = true;
    target_window = 0;
    release_window_buffer();
    close_wgc();
    return false;
  }
  target_window_closed = false;

  // Windows Graphics Capture first: the window's frames on the GPU at its own rate (a game in a window came out at
  // about 5 frames a second through PrintWindow). A window it can't open goes on with PrintWindow below
  if ((std::uintptr_t)hwnd != wgc_refused_window) {
    if (!wgc_session || wgc_window != (std::uintptr_t)hwnd) {
      close_wgc();
      wgc_session = fan::graphics::wgc::open(device, hwnd);
      wgc_window = (std::uintptr_t)hwnd;
      if (!wgc_session) {
        wgc_refused_window = (std::uintptr_t)hwnd;
        fan::print_impl("window capture: Windows Graphics Capture can't take this window, using PrintWindow (slow for games)\n");
      }
      else {
        fan::print_impl("window capture: Windows Graphics Capture\n");
      }
    }
    if (wgc_session) {
      std::int32_t origin_x = 0, origin_y = 0;
      if (fan::graphics::wgc::visible_origin(hwnd, origin_x, origin_y)) {
        desktop_origin = fan::vec2i(origin_x, origin_y);
      }
      ID3D11Texture2D* frame = nullptr;
      std::uint32_t width = 0, height = 0;
      if (!fan::graphics::wgc::take((fan::graphics::wgc::session_t*)wgc_session, &frame, width, height)) {
        // nothing new yet: the encode loop asks again, a short rest keeps this from spinning
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return false;
      }
      bool ok = width >= 16 && height >= 16;
      if (ok && (Geometry.Resolution.x != width || Geometry.Resolution.y != height)) {
        ok = create_textures(width, height);
      }
      if (ok) {
        D3D11_BOX box {0, 0, 0, width, height, 1};
        context->CopySubresourceRegion(texture, 0, 0, 0, 0, frame, 0, &box);
        hdr_active = false;
        has_frame = true;
      }
      frame->Release();
      return ok;
    }
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

// Another monitor on the same graphics card: only the output and its duplication change, the device stays (the
// encoder holds it). One on another card asks for a full reopen, which makes the device for that card
bool fan::graphics::desktop_capture_t::switch_output(std::uintptr_t want) {
  if (!device) {
    restart_requested = true;
    return false;
  }
  IDXGIDevice* dxgi_device = nullptr;
  IDXGIAdapter* adapter = nullptr;
  if (device->QueryInterface(IID_PPV_ARGS(&dxgi_device)) != S_OK) {
    restart_requested = true;
    return false;
  }
  HRESULT result = dxgi_device->GetAdapter(&adapter);
  dxgi_device->Release();
  if (result != S_OK) {
    restart_requested = true;
    return false;
  }
  IDXGIOutput* best_output = nullptr;
  DXGI_OUTPUT_DESC best_desc {};
  int best = -1;
  IDXGIOutput* out = nullptr;
  for (UINT j = 0; adapter->EnumOutputs(j, &out) != DXGI_ERROR_NOT_FOUND; ++j) {
    DXGI_OUTPUT_DESC desc;
    int score = capture_output_score(out, want, desc);
    if (score > best) {
      release_com(best_output);
      best_output = out;
      best_desc = desc;
      best = score;
    }
    else {
      out->Release();
    }
  }
  adapter->Release();
  // the monitor asked for is not on this card: the capture has to open the other card's device
  if (!best_output || (want != 0 && best != 2)) {
    release_com(best_output);
    restart_requested = true;
    return false;
  }
  IDXGIOutput1* next = nullptr;
  if (best_output->QueryInterface(IID_PPV_ARGS(&next)) != S_OK) {
    release_com(best_output);
    restart_requested = true;
    return false;
  }
  release_com(duplication);
  release_com(output);
  output = next;
  opened_monitor = (std::uintptr_t)best_desc.Monitor;
  {
    MONITORINFO info {};
    info.cbSize = sizeof(info);
    captured_primary = GetMonitorInfoW(best_desc.Monitor, &info) && (info.dwFlags & MONITORINFOF_PRIMARY) != 0;
  }
  monitor_origin = fan::vec2i(best_desc.DesktopCoordinates.left, best_desc.DesktopCoordinates.top);
  // duplicated again at once, and the textures follow the new size with the first frame
  next_reopen = std::chrono::steady_clock::time_point {};
  source_changed = true;
  release_com(best_output);
  return true;
}

bool fan::graphics::desktop_capture_t::acquire(std::uint32_t timeout_ms) {
  std::uintptr_t window = target_window.load();
  if (window != 0) {
    if (window != capturing_window) {
      capturing_window = window;
      source_changed = true;
    }
    release_com(duplication);
    return acquire_window(timeout_ms);
  }
  // the whole screen again: the window's capture session isn't needed
  if (wgc_session) {
    close_wgc();
  }
  if (capturing_window != 0) {
    capturing_window = 0;
    source_changed = true;
  }
  // the window capture moved it to the window, the monitor's own corner again
  desktop_origin = monitor_origin;

  auto now = std::chrono::steady_clock::now();
  if (now >= next_restart_try && restart_requested.exchange(false)) {
    next_restart_try = now + std::chrono::seconds(2);
    close();
    if (!open()) {
      release_duplication();
      std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms));
      return false;
    }
    source_changed = true;
    return false; // the next frame comes from the new device
  }

  std::uintptr_t want = target_monitor.load();
  bool want_primary = want == 0;
  bool matches = want_primary ? captured_primary : (opened_monitor == want);
  if (!matches && now >= next_switch_try) {
    next_switch_try = now + std::chrono::milliseconds(500);
    switch_output(want);
  }

  if (!output) {
    restart_requested = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms));
    return false;
  }
  if (!duplication) {
    // waiting out the retry delay must not push it further: re-arming it on every wait (a wait is shorter than the
    // delay) never let the capture reopen after the administrator prompt or a fullscreen switch took it away
    if (std::chrono::steady_clock::now() < next_reopen) {
      std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms));
      return false;
    }
    if (duplicate() != S_OK) {
      // still on the secure desktop (administrator prompt, lock screen): try again shortly
      release_duplication();
      if (++failure_streak >= 30 && now >= next_switch_try) {
        failure_streak = 0;
        next_switch_try = now + std::chrono::milliseconds(500);
        // a monitor turned off and on again can leave the output stale: pick it again
        switch_output(want);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms));
      return false;
    }
    failure_streak = 0;
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
    if (++failure_streak >= 30 && now >= next_switch_try) {
      failure_streak = 0;
      next_switch_try = now + std::chrono::milliseconds(500);
      switch_output(want);
    }
    return false;
  }
  failure_streak = 0;

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
  auto acquire_start = std::chrono::steady_clock::now();
  bool new_frame = mdscr.acquire(wait_ms);

  auto now = std::chrono::steady_clock::now();
  if (new_frame) {
    ++new_frames;
    screen_wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(now - acquire_start).count();
  }
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

#if defined(_WIN32)

#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "ole32.lib")

namespace {
  struct media_foundation_t {
    media_foundation_t() {
      com = SUCCEEDED(CoInitializeEx(nullptr, COINIT_MULTITHREADED));
      started = SUCCEEDED(MFStartup(MF_VERSION, MFSTARTUP_LITE));
    }
    ~media_foundation_t() {
      if (started) {
        MFShutdown();
      }
      if (com) {
        CoUninitialize();
      }
    }
    bool com = false;
    bool started = false;
  };

  struct camera_impl_t {
    media_foundation_t media_foundation;
    IMFSourceReader* reader = nullptr;
    LONG stride = 0;
  };

  std::string camera_string(IMFActivate* device, const GUID& key) {
    WCHAR* wide = nullptr;
    UINT32 length = 0;
    if (FAILED(device->GetAllocatedString(key, &wide, &length))) {
      return {};
    }
    std::string utf8(WideCharToMultiByte(CP_UTF8, 0, wide, (int)length, nullptr, 0, nullptr, nullptr), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide, (int)length, utf8.data(), (int)utf8.size(), nullptr, nullptr);
    CoTaskMemFree(wide);
    return utf8;
  }

  UINT32 enumerate_cameras(IMFActivate*** devices) {
    IMFAttributes* attributes = nullptr;
    UINT32 count = 0;
    *devices = nullptr;
    if (SUCCEEDED(MFCreateAttributes(&attributes, 1))) {
      attributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
      if (FAILED(MFEnumDeviceSources(attributes, devices, &count))) {
        count = 0;
      }
      attributes->Release();
    }
    return count;
  }

  void release_cameras(IMFActivate** devices, UINT32 count) {
    for (UINT32 i = 0; i < count; ++i) {
      devices[i]->Release();
    }
    CoTaskMemFree(devices);
  }
}

std::vector<fan::graphics::camera_device_t> fan::graphics::get_camera_devices() {
  std::vector<camera_device_t> cameras;
  media_foundation_t media_foundation;
  if (!media_foundation.started) {
    return cameras;
  }
  IMFActivate** devices = nullptr;
  UINT32 count = enumerate_cameras(&devices);
  for (UINT32 i = 0; i < count; ++i) {
    cameras.push_back({
      camera_string(devices[i], MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK),
      camera_string(devices[i], MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME)
    });
  }
  release_cameras(devices, count);
  return cameras;
}

bool fan::graphics::camera_capture_t::open(const std::string& device_id, std::uint32_t wanted_width, std::uint32_t wanted_height, std::uint32_t wanted_fps) {
  close();
  error.clear();
  auto* c = new camera_impl_t;
  impl = c;
  auto fail = [&](const char* reason) {
    error = reason;
    close();
    return false;
  };
  if (!c->media_foundation.started) {
    return fail("media foundation is not available");
  }
  IMFActivate** devices = nullptr;
  UINT32 count = enumerate_cameras(&devices);
  IMFMediaSource* source = nullptr;
  for (UINT32 i = 0; i < count && !source; ++i) {
    if (device_id.empty() || camera_string(devices[i], MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK) == device_id) {
      devices[i]->ActivateObject(IID_PPV_ARGS(&source));
    }
  }
  release_cameras(devices, count);
  if (!source) {
    return fail("the camera is not there or is in use");
  }
  IMFAttributes* attributes = nullptr;
  if (SUCCEEDED(MFCreateAttributes(&attributes, 1))) {
    attributes->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE);
  }
  HRESULT created = MFCreateSourceReaderFromMediaSource(source, attributes, &c->reader);
  if (attributes) {
    attributes->Release();
  }
  source->Release();
  if (FAILED(created)) {
    return fail("failed to read from the camera");
  }
  const DWORD stream = (DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM;

  IMFMediaType* best = nullptr;
  std::uint64_t best_score = (std::uint64_t)-1;
  IMFMediaType* type = nullptr;
  for (DWORD i = 0; SUCCEEDED(c->reader->GetNativeMediaType(stream, i, &type)); ++i) {
    UINT32 w = 0, h = 0, rate = 0, rate_base = 1;
    MFGetAttributeSize(type, MF_MT_FRAME_SIZE, &w, &h);
    MFGetAttributeRatio(type, MF_MT_FRAME_RATE, &rate, &rate_base);
    std::int64_t fps = rate_base ? rate / rate_base : 0;
    std::uint64_t score = (std::uint64_t)std::abs((std::int64_t)w * h - (std::int64_t)wanted_width * wanted_height) * 1000 +
      (std::uint64_t)std::abs(fps - (std::int64_t)wanted_fps);
    if (score < best_score) {
      if (best) {
        best->Release();
      }
      best = type;
      best_score = score;
    }
    else {
      type->Release();
    }
  }
  if (best) {
    c->reader->SetCurrentMediaType(stream, nullptr, best);
    best->Release();
  }
  IMFMediaType* output = nullptr;
  if (FAILED(MFCreateMediaType(&output))) {
    return fail("failed to set the camera's format");
  }
  output->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
  output->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
  HRESULT set = c->reader->SetCurrentMediaType(stream, nullptr, output);
  output->Release();
  IMFMediaType* current = nullptr;
  if (FAILED(set) || FAILED(c->reader->GetCurrentMediaType(stream, &current))) {
    return fail("the camera has no format that can be used");
  }
  UINT32 w = 0, h = 0, stride = 0;
  MFGetAttributeSize(current, MF_MT_FRAME_SIZE, &w, &h);
  c->stride = SUCCEEDED(current->GetUINT32(MF_MT_DEFAULT_STRIDE, &stride)) ? (LONG)stride : (LONG)w * 4;
  current->Release();
  if (w == 0 || h == 0) {
    return fail("the camera has no format that can be used");
  }
  width = w;
  height = h;
  frame.assign((std::size_t)width * height * 4, 0);
  return true;
}

void fan::graphics::camera_capture_t::close() {
  auto* c = (camera_impl_t*)impl;
  if (!c) {
    return;
  }
  if (c->reader) {
    c->reader->Release();
  }
  delete c;
  impl = nullptr;
}

std::uint8_t* fan::graphics::camera_capture_t::read(std::uint32_t) {
  auto* c = (camera_impl_t*)impl;
  if (!c) {
    return nullptr;
  }
  DWORD flags = 0;
  LONGLONG timestamp = 0;
  IMFSample* sample = nullptr;
  HRESULT result = c->reader->ReadSample((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, nullptr, &flags, &timestamp, &sample);
  if (FAILED(result) || (flags & (MF_SOURCE_READERF_ERROR | MF_SOURCE_READERF_ENDOFSTREAM))) {
    if (sample) {
      sample->Release();
    }
    error = "the camera stopped";
    close();
    return nullptr;
  }
  if (!sample) {
    return nullptr;
  }
  IMFMediaBuffer* buffer = nullptr;
  BYTE* data = nullptr;
  DWORD length = 0;
  bool copied = false;
  if (SUCCEEDED(sample->ConvertToContiguousBuffer(&buffer))) {
    std::size_t row_size = (std::size_t)width * 4;
    std::size_t source_stride = (std::size_t)std::abs(c->stride);
    if (SUCCEEDED(buffer->Lock(&data, nullptr, &length))) {
      if (source_stride >= row_size && length >= source_stride * (height - 1) + row_size) {
        for (std::uint32_t y = 0; y < height; ++y) {
          const BYTE* row = data + source_stride * (c->stride < 0 ? height - 1 - y : y);
          std::uint8_t* out = frame.data() + row_size * y;
          std::memcpy(out, row, row_size);
          for (std::size_t x = 3; x < row_size; x += 4) {
            out[x] = 255;
          }
        }
        copied = true;
      }
      buffer->Unlock();
    }
    buffer->Release();
  }
  sample->Release();
  return copied ? frame.data() : nullptr;
}

#elif defined(__linux__)

namespace {
  struct camera_impl_t {
    struct buffer_t {
      void* data = nullptr;
      std::size_t size = 0;
    };
    int fd = -1;
    std::uint32_t pixel_format = 0;
    std::uint32_t line_size = 0;
    std::vector<buffer_t> buffers;
    AVCodecContext* decoder = nullptr;
    AVFrame* decoded = nullptr;
    AVPacket* packet = nullptr;
    SwsContext* scaler = nullptr;
    std::vector<std::uint8_t> compressed;
  };

  int camera_ioctl(int fd, unsigned long request, void* argument) {
    int result;
    do {
      result = ioctl(fd, request, argument);
    } while (result == -1 && errno == EINTR);
    return result;
  }
}

std::vector<fan::graphics::camera_device_t> fan::graphics::get_camera_devices() {
  std::vector<camera_device_t> cameras;
  for (int i = 0; i < 64; ++i) {
    std::string path = "/dev/video" + std::to_string(i);
    int fd = ::open(path.c_str(), O_RDWR | O_NONBLOCK);
    if (fd < 0) {
      continue;
    }
    v4l2_capability capability {};
    v4l2_fmtdesc format {};
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (camera_ioctl(fd, VIDIOC_QUERYCAP, &capability) == 0 &&
      ((capability.capabilities & V4L2_CAP_DEVICE_CAPS ? capability.device_caps : capability.capabilities) & V4L2_CAP_VIDEO_CAPTURE) &&
      camera_ioctl(fd, VIDIOC_ENUM_FMT, &format) == 0) {
      cameras.push_back({path, (const char*)capability.card});
    }
    ::close(fd);
  }
  return cameras;
}

bool fan::graphics::camera_capture_t::open(const std::string& device_id, std::uint32_t wanted_width, std::uint32_t wanted_height, std::uint32_t wanted_fps) {
  close();
  error.clear();
  auto fail = [&](const std::string& reason) {
    error = reason;
    close();
    return false;
  };
  std::string path = device_id;
  if (path.empty()) {
    auto cameras = get_camera_devices();
    if (cameras.empty()) {
      return fail("no camera found");
    }
    path = cameras.front().id;
  }
  auto* c = new camera_impl_t;
  impl = c;
  c->fd = ::open(path.c_str(), O_RDWR | O_NONBLOCK);
  if (c->fd < 0) {
    return fail("failed to open " + path + ": " + std::strerror(errno));
  }
  v4l2_format format {};
  for (std::uint32_t pixel_format : {(std::uint32_t)V4L2_PIX_FMT_MJPEG, (std::uint32_t)V4L2_PIX_FMT_YUYV}) {
    format = {};
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width = wanted_width;
    format.fmt.pix.height = wanted_height;
    format.fmt.pix.pixelformat = pixel_format;
    format.fmt.pix.field = V4L2_FIELD_ANY;
    if (camera_ioctl(c->fd, VIDIOC_S_FMT, &format) == 0 && format.fmt.pix.pixelformat == pixel_format) {
      c->pixel_format = pixel_format;
      break;
    }
  }
  if (c->pixel_format == 0 || format.fmt.pix.width == 0 || format.fmt.pix.height == 0) {
    return fail("the camera has no format that can be used (mjpeg or yuyv)");
  }
  width = format.fmt.pix.width;
  height = format.fmt.pix.height;
  c->line_size = format.fmt.pix.bytesperline ? format.fmt.pix.bytesperline : width * 2;

  v4l2_streamparm rate {};
  rate.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  rate.parm.capture.timeperframe.numerator = 1;
  rate.parm.capture.timeperframe.denominator = wanted_fps;
  camera_ioctl(c->fd, VIDIOC_S_PARM, &rate);

  v4l2_requestbuffers request {};
  request.count = 4;
  request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  request.memory = V4L2_MEMORY_MMAP;
  if (camera_ioctl(c->fd, VIDIOC_REQBUFS, &request) != 0 || request.count < 2) {
    return fail("the camera gave no buffers");
  }
  for (std::uint32_t i = 0; i < request.count; ++i) {
    v4l2_buffer buffer {};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.index = i;
    if (camera_ioctl(c->fd, VIDIOC_QUERYBUF, &buffer) != 0) {
      return fail("the camera gave no buffers");
    }
    void* data = mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, c->fd, buffer.m.offset);
    if (data == MAP_FAILED) {
      return fail("the camera's buffers can't be mapped");
    }
    c->buffers.push_back({data, buffer.length});
    if (camera_ioctl(c->fd, VIDIOC_QBUF, &buffer) != 0) {
      return fail("the camera gave no buffers");
    }
  }
  if (c->pixel_format == V4L2_PIX_FMT_MJPEG) {
    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_MJPEG);
    c->decoder = codec ? avcodec_alloc_context3(codec) : nullptr;
    c->decoded = av_frame_alloc();
    c->packet = av_packet_alloc();
    if (!c->decoder || !c->decoded || !c->packet || avcodec_open2(c->decoder, codec, nullptr) < 0) {
      return fail("no mjpeg decoder for the camera");
    }
  }
  int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (camera_ioctl(c->fd, VIDIOC_STREAMON, &type) != 0) {
    return fail(std::string("the camera did not start: ") + std::strerror(errno));
  }
  frame.assign((std::size_t)width * height * 4, 255);
  return true;
}

void fan::graphics::camera_capture_t::close() {
  auto* c = (camera_impl_t*)impl;
  if (!c) {
    return;
  }
  if (c->fd >= 0) {
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    camera_ioctl(c->fd, VIDIOC_STREAMOFF, &type);
  }
  for (auto& buffer : c->buffers) {
    munmap(buffer.data, buffer.size);
  }
  if (c->fd >= 0) {
    ::close(c->fd);
  }
  if (c->decoder) {
    avcodec_free_context(&c->decoder);
  }
  av_frame_free(&c->decoded);
  av_packet_free(&c->packet);
  sws_freeContext(c->scaler);
  delete c;
  impl = nullptr;
}

std::uint8_t* fan::graphics::camera_capture_t::read(std::uint32_t timeout_ms) {
  auto* c = (camera_impl_t*)impl;
  if (!c) {
    return nullptr;
  }
  pollfd waiting {c->fd, POLLIN, 0};
  int ready = poll(&waiting, 1, (int)timeout_ms);
  if (ready > 0 && (waiting.revents & (POLLERR | POLLHUP | POLLNVAL))) {
    error = "the camera stopped";
    close();
    return nullptr;
  }
  if (ready <= 0) {
    return nullptr;
  }
  v4l2_buffer newest {};
  bool have = false;
  for (;;) {
    v4l2_buffer buffer {};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    if (camera_ioctl(c->fd, VIDIOC_DQBUF, &buffer) != 0) {
      break;
    }
    if (have) {
      camera_ioctl(c->fd, VIDIOC_QBUF, &newest);
    }
    newest = buffer;
    have = true;
  }
  if (!have || newest.index >= c->buffers.size()) {
    return nullptr;
  }
  const std::uint8_t* source[4] = {(const std::uint8_t*)c->buffers[newest.index].data, nullptr, nullptr, nullptr};
  int source_stride[4] = {(int)c->line_size, 0, 0, 0};
  int source_format = AV_PIX_FMT_YUYV422;
  bool usable = !(newest.flags & V4L2_BUF_FLAG_ERROR) && newest.bytesused <= c->buffers[newest.index].size;
  if (usable && c->pixel_format == V4L2_PIX_FMT_MJPEG) {
    c->compressed.assign(source[0], source[0] + newest.bytesused);
    c->compressed.resize(newest.bytesused + AV_INPUT_BUFFER_PADDING_SIZE, 0);
    c->packet->data = c->compressed.data();
    c->packet->size = (int)newest.bytesused;
    usable = avcodec_send_packet(c->decoder, c->packet) == 0 && avcodec_receive_frame(c->decoder, c->decoded) == 0 &&
      (std::uint32_t)c->decoded->width == width && (std::uint32_t)c->decoded->height == height;
    if (usable) {
      for (int i = 0; i < 4; ++i) {
        source[i] = c->decoded->data[i];
        source_stride[i] = c->decoded->linesize[i];
      }
      source_format = c->decoded->format;
    }
  }
  else if (usable) {
    usable = newest.bytesused >= (std::size_t)c->line_size * height;
  }
  if (usable) {
    c->scaler = sws_getCachedContext(c->scaler, (int)width, (int)height, (AVPixelFormat)source_format,
      (int)width, (int)height, AV_PIX_FMT_BGRA, SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
    std::uint8_t* target[4] = {frame.data(), nullptr, nullptr, nullptr};
    int target_stride[4] = {(int)width * 4, 0, 0, 0};
    usable = c->scaler && sws_scale(c->scaler, source, source_stride, 0, (int)height, target, target_stride) > 0;
  }
  camera_ioctl(c->fd, VIDIOC_QBUF, &newest);
  return usable ? frame.data() : nullptr;
}

#else

std::vector<fan::graphics::camera_device_t> fan::graphics::get_camera_devices() {
  return {};
}

bool fan::graphics::camera_capture_t::open(const std::string&, std::uint32_t, std::uint32_t, std::uint32_t) {
  error = "cameras are not supported on this system";
  return false;
}

void fan::graphics::camera_capture_t::close() {}

std::uint8_t* fan::graphics::camera_capture_t::read(std::uint32_t) {
  return nullptr;
}

#endif

bool fan::graphics::screen_encode_t::encode_write() {
  if (!ensure_mdscr_initialized()) return false;

#if defined(_WIN32)
  bool capture_source_changed = mdscr.source_changed.exchange(false);
  if (mdscr.device_generation != encoded_device_generation_) {
    // the capture moved to another graphics card: the encoder has to open on the new device
    encoded_device_generation_ = mdscr.device_generation;
    encoder_.gpu_device_ = mdscr.device;
    if (!encoder_.open(config_, fan::vec2(mdscr.Geometry.Resolution.x, mdscr.Geometry.Resolution.y))) return false;
    {
      std::lock_guard<std::mutex> lock(mutex);
      update_flags = 0;
    }
    encoded_texture_generation_ = mdscr.texture_generation;
    capture_source_changed = true;
  }
#else
  bool capture_source_changed = false;
#endif

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
  bool force_keyframe = capture_source_changed ||
    (encode_write_flags.fetch_and((std::uint8_t)~codec_update_e::force_keyframe) & codec_update_e::force_keyframe) != 0;
  // a frame that never reaches the encoder must not use the request up
  auto keep_keyframe_request = [&] {
    if (force_keyframe) {
      encode_write_flags |= codec_update_e::force_keyframe;
    }
  };

  std::vector<libav_encoder_t::encode_result_t> results;

#if defined(_WIN32)
  if (encoder_.uses_gpu_frames()) {
    // The capture made its texture again (a game switching to fullscreen or HDR as a raid starts, the screen's mode
    // changing): the new one can have the old one's address, and the encoder kept reading through its view of the
    // gone one, a black picture until sharing was turned off and on. Its view is made again, and a keyframe sent
    if (mdscr.texture_generation != encoded_texture_generation_) {
      encoded_texture_generation_ = mdscr.texture_generation;
      encoder_.forget_input_texture();
      force_keyframe = true;
    }
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

// ---- zero copy between the NVIDIA decoder and the renderer ------------------------------------------------------
// The CUDA driver is loaded when it is there (nvcuda.dll, libcuda.so.1): an AMD or Intel computer has none and
// must still start, so nothing is linked against it. The few types are the driver API's own, by value
namespace {
  using cu_result_t = int;
  using cu_context_t = void*;
  using cu_device_t = int;
  using cu_deviceptr_t = unsigned long long;
  using cu_external_memory_t = void*;
  using cu_stream_t = void*;

  struct cu_external_memory_handle_desc_t {
    int type; // 1 opaque fd, 2 opaque win32
    union {
      int fd;
      struct {
        void* handle;
        const void* name;
      } win32;
      const void* nv_sci_buf_object;
    } handle;
    unsigned long long size;
    unsigned int flags; // 1 = dedicated
    unsigned int reserved[16];
  };
  struct cu_external_memory_buffer_desc_t {
    unsigned long long offset;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
  };
  struct cu_memcpy_2d_t {
    std::size_t src_x_in_bytes, src_y;
    int src_memory_type;
    const void* src_host;
    cu_deviceptr_t src_device;
    void* src_array;
    std::size_t src_pitch;
    std::size_t dst_x_in_bytes, dst_y;
    int dst_memory_type;
    void* dst_host;
    cu_deviceptr_t dst_device;
    void* dst_array;
    std::size_t dst_pitch;
    std::size_t width_in_bytes, height;
  };
  constexpr int cu_memorytype_device = 2;

  struct cuda_api_t {
    cu_result_t (*ctx_push_current)(cu_context_t) = nullptr;
    cu_result_t (*ctx_pop_current)(cu_context_t*) = nullptr;
    cu_result_t (*ctx_get_device)(cu_device_t*) = nullptr;
    cu_result_t (*device_get_uuid)(unsigned char*, cu_device_t) = nullptr;
    cu_result_t (*import_external_memory)(cu_external_memory_t*, const cu_external_memory_handle_desc_t*) = nullptr;
    cu_result_t (*external_memory_get_mapped_buffer)(cu_deviceptr_t*, cu_external_memory_t, const cu_external_memory_buffer_desc_t*) = nullptr;
    cu_result_t (*destroy_external_memory)(cu_external_memory_t) = nullptr;
    cu_result_t (*mem_free)(cu_deviceptr_t) = nullptr;
    cu_result_t (*memcpy_2d_async)(const cu_memcpy_2d_t*, cu_stream_t) = nullptr;
    cu_result_t (*stream_synchronize)(cu_stream_t) = nullptr;

    bool load() {
    #if defined(_WIN32)
      HMODULE library = LoadLibraryA("nvcuda.dll");
      auto find = [&](const char* name) { return library ? (void*)GetProcAddress(library, name) : nullptr; };
    #else
      void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
      auto find = [&](const char* name) { return library ? dlsym(library, name) : nullptr; };
    #endif
      if (!library) {
        return false;
      }
      ctx_push_current = (decltype(ctx_push_current))find("cuCtxPushCurrent_v2");
      ctx_pop_current = (decltype(ctx_pop_current))find("cuCtxPopCurrent_v2");
      ctx_get_device = (decltype(ctx_get_device))find("cuCtxGetDevice");
      device_get_uuid = (decltype(device_get_uuid))find("cuDeviceGetUuid");
      import_external_memory = (decltype(import_external_memory))find("cuImportExternalMemory");
      external_memory_get_mapped_buffer = (decltype(external_memory_get_mapped_buffer))find("cuExternalMemoryGetMappedBuffer");
      destroy_external_memory = (decltype(destroy_external_memory))find("cuDestroyExternalMemory");
      mem_free = (decltype(mem_free))find("cuMemFree_v2");
      memcpy_2d_async = (decltype(memcpy_2d_async))find("cuMemcpy2DAsync_v2");
      stream_synchronize = (decltype(stream_synchronize))find("cuStreamSynchronize");
      return ctx_push_current && ctx_pop_current && ctx_get_device && device_get_uuid && import_external_memory &&
        external_memory_get_mapped_buffer && destroy_external_memory && mem_free && memcpy_2d_async && stream_synchronize;
    }
  };

  // Frames go round through a few slots. Each slot is one Vulkan buffer (the Y plane, then the UV plane, both
  // tightly packed) that CUDA sees as device memory. The decoder writes a slot and the render thread copies it
  // into the picture's images, both under the mutex, so a slot is never written while it is read. Only the newest
  // few frames wait to be shown, so a slot comes round again after its frame was shown or dropped; its id tells a
  // frame that was written over anyway
  struct gpu_bridge_t {
    static constexpr std::size_t slot_count = 6;
    struct slot_t {
      fan::vulkan::context_t::external_buffer_t vk;
      cu_external_memory_t cuda_memory = nullptr;
      cu_deviceptr_t pointer = 0;
      std::uint64_t frame_id = 0;
    };

    cuda_api_t cuda;
    bool cuda_loaded = false;
    std::mutex mutex;
    std::array<slot_t, slot_count> slots;
    cu_context_t cuda_context = nullptr;
    fan::vec2ui size {0, 0};
    std::size_t next = 0;
    std::uint64_t next_id = 0;
    // a CUDA context that can't be used (another GPU than the window's, an import that failed): not tried again
    cu_context_t refused_context = nullptr;
    bool told = false;
    int copy_failures = 0;

    void release_slots(bool context_alive) {
      auto& vk = fan::graphics::get_vk_context();
      for (auto& slot : slots) {
        if (context_alive && cuda_context && slot.cuda_memory) {
          cuda.ctx_push_current(cuda_context);
          if (slot.pointer) {
            cuda.mem_free(slot.pointer);
          }
          cuda.destroy_external_memory(slot.cuda_memory);
          cu_context_t popped = nullptr;
          cuda.ctx_pop_current(&popped);
        }
        if (slot.vk.buffer != VK_NULL_HANDLE) {
          vk.external_buffer_destroy(slot.vk);
        }
        slot = {};
      }
      size = {0, 0};
    }

    // the slots for frames of this size in this CUDA context, made again when either changed
    bool ensure(cu_context_t context, const fan::vec2ui& frame_size) {
      if (context == refused_context) {
        return false;
      }
      if (context == cuda_context && frame_size == size) {
        return true;
      }
      auto& vk = fan::graphics::get_vk_context();
      // memory imported into a context belongs to it: a new context (the decoder was opened again, its old
      // context is gone with everything in it) gets new buffers
      release_slots(context == cuda_context);
      cuda_context = context;

      // the same GPU, or CUDA's pointers mean nothing to Vulkan
      cu_device_t device = 0;
      unsigned char uuid[16] {};
      cuda.ctx_push_current(context);
      bool same_gpu = cuda.ctx_get_device(&device) == 0 && cuda.device_get_uuid(uuid, device) == 0 &&
        std::memcmp(uuid, vk.device_uuid.data(), 16) == 0;
      if (!same_gpu) {
        cu_context_t popped = nullptr;
        cuda.ctx_pop_current(&popped);
        refused_context = context;
        if (!told) {
          told = true;
          fan::print("video: zero copy off, the decoding GPU is not the one drawing the window");
        }
        return false;
      }

      std::uint64_t y_bytes = (std::uint64_t)frame_size.x * frame_size.y;
      std::uint64_t uv_bytes = (std::uint64_t)frame_size.x * ((frame_size.y + 1) / 2);
      bool ok = true;
      for (auto& slot : slots) {
        if (!vk.external_buffer_create(y_bytes + uv_bytes, slot.vk)) {
          ok = false;
          break;
        }
        cu_external_memory_handle_desc_t handle_desc {};
      #if defined(_WIN32)
        handle_desc.type = 2;
        handle_desc.handle.win32.handle = (void*)slot.vk.handle;
      #else
        handle_desc.type = 1;
        handle_desc.handle.fd = (int)slot.vk.handle;
      #endif
        handle_desc.size = slot.vk.size;
        handle_desc.flags = 1;
        bool imported = cuda.import_external_memory(&slot.cuda_memory, &handle_desc) == 0;
      #if defined(_WIN32)
        // the NT handle stays ours to close, CUDA holds the memory now
        CloseHandle((HANDLE)slot.vk.handle);
      #else
        // the fd is CUDA's after an import, closed here only when the import failed
        if (!imported) {
          close((int)slot.vk.handle);
        }
      #endif
        slot.vk.handle = -1;
        if (!imported) {
          slot.cuda_memory = nullptr;
          ok = false;
          break;
        }
        cu_external_memory_buffer_desc_t buffer_desc {};
        buffer_desc.size = y_bytes + uv_bytes;
        if (cuda.external_memory_get_mapped_buffer(&slot.pointer, slot.cuda_memory, &buffer_desc) != 0) {
          slot.pointer = 0;
          ok = false;
          break;
        }
      }
      cu_context_t popped = nullptr;
      cuda.ctx_pop_current(&popped);
      if (!ok) {
        release_slots(true);
        refused_context = context;
        if (!told) {
          told = true;
          fan::print("video: zero copy off, CUDA could not use Vulkan's memory");
        }
        return false;
      }
      size = frame_size;
      return true;
    }

    // on the decoding thread: the CUDA frame into the next slot, on the GPU. The slot, or -1
    int write(cu_context_t context, const std::array<const std::uint8_t*, 2>& planes, const std::array<int, 2>& pitches,
      const fan::vec2ui& frame_size) {
      std::lock_guard<std::mutex> lock(mutex);
      if (!ensure(context, frame_size)) {
        return -1;
      }
      std::size_t index = next;
      next = (next + 1) % slot_count;
      auto& slot = slots[index];
      std::uint64_t y_bytes = (std::uint64_t)frame_size.x * frame_size.y;

      cuda.ctx_push_current(context);
      auto copy = [&](const std::uint8_t* source, int pitch, cu_deviceptr_t destination, std::size_t rows) {
        cu_memcpy_2d_t c {};
        c.src_memory_type = cu_memorytype_device;
        c.src_device = (cu_deviceptr_t)(std::uintptr_t)source;
        c.src_pitch = (std::size_t)pitch;
        c.dst_memory_type = cu_memorytype_device;
        c.dst_device = destination;
        c.dst_pitch = frame_size.x;
        c.width_in_bytes = frame_size.x;
        c.height = rows;
        return cuda.memcpy_2d_async(&c, nullptr) == 0;
      };
      bool ok = copy(planes[0], pitches[0], slot.pointer, frame_size.y) &&
        copy(planes[1], pitches[1], slot.pointer + y_bytes, (frame_size.y + 1) / 2);
      // finished before the frame is handed on: Vulkan reads it next, and the decoder reuses its surface
      ok = cuda.stream_synchronize(nullptr) == 0 && ok;
      cu_context_t popped = nullptr;
      cuda.ctx_pop_current(&popped);
      if (!ok) {
        // whatever the slot had is not a whole frame now. Made again from the start next time (the context may be
        // a new one at the old address), and after a few failures in a row not tried again for this context
        slot.frame_id = 0;
        release_slots(false);
        if (++copy_failures >= 3) {
          refused_context = context;
          if (!told) {
            told = true;
            fan::print("video: zero copy off, copying the frames failed");
          }
        }
        cuda_context = nullptr;
        return -1;
      }
      copy_failures = 0;
      slot.frame_id = ++next_id;
      return (int)index;
    }
  };
}

void fan::graphics::screen_decode_t::gpu_bridge_install() {
  if (gpu_bridge_) {
    return;
  }
  auto* bridge = new gpu_bridge_t;
  bool cuda_found = bridge->cuda.load();
  bool vulkan_ready = fan::graphics::get_vk_context().external_memory_supported;
  bridge->cuda_loaded = cuda_found && vulkan_ready;
  gpu_bridge_ = bridge;
  if (!bridge->cuda_loaded) {
    // an NVIDIA driver is there but something is missing: said once, the frames download as before
    if (cuda_found) {
      fan::print("video: zero copy off, the Vulkan driver can't share memory");
    }
    return;
  }
  fan::print("video: zero copy available (NVIDIA decoding straight into the picture)");
  decoder_.gpu_frame_hook = [this, bridge](void* context, const std::array<const std::uint8_t*, 2>& planes,
    const std::array<int, 2>& pitches, fan::vec2ui size) -> int {
    if (!zero_copy.load(std::memory_order_relaxed)) {
      zero_copy_used_.store(false, std::memory_order_relaxed);
      return -1;
    }
    int slot = bridge->write(context, planes, pitches, size);
    zero_copy_used_.store(slot >= 0, std::memory_order_relaxed);
    return slot;
  };
}

void fan::graphics::screen_decode_t::gpu_bridge_destroy() {
  auto* bridge = static_cast<gpu_bridge_t*>(gpu_bridge_);
  if (!bridge) {
    return;
  }
  decoder_.gpu_frame_hook = nullptr;
  {
    std::lock_guard<std::mutex> lock(bridge->mutex);
    // the decoder is closed: its CUDA context and what was imported into it are gone
    bridge->release_slots(false);
  }
  delete bridge;
  gpu_bridge_ = nullptr;
}

bool fan::graphics::screen_decode_t::show_gpu_frame(const decode_data_t& frame, fan::graphics::shape_t& universal_image_renderer) {
  auto* bridge = static_cast<gpu_bridge_t*>(gpu_bridge_);
  if (!bridge || frame.gpu_slot < 0 || frame.gpu_slot >= (int)gpu_bridge_t::slot_count) {
    return false;
  }
  std::lock_guard<std::mutex> lock(bridge->mutex);
  auto& slot = bridge->slots[frame.gpu_slot];
  if (slot.frame_id != frame.gpu_frame_id || slot.vk.buffer == VK_NULL_HANDLE || bridge->size != frame.image_size) {
    return false;
  }
  VkBuffer buffer = slot.vk.buffer;
  VkDeviceSize y_bytes = (VkDeviceSize)frame.image_size.x * frame.image_size.y;
  universal_image_renderer.reload_planes(fan::graphics::image_format_e::nv12, fan::vec2(frame.image_size),
    [&](fan::graphics::image_t image, std::uint32_t plane, const fan::vec2ui& plane_size, const fan::graphics::image_load_properties_t& lp) {
      fan::graphics::get_vk_context().image_reload_from_buffer(image, buffer, plane == 0 ? 0 : y_bytes, plane_size,
        fan::graphics::format_converter::image_global_to_vulkan(lp));
    });
  return true;
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

  // still on the GPU, in a slot of the zero copy bridge
  if (frame.gpu_slot >= 0) {
    ret.pixel_format = fan::graphics::image_format_e::nv12;
    ret.gpu_slot = frame.gpu_slot;
    ret.gpu_frame_id = static_cast<gpu_bridge_t*>(gpu_bridge_)->slots[frame.gpu_slot].frame_id;
    ret.stride[0] = {frame.image_size.x, 0};
    frame_size_ = frame.image_size;
    decoded_size = frame.image_size;
    return ret;
  }

  switch (frame.pixel_format) {
  case AV_PIX_FMT_YUV420P: ret.pixel_format = fan::graphics::image_format_e::yuv420p; break;
  case AV_PIX_FMT_YUV444P:
  case AV_PIX_FMT_YUVJ444P: ret.pixel_format = fan::graphics::image_format_e::yuv444p; break;

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
