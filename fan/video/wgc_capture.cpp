// Windows Graphics Capture of one window, see wgc_capture.h. A plain source file (not a module): C++/WinRT's
// headers are too big and too macro heavy for a module's global fragment
#if defined(_WIN32)

#include "wgc_capture.h"

#ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
  #define NOMINMAX
#endif
#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <dwmapi.h>
#include <Windows.Graphics.Capture.Interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>
#include <winrt/base.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Metadata.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>

#pragma comment(lib, "windowsapp.lib")
#pragma comment(lib, "dwmapi.lib")

namespace capture = winrt::Windows::Graphics::Capture;
namespace d3d = winrt::Windows::Graphics::DirectX::Direct3D11;

namespace fan::graphics::wgc {
  struct session_t {
    d3d::IDirect3DDevice device {nullptr};
    capture::GraphicsCaptureItem item {nullptr};
    capture::Direct3D11CaptureFramePool pool {nullptr};
    capture::GraphicsCaptureSession session {nullptr};
    winrt::Windows::Graphics::SizeInt32 pool_size {};
  };

  // the texture inside a frame's surface
  static winrt::com_ptr<ID3D11Texture2D> texture_of(const d3d::IDirect3DSurface& surface) {
    auto access = surface.as<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
    winrt::com_ptr<ID3D11Texture2D> texture;
    winrt::check_hresult(access->GetInterface(__uuidof(ID3D11Texture2D), texture.put_void()));
    return texture;
  }

  session_t* open(ID3D11Device* device, void* hwnd) {
    if (!device || !hwnd) {
      return nullptr;
    }
    try {
      // this thread (the capture's) in the multi threaded apartment; already in one is fine
      try {
        winrt::init_apartment(winrt::apartment_type::multi_threaded);
      }
      catch (...) {}
      if (!capture::GraphicsCaptureSession::IsSupported()) {
        return nullptr;
      }
      auto out = new session_t;
      // the WinRT device over the capture's own D3D11 device, so the frames are textures it can copy from
      winrt::com_ptr<IDXGIDevice> dxgi;
      winrt::check_hresult(device->QueryInterface(IID_PPV_ARGS(dxgi.put())));
      winrt::com_ptr<::IInspectable> inspectable;
      winrt::check_hresult(CreateDirect3D11DeviceFromDXGIDevice(dxgi.get(), inspectable.put()));
      out->device = inspectable.as<d3d::IDirect3DDevice>();

      auto interop = winrt::get_activation_factory<capture::GraphicsCaptureItem, IGraphicsCaptureItemInterop>();
      winrt::check_hresult(interop->CreateForWindow((HWND)hwnd, winrt::guid_of<capture::GraphicsCaptureItem>(),
        winrt::put_abi(out->item)));

      out->pool_size = out->item.Size();
      out->pool = capture::Direct3D11CaptureFramePool::CreateFreeThreaded(out->device,
        winrt::Windows::Graphics::DirectX::DirectXPixelFormat::B8G8R8A8UIntNormalized, 2, out->pool_size);
      out->session = out->pool.CreateCaptureSession(out->item);
      // the sharer's pointer is drawn by the app itself, as with the whole screen; no yellow frame where Windows
      // lets it go (11 and later, the property is missing on older ones)
      using metadata = winrt::Windows::Foundation::Metadata::ApiInformation;
      if (metadata::IsPropertyPresent(L"Windows.Graphics.Capture.GraphicsCaptureSession", L"IsCursorCaptureEnabled")) {
        out->session.IsCursorCaptureEnabled(false);
      }
      if (metadata::IsPropertyPresent(L"Windows.Graphics.Capture.GraphicsCaptureSession", L"IsBorderRequired")) {
        try {
          out->session.IsBorderRequired(false);
        }
        catch (...) {}
      }
      out->session.StartCapture();
      return out;
    }
    catch (...) {
      return nullptr;
    }
  }

  void close(session_t* session) {
    if (!session) {
      return;
    }
    try {
      if (session->session) {
        session->session.Close();
      }
      if (session->pool) {
        session->pool.Close();
      }
    }
    catch (...) {}
    delete session;
  }

  bool visible_origin(void* hwnd, std::int32_t& x, std::int32_t& y) {
    RECT bounds;
    if (SUCCEEDED(DwmGetWindowAttribute((HWND)hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, &bounds, sizeof(bounds)))) {
      x = bounds.left;
      y = bounds.top;
      return true;
    }
    if (GetWindowRect((HWND)hwnd, &bounds)) {
      x = bounds.left;
      y = bounds.top;
      return true;
    }
    return false;
  }

  bool take(session_t* session, ID3D11Texture2D** texture, std::uint32_t& width, std::uint32_t& height) {
    if (!session || !texture) {
      return false;
    }
    try {
      // only the newest: older ones waiting in the pool would show late
      capture::Direct3D11CaptureFrame newest {nullptr};
      while (auto frame = session->pool.TryGetNextFrame()) {
        newest = frame;
      }
      if (!newest) {
        return false;
      }
      auto size = newest.ContentSize();
      if (size.Width <= 0 || size.Height <= 0) {
        return false;
      }
      // the window got another size: the pool makes its textures again, the next frames have it
      if (size.Width != session->pool_size.Width || size.Height != session->pool_size.Height) {
        session->pool_size = size;
        session->pool.Recreate(session->device,
          winrt::Windows::Graphics::DirectX::DirectXPixelFormat::B8G8R8A8UIntNormalized, 2, size);
      }
      auto frame_texture = texture_of(newest.Surface());
      D3D11_TEXTURE2D_DESC desc;
      frame_texture->GetDesc(&desc);
      width = (std::uint32_t)std::min<std::int32_t>(size.Width, (std::int32_t)desc.Width);
      height = (std::uint32_t)std::min<std::int32_t>(size.Height, (std::int32_t)desc.Height);
      *texture = frame_texture.detach();
      return true;
    }
    catch (...) {
      return false;
    }
  }
}

#endif
