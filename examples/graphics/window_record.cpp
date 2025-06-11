#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winuser.h>
#undef min
#undef max
#include <future>

#include <fan/graphics/third_party/direct3d11.interop.h>
#include <fan/graphics/third_party/capture.interop.h>
#include <fan/graphics/third_party/d3dHelpers.h>

#include <dwmapi.h>

#pragma comment(lib, "windowsapp.lib")
#pragma comment(lib, "Dwmapi.lib")

struct Window
{
public:
  Window(nullptr_t) {}
  Window(HWND hwnd, std::string const& title, std::string& className)
  {
    m_hwnd = hwnd;
    m_title = title;
    m_className = className;
  }

  HWND Hwnd() const noexcept { return m_hwnd; }
  std::string Title() const noexcept { return m_title; }
  std::string ClassName() const noexcept { return m_className; }

private:
  HWND m_hwnd;
  std::string m_title;
  std::string m_className;
};

std::string GetClassName(HWND hwnd)
{
  std::array<CHAR, 1024> className;
  ::GetClassName(hwnd, className.data(), (int)className.size());

  std::string title(className.data());
  return title;
}

std::string GetWindowText(HWND hwnd)
{
  std::array<CHAR, 1024> windowText;

  ::GetWindowText(hwnd, windowText.data(), (int)windowText.size());

  std::string title(windowText.data());
  return title;
}

static bool IsAltTabWindow(Window const& window)
{
  HWND hwnd = window.Hwnd();
  HWND shellWindow = GetShellWindow();

  auto title = window.Title();
  auto className = window.ClassName();

  if (hwnd == shellWindow)
  {
    return false;
  }

  if (title.length() == 0)
  {
    return false;
  }

  if (!IsWindowVisible(hwnd))
  {
    return false;
  }

  if (GetAncestor(hwnd, GA_ROOT) != hwnd)
  {
    return false;
  }

  LONG style = GetWindowLong(hwnd, GWL_STYLE);
  if (!((style & WS_DISABLED) != WS_DISABLED))
  {
    return false;
  }

  DWORD cloaked = FALSE;
  HRESULT hrTemp = DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, &cloaked, sizeof(cloaked));
  if (SUCCEEDED(hrTemp) &&
    cloaked == DWM_CLOAKED_SHELL)
  {
    return false;
  }

  return true;
}

BOOL CALLBACK EnumWindowsProc(HWND hwnd, LPARAM lParam)
{
  auto class_name = GetClassName(hwnd);
  auto title = GetWindowText(hwnd);

  auto window = Window(hwnd, title, class_name);

  if (!IsAltTabWindow(window))
  {
    return TRUE;
  }

  std::vector<Window>& windows = *reinterpret_cast<std::vector<Window>*>(lParam);
  windows.push_back(window);

  return TRUE;
}



//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH 
// THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
//*********************************************************

struct SimpleCapture {

  SimpleCapture(
    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice const& device,
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem const& item);
  ~SimpleCapture() { Close(); }

  void Close();

  void StartCapture();

  winrt::Windows::UI::Composition::ICompositionSurface CreateSurface(
    winrt::Windows::UI::Composition::Compositor const& compositor);

  void OnFrameArrived(
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool const& sender,
    winrt::Windows::Foundation::IInspectable const& args);

  void CheckClosed()
  {
    if (m_closed.load() == true)
    {
      throw winrt::hresult_error(RO_E_CLOSED);
    }
  }

private:
  winrt::Windows::Graphics::Capture::GraphicsCaptureItem m_item{ nullptr };
  winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool m_framePool{ nullptr };
  winrt::Windows::Graphics::Capture::GraphicsCaptureSession m_session{ nullptr };
  winrt::Windows::Graphics::SizeInt32 m_lastSize;

  winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice m_device{ nullptr };
  winrt::com_ptr<IDXGISwapChain1> m_swapChain{ nullptr };
  winrt::com_ptr<ID3D11DeviceContext> m_d3dContext{ nullptr };

  std::atomic<bool> m_closed = false;
  winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool::FrameArrived_revoker m_frameArrived;
};

using namespace winrt;
using namespace Windows;
using namespace Windows::Foundation;
using namespace Windows::System;
using namespace Windows::Graphics;
using namespace Windows::Graphics::Capture;
using namespace Windows::Graphics::DirectX;
using namespace Windows::Graphics::DirectX::Direct3D11;
using namespace Windows::Foundation::Numerics;
using namespace Windows::UI;
using namespace Windows::UI::Composition;

SimpleCapture::SimpleCapture(
  IDirect3DDevice const& device,
  GraphicsCaptureItem const& item)
{
  m_item = item;
  m_device = device;

  // Set up 
  auto d3dDevice = GetDXGIInterfaceFromObject<ID3D11Device>(m_device);
  d3dDevice->GetImmediateContext(m_d3dContext.put());

  auto size = m_item.Size();

  m_swapChain = CreateDXGISwapChain(
    d3dDevice,
    static_cast<uint32_t>(size.Width),
    static_cast<uint32_t>(size.Height),
    static_cast<DXGI_FORMAT>(DirectXPixelFormat::B8G8R8A8UIntNormalized),
    2);

  // Create framepool, define pixel format (DXGI_FORMAT_B8G8R8A8_UNORM), and frame size. 
  m_framePool = Direct3D11CaptureFramePool::Create(
    m_device,
    DirectXPixelFormat::B8G8R8A8UIntNormalized,
    2,
    size);
  m_session = m_framePool.CreateCaptureSession(m_item);
  m_lastSize = size;
  m_frameArrived = m_framePool.FrameArrived(auto_revoke, { this, &SimpleCapture::OnFrameArrived });
}

// Start sending capture frames
void SimpleCapture::StartCapture()
{
  CheckClosed();
  m_session.StartCapture();
}

ICompositionSurface SimpleCapture::CreateSurface(
  Compositor const& compositor)
{
  CheckClosed();
  return CreateCompositionSurfaceForSwapChain(compositor, m_swapChain.get());
}

// Process captured frames
void SimpleCapture::Close()
{
  auto expected = false;
  if (m_closed.compare_exchange_strong(expected, true))
  {
    m_frameArrived.revoke();
    m_framePool.Close();
    m_session.Close();

    m_swapChain = nullptr;
    m_framePool = nullptr;
    m_session = nullptr;
    m_item = nullptr;
  }
}

loco_t::image_t window_image;

#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>

// Task queue to hold frame processing tasks
class TaskQueue {
public:
  void Push(std::function<void()> task) {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_tasks.push(std::move(task));
    m_cv.notify_one();
  }

  std::function<void()> Pop() {
    std::unique_lock<std::mutex> lock(m_mutex);
    while (m_tasks.empty()) {
      m_cv.wait(lock);
    }
    auto task = std::move(m_tasks.front());
    m_tasks.pop();
    return task;
  }

  bool Empty() {
    std::unique_lock<std::mutex> lock(m_mutex);
    return m_tasks.empty();
  }

private:
  std::queue<std::function<void()>> m_tasks;
  std::mutex m_mutex;
  std::condition_variable m_cv;
};

TaskQueue g_taskQueue;


void SimpleCapture::OnFrameArrived(
  Direct3D11CaptureFramePool const& sender,
  winrt::Windows::Foundation::IInspectable const&)
{
  g_taskQueue.Push([this, sender]() {
    auto newSize = false;

    {
      auto frame = sender.TryGetNextFrame();
      auto frameContentSize = frame.ContentSize();

      if (frameContentSize.Width != m_lastSize.Width ||
        frameContentSize.Height != m_lastSize.Height)
      {
        newSize = true;
        m_lastSize = frameContentSize;
        m_swapChain->ResizeBuffers(
          2,
          static_cast<uint32_t>(m_lastSize.Width),
          static_cast<uint32_t>(m_lastSize.Height),
          static_cast<DXGI_FORMAT>(DirectXPixelFormat::B8G8R8A8UIntNormalized),
          0);
      }

      {
        auto frameSurface = GetDXGIInterfaceFromObject<ID3D11Texture2D>(frame.Surface());

        D3D11_TEXTURE2D_DESC desc;
        frameSurface->GetDesc(&desc);

        D3D11_TEXTURE2D_DESC stagingDesc = desc;
        stagingDesc.Usage = D3D11_USAGE_STAGING;
        stagingDesc.BindFlags = 0;
        stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        stagingDesc.MiscFlags = 0;

        auto d3dDevice = GetDXGIInterfaceFromObject<ID3D11Device>(m_device);

        com_ptr<ID3D11Texture2D> stagingTexture;
        check_hresult(d3dDevice->CreateTexture2D(&stagingDesc, nullptr, stagingTexture.put()));

        m_d3dContext->CopyResource(stagingTexture.get(), frameSurface.get());

        D3D11_MAPPED_SUBRESOURCE mappedResource;
        check_hresult(m_d3dContext->Map(stagingTexture.get(), 0, D3D11_MAP_READ, 0, &mappedResource));

        uint8_t* data = reinterpret_cast<uint8_t*>(mappedResource.pData);
        uint32_t rowPitch = mappedResource.RowPitch;

        fan::image::info_t ii;
        ii.size = fan::vec2(rowPitch / 4, desc.Height);
        ii.data = data;

        loco_t::image_load_properties_t lp;
        lp.format = fan::graphics::image_format::b8g8r8a8_unorm;
        lp.internal_format = GL_RGBA;

        if (window_image.iic() == false) {
          gloco->image_unload(window_image);
        }

        window_image = gloco->image_load(ii, lp);

        m_d3dContext->Unmap(stagingTexture.get(), 0);
      }
    }

    DXGI_PRESENT_PARAMETERS presentParameters = { 0 };
    m_swapChain->Present1(1, 0, &presentParameters);

    if (newSize) {
      m_framePool.Recreate(
        m_device,
        DirectXPixelFormat::B8G8R8A8UIntNormalized,
        2,
        m_lastSize);
    }
    });
}


//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH 
// THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
//*********************************************************


class SimpleCapture;

class App
{
public:
  App() {}
  ~App() {}

  void Initialize(
    winrt::Windows::UI::Composition::ContainerVisual const& root);

  void StartCapture(HWND hwnd);

private:
  winrt::Windows::UI::Composition::Compositor m_compositor{ nullptr };
  winrt::Windows::UI::Composition::ContainerVisual m_root{ nullptr };
  winrt::Windows::UI::Composition::SpriteVisual m_content{ nullptr };
  winrt::Windows::UI::Composition::CompositionSurfaceBrush m_brush{ nullptr };

  winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice m_device{ nullptr };
  std::unique_ptr<SimpleCapture> m_capture{ nullptr };
};

using namespace winrt;
using namespace Windows::System;
using namespace Windows::Foundation;
using namespace Windows::UI;
using namespace Windows::UI::Composition;
using namespace Windows::Graphics::Capture;

void App::Initialize(
  ContainerVisual const& root)
{
  auto queue = DispatcherQueue::GetForCurrentThread();

  m_compositor = root.Compositor();
  m_root = m_compositor.CreateContainerVisual();
  m_content = m_compositor.CreateSpriteVisual();
  m_brush = m_compositor.CreateSurfaceBrush();

  m_root.RelativeSizeAdjustment({ 1, 1 });
  root.Children().InsertAtTop(m_root);

  m_content.AnchorPoint({ 0.5f, 0.5f });
  m_content.RelativeOffsetAdjustment({ 0.5f, 0.5f, 0 });
  m_content.RelativeSizeAdjustment({ 1, 1 });
  m_content.Size({ -80, -80 });
  m_content.Brush(m_brush);
  m_brush.HorizontalAlignmentRatio(0.5f);
  m_brush.VerticalAlignmentRatio(0.5f);
  m_brush.Stretch(CompositionStretch::Uniform);
  auto shadow = m_compositor.CreateDropShadow();
  shadow.Mask(m_brush);
  m_content.Shadow(shadow);
  m_root.Children().InsertAtTop(m_content);

  auto d3dDevice = CreateD3DDevice();
  auto dxgiDevice = d3dDevice.as<IDXGIDevice>();
  m_device = CreateDirect3DDevice(dxgiDevice.get());
}

void App::StartCapture(HWND hwnd)
{
  if (m_capture)
  {
    m_capture->Close();
    m_capture = nullptr;
  }

  auto item = CreateCaptureItemForWindow(hwnd);

  m_capture = std::make_unique<SimpleCapture>(m_device, item);

  //auto surface = m_capture->CreateSurface(m_compositor);
  //m_brush.Surface(surface);

  m_capture->StartCapture();
}
#include <Unknwn.h>
#include <inspectable.h>

// WinRT
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.System.h>
#include <winrt/Windows.UI.h>
#include <winrt/Windows.UI.Composition.h>
#include <winrt/Windows.UI.Composition.Desktop.h>
#include <winrt/Windows.UI.Popups.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3d11.h>

#include <windows.ui.composition.interop.h>
#include <DispatcherQueue.h>

// STL
#include <atomic>
#include <memory>

// D3D
#include <d3d11_4.h>
#include <dxgi1_6.h>
#include <d2d1_3.h>
#include <wincodec.h>


auto CreateDispatcherQueueController()
{
  namespace abi = ABI::Windows::System;

  DispatcherQueueOptions options
  {
      sizeof(DispatcherQueueOptions),
      DQTYPE_THREAD_CURRENT,
      DQTAT_COM_STA
  };

  Windows::System::DispatcherQueueController controller{ nullptr };
  check_hresult(CreateDispatcherQueueController(options, reinterpret_cast<abi::IDispatcherQueueController**>(put_abi(controller))));
  return controller;
}

winrt::Windows::UI::Composition::Desktop::DesktopWindowTarget CreateDesktopWindowTarget(Compositor const& compositor, HWND window)
{
  namespace abi = ABI::Windows::UI::Composition::Desktop;

  auto interop = compositor.as<abi::ICompositorDesktopInterop>();
  winrt::Windows::UI::Composition::Desktop::DesktopWindowTarget target{ nullptr };
  check_hresult(interop->CreateDesktopWindowTarget(window, true, reinterpret_cast<abi::IDesktopWindowTarget**>(put_abi(target))));
  return target;
}

void ProcessTasks() {
  while (!g_taskQueue.Empty()) {
    auto task = g_taskQueue.Pop();
    task();
  }
}

int main() {

  init_apartment(apartment_type::single_threaded);

  loco_t loco;

  loco.camera_set_ortho(
    loco.orthographic_camera.camera,
    fan::vec2(-1, 1),
    fan::vec2(-1, 1)
  );


  std::vector<Window> windows;
  windows.push_back(Window(nullptr));
  EnumWindows(EnumWindowsProc, reinterpret_cast<LPARAM>(&windows));

  App g_app;

  auto controller = CreateDispatcherQueueController();

  window_image = gloco->default_texture;

  fan::graphics::sprite_t window_sprite{ {
    .position = fan::vec3(0, 0, 0),
    .size = 1,
    .rotation_point = fan::vec2(100, 0),
    .image = window_image,
  } };


  loco.input_action.add_keycombo({ fan::key_left_alt, fan::key_5 }, "toggle_overlay");

  loco.shader_set_vertex(
    loco.gl.m_fbo_final_shader,
    loco.shader_get(loco.gl.m_fbo_final_shader).gl->svertex
  );

  loco.shader_set_fragment(
    loco.gl.m_fbo_final_shader,
    R"(#version 330

in vec2 texture_coordinate;

layout (location = 0) out vec4 o_attachment0;

uniform sampler2D _t00; // HDR color texture
uniform sampler2D _t01; // Bloom texture
uniform sampler2D _t02;


uniform float shadows = 3;
uniform float highlights = 1.5;

uniform float brightness = 0.1;
uniform float contrast = 0.7;
uniform float gamma = .2;

uniform vec3 color_override;

void main() {
    vec3 hdrColor = texture(_t00, texture_coordinate).rgb;

    // Calculate luminance
    float luminance = 0.2126 * hdrColor.r + 0.7152 * hdrColor.g + 0.0722 * hdrColor.b;

    // Tone mapping adjustments (shadows, highlights)
    float shadow = clamp((pow(luminance, 1.0 / shadows) + (-0.76) * pow(luminance, 2.0 / shadows)) - luminance, 0.0, 1.0);
    float highlight = clamp((1.0 - pow(1.0 - luminance, highlights)) - luminance, -1.0, 0.0);

    vec3 result = vec3(0.0) + ((luminance + shadow + highlight) - 0.0) * ((hdrColor.rgb - vec3(0.0)) / (luminance - 0.0));

    result += brightness;

    result = clamp(result, 0.0, 1.0);

    result = (result - 0.5) / (1.0 - contrast) + 0.5;

    result = pow(result, vec3(1.0 / gamma));

    o_attachment0 = vec4(result * color_override, 1.0);
})"
);

  loco.shader_compile(loco.gl.m_fbo_final_shader);


  // override console commands


  loco.console.commands.add("set_brightness", [](const fan::commands_t::arg_t& args) {
    if (args.size() != 1) {
      gloco->console.commands.print_invalid_arg_count();
      return;
    }
    gloco->shader_set_value(gloco->gl.m_fbo_final_shader, "brightness", std::stof(args[0]));
    }).description = "sets brightness for postprocess shader";

  loco.console.commands.add("set_contrast", [](const fan::commands_t::arg_t& args) {
    if (args.size() != 1) {
      gloco->console.commands.print_invalid_arg_count();
      return;
    }
    gloco->shader_set_value(gloco->gl.m_fbo_final_shader, "contrast", std::stof(args[0]));
    }).description = "sets contrast for postprocess shader";

  loco.console.commands.add("set_gamma", [](const fan::commands_t::arg_t& args) {
    if (args.size() != 1) {
      gloco->console.commands.print_invalid_arg_count();
      return;
    }
    gloco->shader_set_value(gloco->gl.m_fbo_final_shader, "gamma", std::stof(args[0]));
    }).description = "sets gamma for postprocess shader";

  loco.console.commands.add("set_color", [](const fan::commands_t::arg_t& args) {
    if (args.size() != 3) {
      gloco->console.commands.print_invalid_arg_count();
      return;
    }
    gloco->shader_set_value(gloco->gl.m_fbo_final_shader, "color_override",
      fan::vec3(
        std::stof(args[0]),
        std::stof(args[1]),
        std::stof(args[2])
      ));
    }).description = "sets color for postprocess shader";

  loco.console.commands.add("set_shadows", [](const fan::commands_t::arg_t& args) {
    if (args.size() != 1) {
      gloco->console.commands.print_invalid_arg_count();
      return;
    }
    gloco->shader_set_value(gloco->gl.m_fbo_final_shader, "shadows", std::stof(args[0]));
    }).description = "sets shadows for postprocess shader";

  loco.console.commands.add("set_highlights", [](const fan::commands_t::arg_t& args) {
    if (args.size() != 1) {
      gloco->console.commands.print_invalid_arg_count();
      return;
    }
    gloco->shader_set_value(gloco->gl.m_fbo_final_shader, "highlights", std::stof(args[0]));
    }).description = "sets highlights for postprocess shader";


  loco.console.commands.call("set_color 1 1 1");


  auto duck = loco.image_load("images/duck.webp");

  fan::graphics::sprite_t s{ {
    .position = fan::vec3(0, 0, 1),
    .size = fan::vec2(0.1, 0.1),
    .image = duck
  } };

  f32_t angle = 0;

  loco.loop([&] {

    s.set_angle(fan::vec3(0, 0, angle));
    angle += loco.delta_time * 4;



    if (loco.input_action.is_active("toggle_overlay")) {


      static bool decorated = true;
      decorated = !decorated;
      decorated ? loco.window.set_windowed() : loco.window.set_borderless();
      glfwSetWindowAttrib(gloco->window.glfw_window, GLFW_DECORATED, decorated);

      static bool resizeable = true;
      resizeable = !resizeable;
      glfwSetWindowAttrib(gloco->window.glfw_window, GLFW_RESIZABLE, resizeable);

      static bool floating = false;
      floating = !floating;
      glfwSetWindowAttrib(gloco->window.glfw_window, GLFW_FLOATING, floating);

      static bool focus_on_show = true;
      focus_on_show = !focus_on_show;
      glfwSetWindowAttrib(gloco->window.glfw_window, GLFW_FOCUS_ON_SHOW, focus_on_show);


      static bool passthrough = false;
      passthrough = !passthrough;
      glfwSetWindowAttrib(gloco->window.glfw_window, GLFW_MOUSE_PASSTHROUGH, passthrough);

      static bool transparent = false;
      transparent = !transparent;
      glfwSetWindowAttrib(gloco->window.glfw_window, GLFW_TRANSPARENT_FRAMEBUFFER, transparent);
    }

    ImGui::Begin("", 0, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoResize | ImGuiDockNodeFlags_NoDockingSplit | ImGuiWindowFlags_NoTitleBar);

    static int current_hwnd = 0;
    if (ImGui::BeginCombo("list of hwnds", windows[current_hwnd].Title().c_str())) {
      for (int i = 0; i < windows.size(); i++) {
        auto title = windows[i].Title();
        if (ImGui::Selectable(title.c_str(), current_hwnd == i)) {
          current_hwnd = i;
          auto compositor = Compositor();
          //auto target = CreateDesktopWindowTarget(compositor, windows[current_hwnd].Hwnd());
          auto root = compositor.CreateContainerVisual();
          root.RelativeSizeAdjustment({ 1.0f, 1.0f });
          //target.Root(root);

          g_app.Initialize(root);
          g_app.StartCapture(windows[current_hwnd].Hwnd());
        }
      }
      ImGui::EndCombo();
    }

    window_sprite.set_image(window_image);

    //ImGui::Image(window_image, ImGui::GetWindowSize() / 1.25);


    ImGui::End();
    ProcessTasks();
    });

  return 0;
}