#include <pch.h>
#include "system.h"

#include <fan/types/print.h>

void fan::sys::set_utf8_cout() {
   #ifdef fan_platform_windows
   SetConsoleOutputCP(CP_UTF8);
   setvbuf(stdout, nullptr, _IOFBF, 1000);
   #endif
}

void fan::sys::set_screen_resolution(const fan::vec2i& size)
{
  #ifdef fan_platform_windows
  DEVMODE screen_settings;
  memset(&screen_settings, 0, sizeof(screen_settings));
  screen_settings.dmSize = sizeof(screen_settings);
  screen_settings.dmPelsWidth = size.x;
  screen_settings.dmPelsHeight = size.y;
  screen_settings.dmBitsPerPel = 32;
  screen_settings.dmFields = DM_BITSPERPEL | DM_PELSWIDTH | DM_PELSHEIGHT;
  ChangeDisplaySettings(&screen_settings, CDS_FULLSCREEN);

  #elif defined(fan_platform_unix)

  #endif

}

void fan::sys::reset_screen_resolution() {

  #ifdef fan_platform_windows

  ChangeDisplaySettings(nullptr, CDS_RESET);

  #elif defined(fan_platform_unix)



  #endif

}

uintptr_t fan::sys::get_screen_refresh_rate()
{

  #ifdef fan_platform_windows

  DEVMODE dmode = { 0 };

  EnumDisplaySettings(nullptr, ENUM_CURRENT_SETTINGS, &dmode);

  return dmode.dmDisplayFrequency;

  #elif defined(fan_platform_unix)

  Display* display = XOpenDisplay(NULL);

  Window root = RootWindow(display, 0);

  XRRScreenConfiguration* conf = XRRGetScreenInfo(display, root);

  short refresh_rate = XRRConfigCurrentRate(conf);

  XCloseDisplay(display);

  return refresh_rate;

  #endif

}

#ifdef fan_platform_unix

void fan::sys::open_lib_handle(const char* lib, void** handle) {
  *handle = dlopen(lib, RTLD_LAZY | RTLD_NODELETE);
  #if fan_debug >= fan_debug_low
  if (*handle == nullptr) {
    fan::throw_error(dlerror());
  }
  #endif
}

void fan::sys::close_lib_handle(void** handle) {
  #if fan_debug >= fan_debug_low
  auto error =
    #endif
    dlclose(*handle);
  #if fan_debug >= fan_debug_low
  if (error != 0) {
    fan::throw_error(dlerror());
  }
  #endif
}

void* fan::sys::get_lib_proc(void** handle, const char* name) {
  dlerror();
  void* result = dlsym(*handle, name);
  #if fan_debug >= fan_debug_low
  char* error;
  if ((error = dlerror()) != NULL) {
    fan::throw_error(error);
  }
  #endif
  return result;
}

bool fan::sys::initialize_display() {
  if (fan::sys::m_display == 0) {
    fan::sys::m_display = XOpenDisplay(NULL);
    if (!fan::sys::m_display) {
      throw std::runtime_error("failed to initialize window");
    }
    fan::sys::m_screen = DefaultScreen(fan::sys::m_display);
    return 1;
  }
  return 1;
}

#endif

fan::vec2i fan::sys::get_screen_resolution() {
#ifdef fan_platform_windows


  return fan::vec2i(GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN));

#elif defined(fan_platform_unix) // close

  fan::vec2i resolution(DisplayWidth(fan::sys::m_display, m_screen), DisplayHeight(fan::sys::m_display, m_screen));

  return resolution;

#endif
}

#if defined(fan_platform_windows)

sint32_t fan::sys::MD_SCR_open(MD_SCR_t* scr) {
  scr->duplication = 0;
  scr->texture = 0;
  scr->device = 0;
  scr->context = 0;
  scr->imDed = 0;
  scr->output1 = 0;

  IDXGIFactory1* factory = 0;

  if (CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)(&factory)) != S_OK) {
    return 1;
  }

  IDXGIAdapter1* adapter = 0;

  std::vector<IDXGIAdapter1*> adapters;

  IDXGIOutput* output = 0;
  std::vector<IDXGIOutput*> outputs;

  while (factory->EnumAdapters1(adapters.size(), &adapter) != DXGI_ERROR_NOT_FOUND) {
    DXGI_ADAPTER_DESC1 desc;
    if (adapter->GetDesc1(&desc) != S_OK) {
      continue;
    }
    adapters.push_back(adapter);
    while (adapter->EnumOutputs(outputs.size(), &output) != DXGI_ERROR_NOT_FOUND) {
      DXGI_OUTPUT_DESC desc;
      if (output->GetDesc(&desc) != S_OK) {
        continue;
      }
      outputs.push_back(output);
    }
  }

  if (!outputs.size()) {
    return 1;
  }
  if (!adapters.size()) {
    return 1;
  }

  D3D_FEATURE_LEVEL feature_level;

  auto result = D3D11CreateDevice(*((IDXGIAdapter1**)adapters.data()),
    D3D_DRIVER_TYPE_UNKNOWN,
    NULL,
    NULL,
    NULL,
    0,
    D3D11_SDK_VERSION,
    &scr->device,
    &feature_level,
    &scr->context
  );

  if (result != S_OK) {
    return 1;
  }

  output = *((IDXGIOutput**)outputs.data());

  if (output->QueryInterface(__uuidof(IDXGIOutput1), (void**)&scr->output1) != S_OK) {
    return 1;
  }

  if (scr->output1->DuplicateOutput(scr->device, &scr->duplication) != S_OK) {
    return 1;
  }

  if (!scr->duplication) {
    return 1;
  }

  DXGI_OUTPUT_DESC output_desc;
  if (output->GetDesc(&output_desc) != S_OK) {
    return 1;
  }

  if (!output_desc.DesktopCoordinates.right || !output_desc.DesktopCoordinates.bottom) {
    return 1;
  }

  scr->tex_desc.Width = output_desc.DesktopCoordinates.right;
  scr->tex_desc.Height = output_desc.DesktopCoordinates.bottom;
  scr->tex_desc.MipLevels = 1;
  scr->tex_desc.ArraySize = 1;
  scr->tex_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
  scr->tex_desc.SampleDesc.Count = 1;
  scr->tex_desc.SampleDesc.Quality = 0;
  scr->tex_desc.Usage = D3D11_USAGE_STAGING;
  scr->tex_desc.BindFlags = 0;
  scr->tex_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
  scr->tex_desc.MiscFlags = 0;

  //scr->Geometry.Resolution.x = scr->tex_desc.Width;
  //scr->Geometry.Resolution.y = scr->tex_desc.Height;

  result = scr->device->CreateTexture2D(&scr->tex_desc, NULL, &scr->texture);

  if (result != S_OK) {
    return 1;
  }

  for (uintptr_t i = 0; i < adapters.size(); i++) {
    IDXGIAdapter1* ca = ((IDXGIAdapter1**)adapters.data())[i];
    if (!ca) {
      continue;
    }
    ca->Release();
    ca = 0;
  }
  for (uintptr_t i = 0; i < outputs.size(); i++) {
    IDXGIOutput* co = ((IDXGIOutput**)outputs.data())[i];
    if (!co) {
      continue;
    }
    co->Release();
    co = 0;
  }

  if (scr->output1) {
    scr->output1->Release();
    scr->output1 = 0;
  }

  if (factory) {
    factory->Release();
    factory = 0;
  }

  return 0;
}

void fan::sys::MD_SCR_close(MD_SCR_t* scr) {
  scr->context->Release();
  scr->device->Release();
  if (scr->texture) {
    scr->texture->Release();
  }
  if (scr->duplication) {
    scr->duplication->Release();
  }
}

uint8_t* fan::sys::MD_SCR_read(MD_SCR_t* scr) {

  if (scr->imDed) {
    if (MD_SCR_open(scr)) {
      return 0;
    }
    scr->imDed = false;
  }
  DXGI_OUTDUPL_DESC duplication_desc;
  if (!scr->duplication) {
    scr->imDed = true;
    return 0;
  }
  scr->duplication->GetDesc(&duplication_desc);

  DXGI_OUTDUPL_FRAME_INFO frame_info;
  IDXGIResource* desktop_resource = NULL;
  ID3D11Texture2D* tex = NULL;
  DXGI_MAPPED_RECT mapped_rect;
  auto result = scr->duplication->AcquireNextFrame(INFINITY, &frame_info, &desktop_resource);
  if (result != S_OK) {
    if (result == 0x887a0026) {
      MD_SCR_close(scr);
      scr->imDed = true;
      return 0;
      //scr->imDed = true;
      //return 0;
    }
    else if (result != 0x887a0027) {
      fan::print(std::hex, result);
    }
    return 0;
  }
  if (desktop_resource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&tex) != S_OK) {
    return 0;
  }

  scr->context->CopyResource(scr->texture, tex);

  D3D11_MAPPED_SUBRESOURCE map;
  HRESULT map_result = scr->context->Map(scr->texture, 0, D3D11_MAP_READ, 0, &map);

  //scr->Geometry.LineSize = map.RowPitch;

  scr->context->Unmap(scr->texture, 0);

  if (tex) {
    tex->Release();
    tex = NULL;
  }
  if (desktop_resource) {
    desktop_resource->Release();
    desktop_resource = NULL;
  }
  if (scr->duplication->ReleaseFrame() != S_OK) {
    return 0;
  }

  if (map_result != S_OK) {
    return 0;
  }
  return (uint8_t*)map.pData;
}

#endif

#if defined(fan_platform_windows)
inline fan::vec2i fan::sys::input::get_mouse_position() {
  POINT p;
  GetCursorPos(&p);

  return fan::vec2i(p.x, p.y);
}

inline auto fan::sys::input::get_key_state(int key) {
  return GetAsyncKeyState(fan::window_input::convert_fan_to_scancode(key));
}

inline void fan::sys::input::set_mouse_position(const fan::vec2i& position) {
  SetCursorPos(position.x, position.y);
}

inline void fan::sys::input::send_mouse_event(int key, fan::mouse_state state) {

  constexpr auto get_key = [](int key, fan::mouse_state state) {

    auto press = state == fan::mouse_state::press;

    switch (key) {
      case fan::mouse_left: {
        if (press) {
          return MOUSEEVENTF_LEFTDOWN;
        }

        return MOUSEEVENTF_LEFTUP;
      }

      case fan::mouse_middle: {
        if (press) {
          return MOUSEEVENTF_MIDDLEDOWN;
        }

        return MOUSEEVENTF_MIDDLEUP;
      }

      case fan::mouse_right: {
        if (press) {
          return MOUSEEVENTF_RIGHTDOWN;
        }

        return MOUSEEVENTF_RIGHTUP;
      }

    }
    };

  INPUT input = {};

  input.type = INPUT_MOUSE;
  input.mi.dwFlags = get_key(key, state);

  if (SendInput(1, &input, sizeof(input)) != 1) {
    fan::throw_error("");
  }
}

inline void fan::sys::input::send_keyboard_event(int key, fan::keyboard_state state) {
  INPUT input;

  input.type = INPUT_KEYBOARD;
  //input.ki.wScan = 0;
  input.ki.time = 0;
  input.ki.dwExtraInfo = 0;
  input.ki.wVk = 0;

  input.ki.wScan = fan::window_input::convert_fan_to_scancode(key);

  input.ki.dwFlags = (state == fan::keyboard_state::press ? 0 : KEYEVENTF_KEYUP) | KEYEVENTF_SCANCODE;

  if (SendInput(1, &input, sizeof(input)) != 1) {
    fan::throw_error("");
  }
}

inline void fan::sys::input::send_string(const std::string& str, uint32_t delay_between) {
  for (int i = 0; i < str.size(); i++) {

    if (str[i] >= '0' && str[i] <= '9') {
      int key = static_cast<int>(fan::key_0 + (str[i] - '0'));
      fan::sys::input::send_keyboard_event(key, fan::keyboard_state::press);
      Sleep(delay_between);
      fan::sys::input::send_keyboard_event(key, fan::keyboard_state::release);
      continue;
    }

    uint32_t offset = 0;

    if (str[i] >= 97) {
      offset -= 32;
    }
    auto send_press = [](uint32_t key, uint32_t delay) {
      fan::sys::input::send_keyboard_event(key, fan::keyboard_state::press);
      Sleep(delay);
      fan::sys::input::send_keyboard_event(key, fan::keyboard_state::release);
      };

    switch (str[i] + offset) {
      case ' ': {
        send_press(fan::key_space, delay_between);
        break;
      }
      case '\n': {
        send_press(fan::key_enter, delay_between);
        break;
      }
      case '?': {
        fan::sys::input::send_keyboard_event(fan::key_left_shift, fan::keyboard_state::press);
        send_press(fan::key_plus, delay_between);
        fan::sys::input::send_keyboard_event(fan::key_left_shift, fan::keyboard_state::release);
        break;
      }
      case '!': {
        fan::sys::input::send_keyboard_event(fan::key_left_shift, fan::keyboard_state::press);
        send_press(fan::key_1, delay_between);
        fan::sys::input::send_keyboard_event(fan::key_left_shift, fan::keyboard_state::release);
        break;
      }
      case '-': {
        send_press(fan::key_minus, delay_between);
        break;
      }
      case '_': {
        fan::sys::input::send_keyboard_event(fan::key_left_shift, fan::keyboard_state::press);
        send_press(fan::key_minus, delay_between);
        fan::sys::input::send_keyboard_event(fan::key_left_shift, fan::keyboard_state::release);
        break;
      }
      default: {
        send_press(str[i] + offset, delay_between);
      }
    }

    if (str[i] >= 65 && str[i] <= 90) {
      fan::sys::input::send_keyboard_event(fan::key_left_shift, fan::keyboard_state::release);
    }
  }
}

// creates another thread, non blocking

inline void fan::sys::input::listen_keyboard(fan::function_t<void(int key, fan::keyboard_state keyboard_state, bool action)> input_callback_) {

  input_callback = input_callback_;

  //thread_loop();

}

// BLOCKS

inline void fan::sys::input::listen_mouse(fan::function_t<void(const fan::vec2i& position)> mouse_move_callback_) {

  mouse_move_callback = mouse_move_callback_;

  // thread_loop();
}

inline DWORD __stdcall fan::sys::input::thread_loop(auto l) {

  HINSTANCE hInstance = GetModuleHandle(NULL);

  if (!hInstance) {
    fan::print("failed to initialize hinstance");
    exit(1);
  }

  mouse_hook = SetWindowsHookEx(WH_MOUSE_LL, (HOOKPROC)mouse_callback, hInstance, NULL);
  keyboard_hook = SetWindowsHookEx(WH_KEYBOARD_LL, (HOOKPROC)keyboard_callback, hInstance, NULL);


  MSG message;

  while (1)
  {
    int ret = GetMessage(&message, NULL, 0, 0);

    if (ret == -1)
    {
      fan::print("failed to get message");
    }
    else
    {
      TranslateMessage(&message);
      DispatchMessage(&message);
      l();
    }
  }

  UnhookWindowsHookEx(mouse_hook);
  UnhookWindowsHookEx(keyboard_hook);
}

inline LRESULT fan::sys::input::keyboard_callback(int nCode, WPARAM wParam, LPARAM lParam)
{
  KBDLLHOOKSTRUCT hooked_key = *((KBDLLHOOKSTRUCT*)lParam);

  auto key = fan::window_input::convert_scancode_to_fan(hooked_key.scanCode);

  auto state = (fan::keyboard_state)((nCode == HC_ACTION) && ((wParam == WM_SYSKEYDOWN) || (wParam == WM_KEYDOWN)));

  if (state == fan::keyboard_state::press && !reset_keys[key]) {
    key_down[key] = true;
  }

  input_callback(key, state, key_down[key]);

  if (state == fan::keyboard_state::release) {
    reset_keys[key] = false;
  }
  else {
    key_down[key] = false;
    reset_keys[key] = true;
  }

  return CallNextHookEx(keyboard_hook, nCode, wParam, lParam);
}

//WM_LBUTTONDOWN

inline LRESULT fan::sys::input::mouse_callback(int nCode, WPARAM wParam, LPARAM lParam)
{
  MSLLHOOKSTRUCT hooked_key = *((MSLLHOOKSTRUCT*)lParam);

  if (nCode != HC_ACTION) {
    return CallNextHookEx(mouse_hook, nCode, wParam, lParam);
  }

  constexpr auto get_mouse_key = [](int key) {
    switch (key) {

      case WM_LBUTTONDOWN: { return fan::input::mouse_left; }
      case WM_MBUTTONDOWN: { return fan::input::mouse_middle; }
      case WM_RBUTTONDOWN: { return fan::input::mouse_right; }

      case WM_LBUTTONUP: { return fan::input::mouse_left; }
      case WM_MBUTTONUP: { return fan::input::mouse_middle; }
      case WM_RBUTTONUP: { return fan::input::mouse_right; }

      case WM_MOUSEWHEEL: { return fan::input::mouse_scroll_up; } // ?
      case WM_MOUSEHWHEEL: { return fan::input::mouse_scroll_down; } // ?
      default: return fan::input::mouse_scroll_down;
    }
    };

  auto key = get_mouse_key(wParam);

  switch (wParam) {
    case WM_MOUSEMOVE: {

      mouse_move_callback(fan::vec2i(hooked_key.pt.x, hooked_key.pt.y));

      break;
    }
    case WM_LBUTTONDOWN:
    case WM_MBUTTONDOWN:
    case WM_RBUTTONDOWN:
    case WM_MOUSEWHEEL:
    case WM_MOUSEHWHEEL:
    {
      if (!reset_keys[key]) {
        key_down[key] = true;
      }

      if (input_callback) {
        input_callback(key, fan::keyboard_state::press, key_down[key]);
      }

      key_down[key] = false;
      reset_keys[key] = true;

      break;
    }
    case WM_LBUTTONUP:
    case WM_MBUTTONUP:
    case WM_RBUTTONUP:
    {
      if (input_callback) {
        input_callback(key, fan::keyboard_state::release, key_down[key]);
      }

      reset_keys[key] = false;

      break;
    }
  }

  return CallNextHookEx(mouse_hook, nCode, wParam, lParam);
}

#endif