#pragma once

#include _FAN_PATH(types/types.h)
#include _FAN_PATH(window/window_input.h)

#include _FAN_PATH(types/vector.h)
#include _FAN_PATH(types/memory.h)

#include <any>

#if defined(fan_platform_windows)
#include <DXGI.h>
#include <DXGI1_2.h>
#include <D3D11.h>

#pragma comment(lib, "Dxgi.lib")
#pragma comment(lib, "D3D11.lib")

#elif defined(fan_platform_unix)

#include <X11/extensions/Xrandr.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>
#include <X11/Xatom.h>
#include <X11/keysym.h>
#include <X11/XKBlib.h>

#include <dlfcn.h>

#undef index // xos.h

#include <sys/time.h>
#include <unistd.h>

#endif


namespace fan {

  namespace sys {

    static fan::vec2i get_screen_resolution() {
      #ifdef fan_platform_windows

      return fan::vec2i(GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN));

      #elif defined(fan_platform_unix) // close

      Display* display = XOpenDisplay(0);

      if (!display) {
        fan::print("failed to open display");
      }

      int screen = DefaultScreen(display);
      fan::vec2i resolution(DisplayWidth(display, screen), DisplayHeight(display, screen));

      XCloseDisplay(display);

      return resolution;

      #endif
    }

    static void set_screen_resolution(const fan::vec2i& size)
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

    static void reset_screen_resolution() {

      #ifdef fan_platform_windows

      ChangeDisplaySettings(nullptr, CDS_RESET);

      #elif defined(fan_platform_unix)



      #endif

    }

    static uintptr_t get_screen_refresh_rate()
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

      static void open_lib_handle(const char* lib, void** handle) {
        *handle = dlopen(lib, RTLD_LAZY | RTLD_NODELETE);
        #if fan_debug >= fan_debug_low
        if (*handle == nullptr) {
          fan::throw_error(dlerror());
        }
        #endif
      }
      static void close_lib_handle(void** handle) {
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

      static void* get_lib_proc(void** handle, const char* name) {
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

      inline Display* m_display;
      static int m_screen;

      static Display* get_display()
      {
        return m_display;
      }

      #endif

      #if defined(fan_platform_windows)

      class input {
      public:

        static inline std::unordered_map<uint16_t, bool> key_down;
        static inline std::unordered_map<uint16_t, bool> reset_keys;

        static fan::vec2i get_mouse_position() {
          POINT p;
          GetCursorPos(&p);

          return fan::vec2i(p.x, p.y);
        }

        static auto get_key_state(uint16_t key) {
          return GetAsyncKeyState(fan::window_input::convert_fan_to_scancode(key));
        }

        static void set_mouse_position(const fan::vec2i& position) {
          SetCursorPos(position.x, position.y);
        }

        static void send_mouse_event(uint16_t key, fan::button_state state) {

          constexpr auto get_key = [](uint16_t key, fan::button_state state) {

            auto press = state == fan::button_state::press;

            switch (key) {
              case fan::button_left: {
                if (press) {
                  return MOUSEEVENTF_LEFTDOWN;
                }

                return MOUSEEVENTF_LEFTUP;
              }

              case fan::button_middle: {
                if (press) {
                  return MOUSEEVENTF_MIDDLEDOWN;
                }

                return MOUSEEVENTF_MIDDLEUP;
              }

              case fan::button_right: {
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

        inline static void send_keyboard_event(uint16_t key, fan::keyboard_state state) {
          INPUT input;

          input.type = INPUT_KEYBOARD;
          input.ki.wScan = 0;
          input.ki.time = 0;
          input.ki.dwExtraInfo = 0;

          input.ki.wVk = fan::window_input::convert_fan_to_scancode(key);

          input.ki.dwFlags = (state == fan::keyboard_state::press ? 0 : KEYEVENTF_KEYUP);

          if (SendInput(1, &input, sizeof(input)) != 1) {
            fan::throw_error("");
          }
        }

        static void send_string(const std::string& str, uint32_t delay_between) {
          for (int i = 0; i < str.size(); i++) {

            bool c = 0;

            // dont peek im lazy xd
            switch (str[i]) {
              case '1': {
                fan::sys::input::send_keyboard_event(fan::key_1, fan::keyboard_state::press);
                Sleep(delay_between);
                fan::sys::input::send_keyboard_event(fan::key_1, fan::keyboard_state::release);
                c = true;
                break;
              }
              case '2': {
                fan::sys::input::send_keyboard_event(fan::key_2, fan::keyboard_state::press);
                Sleep(delay_between);
                fan::sys::input::send_keyboard_event(fan::key_2, fan::keyboard_state::release);
                c = true;
                break;
              }
              case '3': {
                fan::sys::input::send_keyboard_event(fan::key_3, fan::keyboard_state::press);
                Sleep(delay_between);
                fan::sys::input::send_keyboard_event(fan::key_3, fan::keyboard_state::release);
                c = true;
                break;
              }
              case '4': {
                fan::sys::input::send_keyboard_event(fan::key_4, fan::keyboard_state::press);
                Sleep(delay_between);
                fan::sys::input::send_keyboard_event(fan::key_4, fan::keyboard_state::release);
                c = true;
                break;
              }
              case '5': {
                fan::sys::input::send_keyboard_event(fan::key_5, fan::keyboard_state::press);
                Sleep(delay_between);
                fan::sys::input::send_keyboard_event(fan::key_5, fan::keyboard_state::release);
                c = true;
                break;
              }
              case '6': {
                fan::sys::input::send_keyboard_event(fan::key_6, fan::keyboard_state::press);
                Sleep(delay_between);
                fan::sys::input::send_keyboard_event(fan::key_6, fan::keyboard_state::release);
                c = true;
                break;
              }
              case '7': {
                fan::sys::input::send_keyboard_event(fan::key_7, fan::keyboard_state::press);
                Sleep(delay_between);
                fan::sys::input::send_keyboard_event(fan::key_7, fan::keyboard_state::release);
                c = true;
                break;
              }
              case '8': {
                fan::sys::input::send_keyboard_event(fan::key_8, fan::keyboard_state::press);
                Sleep(delay_between);
                fan::sys::input::send_keyboard_event(fan::key_8, fan::keyboard_state::release);
                c = true;
                break;
              }
              case '9': {
                fan::sys::input::send_keyboard_event(fan::key_9, fan::keyboard_state::press);
                Sleep(delay_between);
                fan::sys::input::send_keyboard_event(fan::key_9, fan::keyboard_state::release);
                c = true;
                break;
              }
              case '0': {
                fan::sys::input::send_keyboard_event(fan::key_0, fan::keyboard_state::press);
                Sleep(delay_between);
                fan::sys::input::send_keyboard_event(fan::key_0, fan::keyboard_state::release);
                c = true;
                break;
              }
            }

            if (c) {
              continue;
            }

            uint32_t offset = 0;

            if (str[i] >= 97) {
              offset -= 32;
            }


            if (str[i] >= 65 && str[i] <= 90) {
              fan::sys::input::send_keyboard_event(fan::key_left_shift, fan::keyboard_state::press);
            }

            auto send_press = [](uint32_t key, uint32_t delay) {
              fan::sys::input::send_keyboard_event(key, fan::keyboard_state::press);
              Sleep(delay);
              fan::sys::input::send_keyboard_event(key, fan::keyboard_state::release);
            };

            switch (str[i] + offset) {
              /*  case '.': {
                  send_press(fan::key_period, delay_between);
                  break;
                }*/
                /* case ',': {
                   send_press(fan::key_comma, delay_between);
                   break;
                 }*/
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
                      /* case '*': {
                         fan::sys::input::send_keyboard_event(fan::key_left_shift, fan::keyboard_state::press);
                         send_press(fan::key_apostrophe, delay_between);
                         fan::sys::input::send_keyboard_event(fan::key_left_shift, fan::keyboard_state::release);
                         break;
                       }*/
                       /*case ';': {
                         fan::sys::input::send_keyboard_event(fan::key_left_shift, fan::keyboard_state::press);
                         send_press(fan::key_comma, delay_between);
                         fan::sys::input::send_keyboard_event(fan::key_left_shift, fan::keyboard_state::release);
                         break;
                       }*/
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
        static void listen(std::function<void(uint16_t key, fan::keyboard_state keyboard_state, bool action, std::any data)> input_callback_) {

          input_callback = input_callback_;

          thread_loop();

        }

        static void listen(std::function<void(const fan::vec2i& position)> mouse_move_callback_) {

          mouse_move_callback = mouse_move_callback_;

          thread_loop();
        }

      private:

        static DWORD WINAPI thread_loop() {

          HINSTANCE hInstance = GetModuleHandle(NULL);

          if (!hInstance) {
            fan::print("failed to initialize hinstance");
            exit(1);
          }

          mouse_hook = SetWindowsHookEx(WH_MOUSE_LL, (HOOKPROC)mouse_callback, hInstance, NULL);
          keyboard_hook = SetWindowsHookEx(WH_KEYBOARD_LL, (HOOKPROC)keyboard_callback, hInstance, NULL);


          MSG message;

          int x = 0;

          while ((x = GetMessage(&message, NULL, 0, 0)))
          {
            if (x == -1)
            {
              fan::print("error");
              // handle the error and possibly exit
            }
            else
            {
              TranslateMessage(&message);
              DispatchMessage(&message);
            }
          }

          UnhookWindowsHookEx(mouse_hook);
          UnhookWindowsHookEx(keyboard_hook);
        }


        static __declspec(dllexport) LRESULT CALLBACK keyboard_callback(int nCode, WPARAM wParam, LPARAM lParam)
        {
          KBDLLHOOKSTRUCT hooked_key = *((KBDLLHOOKSTRUCT*)lParam);

          auto key = fan::window_input::convert_scancode_to_fan(hooked_key.vkCode);

          auto state = (fan::keyboard_state)((nCode == HC_ACTION) && ((wParam == WM_SYSKEYDOWN) || (wParam == WM_KEYDOWN)));

          if (state == fan::keyboard_state::press && !reset_keys[key]) {
            key_down[key] = true;
          }

          input_callback(key, state, key_down[key], 0);

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
        static __declspec(dllexport) LRESULT CALLBACK mouse_callback(int nCode, WPARAM wParam, LPARAM lParam)
        {
          MSLLHOOKSTRUCT hooked_key = *((MSLLHOOKSTRUCT*)lParam);

          if (nCode != HC_ACTION) {
            return CallNextHookEx(mouse_hook, nCode, wParam, lParam);
          }

          constexpr auto get_mouse_key = [](uint16_t key) {
            switch (key) {

              case WM_LBUTTONDOWN: { return fan::input::button_left; }
              case WM_MBUTTONDOWN: { return fan::input::button_middle; }
              case WM_RBUTTONDOWN: { return fan::input::button_right; }

              case WM_LBUTTONUP: { return fan::input::button_left; }
              case WM_MBUTTONUP: { return fan::input::button_middle; }
              case WM_RBUTTONUP: { return fan::input::button_right; }

              case WM_MOUSEWHEEL: { return fan::input::mouse_scroll_up; } // ?
              case WM_MOUSEHWHEEL: { return fan::input::mouse_scroll_down; } // ?

            }
          };

          auto key = get_mouse_key(wParam);

          switch (wParam) {
            case WM_MOUSEMOVE: {

              if (mouse_move_callback) {

                mouse_move_callback(fan::vec2i(hooked_key.pt.x, hooked_key.pt.y));
              }

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
                input_callback(key, fan::keyboard_state::press, key_down[key], fan::vec2i(hooked_key.pt.x, hooked_key.pt.y));
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
                input_callback(key, fan::keyboard_state::release, key_down[key], fan::vec2i(hooked_key.pt.x, hooked_key.pt.y));
              }

              reset_keys[key] = false;

              break;
            }
          }

          return CallNextHookEx(mouse_hook, nCode, wParam, lParam);
        }

        static inline HHOOK keyboard_hook;
        static inline HHOOK mouse_hook;

        static inline std::function<void(uint16_t, fan::keyboard_state, bool action, std::any)> input_callback;
        static inline std::function<void(const fan::vec2i& position)> mouse_move_callback;

      };

      //sint32_t MD_SCR_Get_Resolution(MD_SCR_Resolution_t *Resolution){
      //  RECT desktop;
      //  const HWND hDesktop = GetDesktopWindow();
      //  GetWindowRect(hDesktop, &desktop);
      //  Resolution->x = desktop.right;
      //  Resolution->y = desktop.bottom;
      //  return 0;
      //}

      typedef struct {
        IDXGIOutputDuplication* duplication;
        ID3D11Texture2D* texture;
        ID3D11Device* device;
        ID3D11DeviceContext* context;
        D3D11_TEXTURE2D_DESC tex_desc;
        bool imDed;
        IDXGIOutput1* output1;

        //sMD_SCR_Geometry_t Geometry;
      }MD_SCR_t;
      static sint32_t MD_SCR_open(MD_SCR_t * scr) {
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

        fan::hector_t<IDXGIAdapter1*> adapters;
        adapters.open();

        IDXGIOutput* output = 0;
        fan::hector_t<IDXGIOutput*> outputs;
        outputs.open();

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
          adapters.close();
          outputs.close();
          return 1;
        }
        if (!adapters.size()) {
          adapters.close();
          outputs.close();
          return 1;
        }

        D3D_FEATURE_LEVEL feature_level;

        auto result = D3D11CreateDevice(*((IDXGIAdapter1**)adapters.ptr),
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
          adapters.close();
          outputs.close();
          return 1;
        }

        output = *((IDXGIOutput**)outputs.ptr);

        if (output->QueryInterface(__uuidof(IDXGIOutput1), (void**)&scr->output1) != S_OK) {
          adapters.close();
          outputs.close();
          return 1;
        }

        if (scr->output1->DuplicateOutput(scr->device, &scr->duplication) != S_OK) {
          adapters.close();
          outputs.close();
          return 1;
        }

        if (!scr->duplication) {
          adapters.close();
          outputs.close();
          return 1;
        }

        DXGI_OUTPUT_DESC output_desc;
        if (output->GetDesc(&output_desc) != S_OK) {
          adapters.close();
          outputs.close();
          return 1;
        }

        if (!output_desc.DesktopCoordinates.right || !output_desc.DesktopCoordinates.bottom) {
          adapters.close();
          outputs.close();
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
          adapters.close();
          outputs.close();
          return 1;
        }

        for (uintptr_t i = 0; i < adapters.size(); i++) {
          IDXGIAdapter1* ca = ((IDXGIAdapter1**)adapters.ptr)[i];
          if (!ca) {
            continue;
          }
          ca->Release();
          ca = 0;
        }

        adapters.close();

        for (uintptr_t i = 0; i < outputs.size(); i++) {
          IDXGIOutput* co = ((IDXGIOutput**)outputs.ptr)[i];
          if (!co) {
            continue;
          }
          co->Release();
          co = 0;
        }
        outputs.close();

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
      static void MD_SCR_close(MD_SCR_t * scr) {
        scr->context->Release();
        scr->device->Release();
        if (scr->texture) {
          scr->texture->Release();
        }
        if (scr->duplication) {
          scr->duplication->Release();
        }
      }

      static uint8_t* MD_SCR_read(MD_SCR_t * scr) {

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
    }
  }