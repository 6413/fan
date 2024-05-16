#pragma once

#include <fan/types/function.h>
#include <fan/window/window_input.h>
#include <fan/window/window_input_common.h>
#include <fan/types/vector.h>

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

    void set_utf8_cout();

    static void set_screen_resolution(const fan::vec2i& size);

    static void reset_screen_resolution();

    static uintptr_t get_screen_refresh_rate();

    #ifdef fan_platform_unix

      static void open_lib_handle(const char* lib, void** handle);
      static void close_lib_handle(void** handle);

      static void* get_lib_proc(void** handle, const char* name);

      inline bool initialize_display();

      inline int m_screen;
      inline Display* m_display = 0;

      inline bool inited = initialize_display();

      static Display* get_display()
      {
        return m_display;
      }

      #endif

      fan::vec2i get_screen_resolution();

      #if defined(fan_platform_windows)

      struct input {

        static inline std::unordered_map<uint16_t, bool> key_down;
        static inline std::unordered_map<uint16_t, bool> reset_keys;

        fan::vec2i get_mouse_position();

        auto get_key_state(int key);

        void set_mouse_position(const fan::vec2i& position);

        void send_mouse_event(int key, fan::mouse_state state);

        static void send_keyboard_event(int key, fan::keyboard_state state);

        void send_string(const std::string & str, uint32_t delay_between);

        // creates another thread, non blocking
        void listen_keyboard(fan::function_t<void(int key, fan::keyboard_state keyboard_state, bool action)> input_callback_);

        // BLOCKS
        void listen_mouse(fan::function_t<void(const fan::vec2i& position)> mouse_move_callback_);

      //private:

        static DWORD WINAPI thread_loop(auto l);


        static __declspec(dllexport) LRESULT CALLBACK keyboard_callback(int nCode, WPARAM wParam, LPARAM lParam);
        //WM_LBUTTONDOWN
        static __declspec(dllexport) LRESULT CALLBACK mouse_callback(int nCode, WPARAM wParam, LPARAM lParam);

        static inline HHOOK keyboard_hook;
        static inline HHOOK mouse_hook;

        static inline fan::function_t<void(uint16_t, fan::keyboard_state, bool action)> input_callback = [](uint16_t, fan::keyboard_state, bool action) {};
        static inline fan::function_t<void(const fan::vec2i& position)> mouse_move_callback = [](const fan::vec2i& position) {};

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
      static sint32_t MD_SCR_open(MD_SCR_t * scr);
      static void MD_SCR_close(MD_SCR_t * scr);

      static uint8_t* MD_SCR_read(MD_SCR_t * scr);
      #endif
    }
  }