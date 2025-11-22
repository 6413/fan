module;

#include <fan/utility.h>

#include <fan/types/bll_raii.h>

#if defined(fan_vulkan)
#include <vulkan/vulkan.h>
#endif
#if defined(fan_platform_windows)
  #define WIN32_LEAN_AND_MEAN
  #define NOMINMAX
  #include <Windows.h>
  #define GLFW_EXPOSE_NATIVE_WIN32
  #define GLFW_EXPOSE_NATIVE_WGL
  #define GLFW_NATIVE_INCLUDE_NONE
#endif
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <functional>
#include <cstdlib>

export module fan.window;

import fan.print;
export import fan.graphics.image_load;
export import fan.window.input_common;
export import fan.window.input_action;

export import fan.types.vector;

export namespace fan {

  struct init_manager_t {
    static bool& initialized();

    static void initialize();
    static void uninitialize();

    struct cleaner_t {
      cleaner_t();
      ~cleaner_t();
    };

    static cleaner_t& cleaner();
  };
  extern fan::init_manager_t::cleaner_t& _cleaner;
}

namespace fan {
  namespace window {
    void mouse_button_callback(GLFWwindow* wnd, int button, int action, int mods);
    void keyboard_keys_callback(GLFWwindow* wnd, int key, int scancode, int action, int mods);
    void text_callback(GLFWwindow* wnd, unsigned int codepoint);
    void move_callback(GLFWwindow* wnd, int x, int y);
    void resize_callback(GLFWwindow* wnd, int width, int height);
    void close_callback(GLFWwindow* wnd);
    void mouse_position_callback(GLFWwindow* wnd, double xpos, double ypos);
    void scroll_callback(GLFWwindow* wnd, double xoffset, double yoffset);
    void window_focus_callback(GLFWwindow* wnd, int focused);
  }
}

export namespace fan {
  struct window_t {

    enum class mode {
      not_set,
      windowed,
      borderless,
      windowed_fullscreen,
      full_screen,
      count
    };

    static constexpr const char* default_window_name = "window";
    static constexpr fan::vec2i default_window_size = fan::vec2i(800, 600);
    static constexpr mode default_size_mode = mode::windowed;

    static constexpr fan::vec2i resolutions[] = {
      {800, 600},
      {1024, 768},
      {1280, 720},
      {1280, 800},
      {1280, 900},
      {1280, 1024},
      {1360, 768},
      {1440, 900},
      {1600, 900},
      {1600, 1024},
      {1680, 1050},
      {1920, 1080},
      {2560, 1440}
    };
    static constexpr const char* resolution_labels[] = {
      "800x600",
      "1024x768",
      "1280x720",
      "1280x800",
      "1280x900",
      "1280x1024",
      "1360x768",
      "1440x900",
      "1600x900",
      "1600x1024",
      "1680x1050",
      "1920x1080",
      "2560x1440"
    };
    int current_resolution = 8;

    struct flags {
      static constexpr std::uint64_t no_mouse = 1 << 0;
      static constexpr std::uint64_t no_resize = 1 << 1;
      static constexpr std::uint64_t mode = 1 << 2;
      static constexpr std::uint64_t borderless = 1 << 3;
      static constexpr std::uint64_t full_screen = 1 << 4;
      static constexpr std::uint64_t undecorated = 1 << 5;
      static constexpr std::uint64_t transparent = 1 << 6;
      static constexpr std::uint64_t hidden = 1 << 7;
      static constexpr std::uint64_t topmost = 1 << 8;
      static constexpr std::uint64_t click_through = 1 << 9;
    };

    struct renderer_t {
      static constexpr uint8_t opengl = 0;
      static constexpr uint8_t vulkan = 1;
    };

    struct buttons_data_t {
      fan::window_t* window;
      uint16_t button;
      fan::vec2 position;
      int state;
    };
    using buttons_cb_t = std::function<void(const buttons_data_t&)>;

    struct keys_data_t {
      fan::window_t* window;
      int key;
      fan::keyboard_state_t state;
      uint16_t scancode;
    };
    using keys_cb_t = std::function<void(const keys_data_t&)>;

    struct key_data_t {
      fan::window_t* window;
    };
    using key_cb_t = std::function<void(const key_data_t&)>;

    struct text_data_t {
      fan::window_t* window;
      uint32_t character;
      fan::keyboard_state_t state;
    };
    using text_cb_t = std::function<void(const text_data_t&)>;

    struct mouse_move_data_t {
      fan::window_t* window;
      fan::vec2d position;
    };
    using mouse_move_cb_t = std::function<void(const mouse_move_data_t&)>;

    struct mouse_motion_data_t {
      fan::window_t* window;
      fan::vec2d motion;
    };
    using mouse_motion_cb_t = std::function<void(const mouse_motion_data_t&)>;

    struct close_data_t {
      fan::window_t* window;
    };
    using close_cb_t = std::function<void(const close_data_t&)>;

    struct resize_data_t {
      fan::window_t* window;
      fan::vec2i size;
    };
    using resize_cb_t = std::function<void(const resize_data_t&)>;

    struct move_data_t {
      fan::window_t* window;
    };
    using move_cb_t = std::function<void(const move_data_t&)>;

    #define BLL_set_prefix buttons_callback
    #define BLL_set_NodeDataType buttons_cb_t
    #include "cb_list_builder_settings.h"
    #include <BLL/BLL.h>

    #define BLL_set_prefix keys_callback
    #define BLL_set_NodeDataType keys_cb_t
    #include "cb_list_builder_settings.h"
    #include <BLL/BLL.h>

    #define BLL_set_prefix key_callback
    #define BLL_set_NodeDataType key_cb_t
    #include "cb_list_builder_settings.h"
    #include <BLL/BLL.h>

    #define BLL_set_prefix text_callback
    #define BLL_set_NodeDataType text_cb_t
    #include "cb_list_builder_settings.h"
    #include <BLL/BLL.h>

    #define BLL_set_prefix move_callback
    #define BLL_set_NodeDataType move_cb_t
    #include "cb_list_builder_settings.h"
    #include <BLL/BLL.h>

    #define BLL_set_prefix resize_callback
    #define BLL_set_NodeDataType resize_cb_t
    #include "cb_list_builder_settings.h"
    #include <BLL/BLL.h>

    #define BLL_set_prefix close_callback
    #define BLL_set_NodeDataType close_cb_t
    #include "cb_list_builder_settings.h"
    #include <BLL/BLL.h>

    #define BLL_set_prefix mouse_position_callback
    #define BLL_set_NodeDataType mouse_move_cb_t
    #include "cb_list_builder_settings.h"
    #include <BLL/BLL.h>

    #define BLL_set_prefix mouse_motion_callback
    #define BLL_set_NodeDataType mouse_motion_cb_t
    #include "cb_list_builder_settings.h"
    #include <BLL/BLL.h>

    #define BLL_set_prefix mouse_down_callbacks
    #define BLL_set_NodeDataType buttons_cb_t
    #include "cb_list_builder_settings.h"
    #include <BLL/BLL.h>

    #define BLL_set_prefix key_down_callbacks
    #define BLL_set_NodeDataType keys_cb_t
    #include "cb_list_builder_settings.h"
    #include <BLL/BLL.h>

    #define FAN_DEFINE_CB_RAII(NAME, STORAGE, NODE_REF, PARAM_TYPE)                         \
      using NAME##_handle_t = bll_nr_t<NODE_REF, window_t, const PARAM_TYPE&>;              \
                                                                                            \
      NAME##_handle_t add_##NAME##_callback(                                                \
        std::function<void(const PARAM_TYPE&)> fn)                                          \
      {                                                                                     \
        using handle_t = NAME##_handle_t;                                                   \
        using fn_t     = typename handle_t::fn_t;                                           \
        using add_fn   = typename handle_t::add_fn;                                         \
        using rem_fn   = typename handle_t::remove_fn;                                      \
                                                                                            \
        add_fn add = [](window_t* w, fn_t cb) {                                             \
          auto nr = w->STORAGE.NewNodeLast();                                               \
          w->STORAGE[nr] = [cb](const PARAM_TYPE& d){ cb(nullptr, d); };                    \
          return nr;                                                                        \
        };                                                                                  \
                                                                                            \
        rem_fn remove = [](window_t* w, NODE_REF nr) {                                      \
          if (w->STORAGE.NodeList.Current) w->STORAGE.unlrec(nr);                           \
        };                                                                                  \
                                                                                            \
        return handle_t(                                                                    \
          this,                                                                             \
          std::move(add),                                                                   \
          std::move(remove),                                                                \
          [fn](window_t*, const PARAM_TYPE& d){ fn(d); }                                    \
        );                                                                                  \
      }

    // Note that buttons callback, the button state will never be repeat, instead use on_button_down
    FAN_DEFINE_CB_RAII(buttons, m_buttons_callback, buttons_callback_NodeReference_t, buttons_data_t);
    FAN_DEFINE_CB_RAII(keys, m_keys_callback, keys_callback_NodeReference_t, keys_data_t);
    FAN_DEFINE_CB_RAII(text, m_text_callback, text_callback_NodeReference_t, text_data_t);
    FAN_DEFINE_CB_RAII(move, m_move_callback, move_callback_NodeReference_t, move_data_t);
    FAN_DEFINE_CB_RAII(resize, m_resize_callback, resize_callback_NodeReference_t, resize_data_t);
    FAN_DEFINE_CB_RAII(close, m_close_callback, close_callback_NodeReference_t, close_data_t);
    FAN_DEFINE_CB_RAII(mouse_move, m_mouse_position_callback, mouse_position_callback_NodeReference_t, mouse_move_data_t);
    FAN_DEFINE_CB_RAII(mouse_motion, m_mouse_motion_callback, mouse_motion_callback_NodeReference_t, mouse_motion_data_t);

    FAN_DEFINE_CB_RAII(key_down, m_key_down_callbacks, key_down_callbacks_NodeReference_t, keys_data_t);
    FAN_DEFINE_CB_RAII(mouse_down, m_mouse_down_callbacks, mouse_down_callbacks_NodeReference_t, buttons_data_t);

    using key_handle_t = bll_nr_t<key_callback_NodeReference_t, window_t, const key_data_t&>;

    using button_data_t = buttons_data_t;
    using mouse_down_data_t = buttons_data_t;
    using mouse_up_data_t = buttons_data_t;
    using mouse_click_data_t = buttons_data_t;
    using key_down_data_t = key_data_t;
    using key_up_data_t = key_data_t;
    using key_click_data_t = key_data_t;

    void open(std::uint64_t flags);
    void open(fan::vec2i window_size, const std::string& name, std::uint64_t flags = 0);
    void close();
    void make_context_current();
    void handle_key_states(); // can be 1 or 2 aka press or repeat
    uint32_t handle_events();

    // Note key state in this cb will give keyboard repeat delay, if you want instant call on key down, use on_key_down
    key_handle_t add_key_callback(int key, keyboard_state_t st, std::function<void(const key_data_t&)> fn);
    buttons_handle_t on_mouse_click(uint16_t button, buttons_cb_t fn);
    mouse_down_handle_t on_mouse_down(uint16_t button, buttons_cb_t fn);
    buttons_handle_t on_mouse_up(uint16_t button, buttons_cb_t fn);
    key_handle_t on_key_click(int key, key_cb_t fn);
    key_handle_t on_key_down(int key, key_cb_t fn);
    key_handle_t on_key_up(int key, key_cb_t fn);
    mouse_move_handle_t on_mouse_move(mouse_move_cb_t fn);
    resize_handle_t on_resize(resize_cb_t fn);
    fan::vec2i get_size() const;
    void set_size(const fan::vec2i& window_size);
    fan::vec2 get_position() const;
    void set_position(const fan::vec2& position);
    GLFWmonitor* get_current_monitor();
    fan::vec2 get_primary_monitor_resolution();
    fan::vec2 get_current_monitor_resolution();
    void set_windowed();
    void set_fullscreen();
    void set_windowed_fullscreen();
    void set_borderless();
    void set_cursor(int flag);
    void toggle_cursor();
    void set_display_mode(const mode& mode);
    fan::vec2d get_mouse_position() const;
    int key_state(int key) const;
    bool key_pressed(int key, int press = fan::keyboard_state::press) const;
    fan::vec2 get_gamepad_axis(int key) const;

    uint8_t get_antialiasing() const {
      return m_antialiasing_samples;
    }
    void set_antialiasing(int samples) {
      if (samples < 0) {
        samples = 0;
      }

      m_antialiasing_samples = samples;

      if (glfw_window != nullptr) {
        fan::throw_error("Call before making window");
      }
    }
    void set_name(const std::string& name) {
      glfwSetWindowTitle(glfw_window, name.c_str());
    }
    void set_icon(const fan::image::info_t& icon_info) {
      GLFWimage icon;
      icon.width = icon_info.size.x;
      icon.height = icon_info.size.y;
      icon.pixels = (decltype(icon.pixels))icon_info.data;
      glfwSetWindowIcon(glfw_window, 1, &icon);
    }

    void swap_buffers() {
      glfwSwapBuffers(glfw_window);
    }

#if defined(fan_platform_windows)
    //---------------------------Windows specific code---------------------------

  private:
    enum WINDOWCOMPOSITIONATTRIB { WCA_USEDARKMODECOLORS = 26 };
    struct WINDOWCOMPOSITIONATTRIBDATA {
      WINDOWCOMPOSITIONATTRIB Attrib;
      PVOID pvData;
      SIZE_T cbData;
    };
    typedef BOOL(WINAPI* fn_should_apps_use_dark_mode)();
    typedef BOOL(WINAPI* fn_is_dark_mode_allowed_for_window)(HWND hWnd);
    typedef BOOL(WINAPI* fn_set_window_composition_attribute)(HWND hWnd, WINDOWCOMPOSITIONATTRIBDATA* data);
    
    fn_should_apps_use_dark_mode _should_apps_use_dark_mode = nullptr;
    fn_is_dark_mode_allowed_for_window _is_dark_mode_allowed_for_window = nullptr;
    fn_set_window_composition_attribute _set_window_composition_attribute = nullptr;
    DWORD g_build_number = 0;
    bool dark_mode_initialized = false;

    void initialize_dark_mode() {
      if (dark_mode_initialized) {
        return;
      }

      // Get Windows build number
      typedef BOOL(WINAPI* fn_rtl_get_nt_version_numbers)(LPDWORD major, LPDWORD minor, LPDWORD build);
      auto rtl_get_nt_version_numbers = reinterpret_cast<fn_rtl_get_nt_version_numbers>(
        GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlGetNtVersionNumbers"));

      if (rtl_get_nt_version_numbers) {
        DWORD major, minor;
        rtl_get_nt_version_numbers(&major, &minor, &g_build_number);
        g_build_number &= ~0xF0000000;
      }

      // Load dark mode functions from uxtheme.dll
      if (g_build_number >= 17763) { // Windows 10 1809+
        HMODULE uxtheme = LoadLibraryExW(L"uxtheme.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (uxtheme) {
          _should_apps_use_dark_mode = reinterpret_cast<fn_should_apps_use_dark_mode>(
            GetProcAddress(uxtheme, MAKEINTRESOURCEA(132)));
          _is_dark_mode_allowed_for_window = reinterpret_cast<fn_is_dark_mode_allowed_for_window>(
            GetProcAddress(uxtheme, MAKEINTRESOURCEA(137)));
        }

        // Load SetWindowCompositionAttribute from user32.dll
        HMODULE user32 = GetModuleHandleW(L"user32.dll");
        if (user32) {
          _set_window_composition_attribute = reinterpret_cast<fn_set_window_composition_attribute>(
            GetProcAddress(user32, "SetWindowCompositionAttribute"));
        }
      }

      dark_mode_initialized = true;
    }
    bool is_high_contrast() const {
      HIGHCONTRASTW high_contrast = { sizeof(high_contrast) };
      if (SystemParametersInfoW(SPI_GETHIGHCONTRAST, sizeof(high_contrast), &high_contrast, FALSE)) {
        return high_contrast.dwFlags & HCF_HIGHCONTRASTON;
      }
      return false;
    }
  public:

    HWND get_win32_handle() {
      return glfwGetWin32Window(glfw_window);
    }

    void set_topmost() {
      SetWindowPos(get_win32_handle(), HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
    }
    void make_click_through() {
      auto handle = get_win32_handle();
      LONG exStyle = GetWindowLong(handle, GWL_EXSTYLE);
      SetWindowLong(handle, GWL_EXSTYLE, exStyle | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW);
    }

    void apply_window_theme() {
      if (!glfw_window) {
        return;
      }

      initialize_dark_mode();

      HWND hwnd = get_win32_handle();
      if (!hwnd) {
        return;
      }

      BOOL is_dark = FALSE;

      // Check if dark mode should be applied
      if (_is_dark_mode_allowed_for_window &&
        _should_apps_use_dark_mode &&
        _is_dark_mode_allowed_for_window(hwnd) &&
        _should_apps_use_dark_mode() &&
        !is_high_contrast()) {
        is_dark = TRUE;
      }

      // Apply dark mode based on Windows build number
      if (g_build_number < 18362) {
        // Windows 10 versions before 1903
        SetPropW(hwnd, L"UseImmersiveDarkModeColors",
          reinterpret_cast<HANDLE>(static_cast<INT_PTR>(is_dark)));
      }
      else if (_set_window_composition_attribute) {
        // Windows 10 1903+ and Windows 11
        WINDOWCOMPOSITIONATTRIBDATA composition_data = {
            WCA_USEDARKMODECOLORS,
            &is_dark,
            sizeof(is_dark)
        };
        _set_window_composition_attribute(hwnd, &composition_data);
      }
    }
    //---------------------------Windows specific code---------------------------
#endif

    uint8_t renderer = renderer_t::opengl;
    double last_frame_time = glfwGetTime();
    f64_t m_delta_time = 1.0 / 256.0;
    uint32_t m_frame_counter = 0;

    buttons_callback_t m_buttons_callback;
    keys_callback_t m_keys_callback;
    key_callback_t m_key_callback;
    text_callback_t m_text_callback;
    move_callback_t m_move_callback;
    resize_callback_t m_resize_callback;
    close_callback_t m_close_callback;
    mouse_position_callback_t m_mouse_position_callback;
    mouse_motion_callback_t m_mouse_motion_callback;
    mouse_down_callbacks_t m_mouse_down_callbacks;
    key_down_callbacks_t m_key_down_callbacks;

    std::uint64_t flags = 0;
    uint8_t m_antialiasing_samples = 0;

    operator GLFWwindow* () {
      return glfw_window;
    }
    operator GLFWwindow* const () const {
      return glfw_window;
    }

    int prev_key_states[fan::key_last + 1]{};
    int key_states[fan::key_last + 1]{};
    GLFWwindow* glfw_window = nullptr;

    fan::vec2d previous_mouse_position = -0xfff;

    fan::vec2 drag_delta_start = -1;
    uint8_t display_mode = (uint8_t)mode::windowed;
  };
}