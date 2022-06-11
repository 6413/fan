#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(window/window.h)
#include _FAN_PATH(math/random.h)
#include _FAN_PATH(types/utf_string.h)
#include _FAN_PATH(font.h)

#include _FAN_PATH(graphics/shared_core.h)

#if fan_renderer == fan_renderer_opengl
//#include <fan/graphics/opengl/gl_init.h>
#endif
//
#ifdef fan_platform_windows

#include <dwmapi.h>

#pragma comment(lib, "Dwmapi.lib")

#include <windowsx.h>

#include <mbctype.h>

#include <hidusage.h>

#undef min
#undef max

#elif defined(fan_platform_unix)

#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS

#include <locale>
#include <codecvt>

#include <X11/extensions/Xrandr.h>

#include <errno.h>

#endif

#include <string>

#define stringify(name) #name

#if defined(fan_platform_windows)
static constexpr const char* shared_library = "opengl32.dll";
#elif defined(fan_platform_unix)
static constexpr const char* shared_library = "libGL.so.1";

static void open_lib_handle(void** handle) {
  *handle = dlopen(shared_library, RTLD_LAZY | RTLD_NODELETE);
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
  void* result = dlsym(*handle, name);
  #if fan_debug >= fan_debug_low
  if (result == nullptr) {
    dlerror();
    dlsym(*handle, name);
    auto error = dlerror();
    if (error != nullptr) {
      fan::throw_error(error);
    }
  }
  #endif
  return result;
}

#endif

fan::vec2i fan::get_screen_resolution() {
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

void fan::set_screen_resolution(const fan::vec2i& size)
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

void fan::reset_screen_resolution() {

  #ifdef fan_platform_windows

  ChangeDisplaySettings(nullptr, CDS_RESET);

  #elif defined(fan_platform_unix)



  #endif

}

uintptr_t fan::get_screen_refresh_rate()
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

fan::window_t* fan::get_window_by_id(fan::window_handle_t wid) {

  uint32_t it = window_id_storage.begin();
  while(it != window_id_storage.end()) {
    if (window_id_storage[it].window_handle == wid) {
      return window_id_storage[it].window_ptr;
    }
    it = window_id_storage.next(it);
  }

  return 0;
}

void fan::set_window_by_id(fan::window_handle_t wid, fan::window_t* window) {
  window_id_storage.push_back({wid, window});
}

std::string fan::window_t::get_name() const
{
  return m_name;
}

void fan::window_t::set_name(const std::string& name)
{

  m_name = name;

  #ifdef fan_platform_windows

  SetWindowTextA(m_window_handle, m_name.c_str());

  #elif defined(fan_platform_unix)

  XStoreName(fan::sys::m_display, m_window_handle, name.c_str());
  XSetIconName(fan::sys::m_display, m_window_handle, name.c_str());

  #endif

}

void fan::window_t::calculate_delta_time()
{
  m_current_frame = fan::time::clock::now();
  m_delta_time = (f64_t)fan::time::clock::elapsed(m_last_frame) / 1000000000;
  m_last_frame = m_current_frame;
}

f64_t fan::window_t::get_delta_time() const
{
  return m_delta_time;
}

fan::vec2i fan::window_t::get_mouse_position() const
{
  return m_mouse_position;
}

fan::vec2i fan::window_t::get_previous_mouse_position() const
{
  return m_previous_mouse_position;
}

fan::vec2i fan::window_t::get_size() const
{
  return m_size;
}

fan::vec2i fan::window_t::get_previous_size() const
{
  return m_previous_size;
}

void fan::window_t::set_size(const fan::vec2i& size)
{
  #ifdef fan_platform_windows

  RECT rect = { 0, 0, size.x, size.y };

  AdjustWindowRectEx(&rect, GetWindowStyle(m_window_handle), FALSE, GetWindowExStyle(m_window_handle));

  if (!SetWindowPos(m_window_handle, 0, 0, 0, rect.right - rect.left, rect.bottom - rect.top, SWP_NOZORDER | SWP_SHOWWINDOW | SWP_NOMOVE)) {
    fan::print("fan window error: failed to set window position", GetLastError());
    exit(1);
  }

  #elif defined(fan_platform_unix)

  int result = XResizeWindow(fan::sys::m_display, m_window_handle, size.x, size.y);

  if (result == BadValue || result == BadWindow) {
    fan::print("fan window error: failed to set window position");
    exit(1);
  }

  const fan::vec2i move_offset = (size - get_previous_size()) / 2;

  this->set_position(this->get_position() - move_offset);

  #endif
  m_previous_size = m_size;
  m_size = size;
}

fan::vec2i fan::window_t::get_position() const
{
  return m_position;
}

void fan::window_t::set_position(const fan::vec2i& position)
{
  #ifdef fan_platform_windows

  if (!SetWindowPos(m_window_handle, 0, position.x, position.y, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_SHOWWINDOW)) {
    fan::print("fan window error: failed to set window position", GetLastError());
    exit(1);
  }

  #elif defined(fan_platform_unix)

  int result = XMoveWindow(fan::sys::m_display, m_window_handle, position.x, position.y);

  if (result == BadValue || result == BadWindow) {
    fan::print("fan window error: failed to set window position");
    exit(1);
  }

  #endif
}

uintptr_t fan::window_t::get_max_fps() const {
  return m_max_fps;
}

void fan::window_t::set_max_fps(uintptr_t fps) {
  m_max_fps = fps;
  m_fps_next_tick = fan::time::clock::now();
}

void fan::window_t::set_full_screen(const fan::vec2i& size)
{
  fan::window_t::set_size_mode(fan::window_t::mode::full_screen);

  fan::vec2i new_size;

  if (size == uninitialized) {
    new_size = fan::get_screen_resolution();
  }
  else {
    new_size = size;
  }

  #ifdef fan_platform_windows

  this->set_resolution(new_size, fan::window_t::get_size_mode());

  this->set_windowed_full_screen();

  #elif defined(fan_platform_unix)

  this->set_windowed_full_screen(); // yeah

  #endif

}

void fan::window_t::set_windowed_full_screen(const fan::vec2i& size)
{

  fan::window_t::set_size_mode(fan::window_t::mode::borderless);

  fan::vec2i new_size;

  if (size == uninitialized) {
    new_size = fan::get_screen_resolution();
  }
  else {
    new_size = size;
  }

  #ifdef fan_platform_windows

  DWORD dwStyle = GetWindowLong(m_window_handle, GWL_STYLE);

  MONITORINFO mi = { sizeof(mi) };

  if (GetMonitorInfo(MonitorFromWindow(m_window_handle, MONITOR_DEFAULTTOPRIMARY), &mi)) {
    SetWindowLong(m_window_handle, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
    SetWindowPos(
        m_window_handle, HWND_TOP,
        mi.rcMonitor.left, mi.rcMonitor.top,
        new_size.x,
        new_size.y,
        SWP_NOOWNERZORDER | SWP_FRAMECHANGED
    );
  }

  #elif defined(fan_platform_unix)

  struct MwmHints {
    unsigned long flags;
    unsigned long functions;
    unsigned long decorations;
    long input_mode;
    unsigned long status;
  };

  enum {
    MWM_HINTS_FUNCTIONS = (1L << 0),
    MWM_HINTS_DECORATIONS = (1L << 1),

    MWM_FUNC_ALL = (1L << 0),
    MWM_FUNC_RESIZE = (1L << 1),
    MWM_FUNC_MOVE = (1L << 2),
    MWM_FUNC_MINIMIZE = (1L << 3),
    MWM_FUNC_MAXIMIZE = (1L << 4),
    MWM_FUNC_CLOSE = (1L << 5)
  };

  Atom mwmHintsProperty = XInternAtom(fan::sys::m_display, "_MOTIF_WM_HINTS", 0);
  struct MwmHints hints;
  hints.flags = MWM_HINTS_DECORATIONS;
  hints.functions = 0;
  hints.decorations = 0;
  XChangeProperty(fan::sys::m_display, m_window_handle, mwmHintsProperty, mwmHintsProperty, 32,
      PropModeReplace, (unsigned char*)&hints, 5);

  XMoveResizeWindow(fan::sys::m_display, m_window_handle, 0, 0, size.x, size.y);

  #endif

}

void fan::window_t::set_windowed(const fan::vec2i& size)
{
  fan::window_t::set_size_mode(fan::window_t::mode::windowed);

  fan::vec2i new_size;
  if (size == uninitialized) {
    new_size = this->get_previous_size();
  }
  else {
    new_size = size;
  }

  #ifdef fan_platform_windows

  this->set_resolution(0, fan::window_t::get_size_mode());

  const fan::vec2i position = fan::get_screen_resolution() / 2 - new_size / 2;

  ShowWindow(m_window_handle, SW_SHOW);

  SetWindowLongPtr(m_window_handle, GWL_STYLE, WS_OVERLAPPEDWINDOW | WS_VISIBLE);

  SetWindowPos(
      m_window_handle,
      0,
      position.x,
      position.y,
      new_size.x,
      new_size.y,
      SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED
  );

  #elif defined(fan_platform_unix)



  #endif
}

void fan::window_t::set_resolution(const fan::vec2i& size, const mode& mode) const
{
  if (mode == mode::full_screen) {
    fan::set_screen_resolution(size);
  }
  else {
    fan::reset_screen_resolution();
  }
}

fan::window_t::mode fan::window_t::get_size_mode() const
{
  return flag_values::m_size_mode;
}

void fan::window_t::set_size_mode(const mode& mode)
{
  flag_values::m_size_mode = mode;
}

fan::window_t::callback_id_t fan::window_t::add_keys_callback(void* user_ptr, keys_callback_cb_t function)
{
  return this->m_keys_callback.push_back(fan::make_pair(function, user_ptr));
}

void fan::window_t::remove_keys_callback(fan::window_t::callback_id_t id)
{
  this->m_keys_callback.unlink(id);
}

fan::window_t::callback_id_t fan::window_t::add_key_callback(uint16_t key, key_state state, void* user_ptr, key_callback_cb_t function)
{
  return this->m_key_callback.push_back(key_callback_t{ key, state, function, user_ptr });
}

void fan::window_t::edit_key_callback(callback_id_t id, uint16_t key, key_state state, void* user_ptr)
{
  m_key_callback[id].key = key;
  m_key_callback[id].state = state;
  m_key_callback[id].user_ptr = user_ptr;
}

void fan::window_t::remove_key_callback(callback_id_t id)
{
  this->m_key_callback.erase(id);
}

fan::window_t::callback_id_t fan::window_t::add_key_combo_callback(uint16_t* keys, uint32_t n, void* user_ptr, keys_combo_callback_cb_t function)
{
  assert(0);
  return 0;

 /* fan::window_t::key_combo_callback_t p;
  p.key_combo.open();
  p.last_key = keys[n - 1];
  for (int i = 0; i < n - 1; i++) {
    p.key_combo.push_back(keys[i]);
  }
  p.function = function;

  this->add_key_callback(p.last_key, fan::key_state::press, user_ptr, [](fan::window_t* w, uint16_t key, void* user_ptr) {
    uint32_t it = w->m_key_combo_callback.begin();
    while(it != w->m_key_combo_callback.end()) {
       if (w->m_key_combo_callback[it].last_key == key) {
         for (uint32_t i = 0; i < w->m_key_combo_callback[it].key_combo.size(); i++) {
           if (!w->key_press(w->m_key_combo_callback[it].key_combo[i])) {
             goto g_skip;
           }
         }
        w->m_key_combo_callback[it].function(w, user_ptr);
        break;
      }
      g_skip:
      it = w->m_key_combo_callback.next(it);
    }
  });
  return m_key_combo_callback.push_back(p);*/
}

fan::window_t::callback_id_t fan::window_t::add_text_callback(void* user_ptr, text_callback_cb_t function)
{
  return m_text_callback.push_back(fan::make_pair(function, user_ptr));
}

void fan::window_t::remove_text_callback(fan::window_t::callback_id_t id)
{
  m_text_callback.erase(id);
}

fan::window_t::callback_id_t fan::window_t::add_close_callback(void* user_ptr, close_callback_cb_t function)
{
  return m_close_callback.push_back(fan::make_pair(function, user_ptr));
}

void fan::window_t::remove_close_callback(fan::window_t::callback_id_t id)
{
  this->m_close_callback.erase(id);
}

fan::window_t::callback_id_t fan::window_t::add_mouse_move_callback(void* user_ptr, mouse_position_callback_cb_t function)
{
  return this->m_mouse_position_callback.push_back(fan::make_pair(function, user_ptr));
}

void fan::window_t::remove_mouse_move_callback(fan::window_t::callback_id_t id)
{
  this->m_mouse_position_callback.erase(id);
}

fan::window_t::callback_id_t fan::window_t::add_resize_callback(void* user_ptr, resize_callback_cb_t function) {
  return this->m_resize_callback.push_back(fan::make_pair(function, user_ptr));
}

void fan::window_t::remove_resize_callback(fan::window_t::callback_id_t id) {
  this->m_resize_callback.erase(id);
}

fan::window_t::callback_id_t fan::window_t::add_move_callback(void* user_ptr, move_callback_cb_t function)
{
  return this->m_move_callback.push_back(fan::make_pair(function, user_ptr));
}

void fan::window_t::remove_move_callback(fan::window_t::callback_id_t id)
{
  this->m_move_callback.erase(id);
}

void fan::window_t::set_background_color(const fan::color& color)
{
  m_background_color = color;
}

fan::window_handle_t fan::window_t::get_handle() const
{
  return m_window_handle;
}

uintptr_t fan::window_t::get_fps(bool window_name, bool print)
{
  if (!m_fps_timer.count()) {
    m_fps_timer.start(fan::time::milliseconds(1000));
  }

  if (m_received_fps) {
    m_fps = 0;
    m_received_fps = false;
  }

  if (m_fps_timer.finished()) {
    std::string fps_info;
    if (window_name || print) {
      fps_info.append(
          std::string("FPS: ") +
          std::to_string(m_fps) +
          std::string(" frame time: ") +
          std::to_string(1.0 / m_fps * 1000) +
          std::string(" ms")
      ).c_str();
    }
    if (window_name) {
      this->set_name(fps_info.c_str());
    }
    if (print) {
      fan::print(fps_info);
    }

    m_fps_timer.restart();
    m_received_fps = true;
    return m_fps;
  }

  m_fps++;
  return 0;
}

void fan::window_t::open(const fan::vec2i& window_size, const std::string& name, uint64_t flags)
{
  m_size = window_size;
  m_mouse_position = 0;
  m_max_fps = 0;
  m_received_fps = 0;
  m_fps = 0;
  m_last_frame = fan::time::clock::now();
  m_current_frame = fan::time::clock::now();
  m_delta_time = 0;
  m_name = name;
  m_flags = flags;
  m_current_key = 0;
  m_reserved_flags = 0;
  m_focused = true;
  m_fps_timer.m_time = 0;
  m_event_flags = 0;

  if (flag_values::m_size_mode == fan::window_t::mode::not_set) {
    flag_values::m_size_mode = fan::window_t::default_size_mode;
  }
  if (static_cast<bool>(flags & fan::window_t::flags::no_mouse)) {
    fan::window_t::flag_values::m_no_mouse = true;
  }
  if (static_cast<bool>(flags & fan::window_t::flags::no_resize)) {
    fan::window_t::flag_values::m_no_resize = true;
  }
  if (static_cast<bool>(flags & fan::window_t::flags::borderless)) {
    fan::window_t::flag_values::m_size_mode = fan::window_t::mode::borderless;
  }
  if (static_cast<bool>(flags & fan::window_t::flags::full_screen)) {
    fan::window_t::flag_values::m_size_mode = fan::window_t::mode::full_screen;
  }

  initialize_window(name, window_size, flags);

  this->calculate_delta_time();

  #if fan_renderer == fan_renderer_vulkan

  m_vulkan = new fan::vulkan(&m_size, (void*)this->get_handle());

  #endif
}

void fan::window_t::close()
{
  #if fan_renderer == fan_renderer_vulkan

  if (m_vulkan) {
    delete m_vulkan;
  }

  #endif

  this->destroy_window();
}

bool fan::window_t::focused() const
{
  #ifdef fan_platform_windows
  return m_focused;
  #elif defined(fan_platform_unix)
  return 1;
  #endif

}

void fan::window_t::destroy_window()
{
  window_id_storage.close();

  #if defined(fan_platform_windows)

  if (!m_window_handle || !m_hdc
#if fan_renderer == fan_renderer_opengl
      || !m_context
#endif
      ) {
    return;
  }

  PostQuitMessage(0);

  #if fan_renderer == fan_renderer_opengl
  wglMakeCurrent(m_hdc, 0);
  #endif

  ReleaseDC(m_window_handle, m_hdc);
  DestroyWindow(m_window_handle);

  #if fan_debug >= fan_debug_low
  m_hdc = 0;
  m_window_handle = 0;
  #endif

  #elif defined(fan_platform_unix)

  if (!fan::sys::m_display || !m_visual || !m_window_attribs.colormap) {
    return;
  }

  XFree(m_visual);
  XFreeColormap(fan::sys::m_display, m_window_attribs.colormap);
  XDestroyWindow(fan::sys::m_display, m_window_handle);

  //    glXDestroyContext(fan::sys::m_display, m_context);

  XCloseDisplay(fan::sys::m_display);
  #if fan_debug >= fan_debug_low
  m_visual = 0;
  m_window_attribs.colormap = 0;
  #endif

  #endif

  m_keys_callback.close();
	m_key_callback.close();
	m_key_combo_callback.close();

	m_text_callback.close();
	m_move_callback.close();
	m_resize_callback.close();
	m_close_callback.close();
	m_mouse_position_callback.close();
}

uint16_t fan::window_t::get_current_key() const
{
  return m_current_key;
}

fan::vec2i fan::window_t::get_raw_mouse_offset() const
{
  return m_raw_mouse_offset;
}

void fan::window_t::window_input_action(fan::window_handle_t window, uint16_t key) {

  fan::window_t* fwindow;

  #ifdef fan_platform_windows

  fwindow = get_window_by_id(window);

  #elif defined(fan_platform_unix)

  fwindow = get_window_by_id(window);

  #endif

  auto it = fwindow->m_key_callback.begin();

  while (it != fwindow->m_key_callback.end()) {

    fwindow->m_key_callback.start_safe_next(it);

    if (key != fwindow->m_key_callback[it].key || fwindow->m_key_callback[it].state == key_state::release) {
      it = fwindow->m_key_callback.end_safe_next();
      continue;
    }

    if (fwindow->m_key_callback[it].function) {
      fwindow->m_key_callback[it].function(fwindow, key, fwindow->m_key_callback[it].user_ptr);
    }

    it = fwindow->m_key_callback.end_safe_next();
  }
}

void fan::window_t::window_input_mouse_action(fan::window_handle_t window, uint16_t key)
{
  fan::window_t* fwindow;

  #ifdef fan_platform_windows

  fwindow = fan::get_window_by_id(window);

  #elif defined(fan_platform_unix)

  fwindow = this;

  #endif

  auto it = fwindow->m_key_callback.begin();

  while (it != fwindow->m_key_callback.end()) {

    fwindow->m_key_callback.start_safe_next(it);

    if (key != fwindow->m_key_callback[it].key || fwindow->m_key_callback[it].state == key_state::release) {
      it = fwindow->m_key_callback.end_safe_next();
      continue;
    }
    if (fwindow->m_key_callback[it].function) {
      fwindow->m_key_callback[it].function(fwindow, key, fwindow->m_key_callback[it].user_ptr);
    }

    it = fwindow->m_key_callback.end_safe_next();
  }
}

void fan::window_t::window_input_up(fan::window_handle_t window, uint16_t key)
{
  fan::window_t* fwindow;
  #ifdef fan_platform_windows

  fwindow = fan::get_window_by_id(window);

  #elif defined(fan_platform_unix)

  fwindow = this;

  #endif

  if (key <= fan::input::key_menu) {
    fan::window_t::window_input_action_reset(window, key);
  }

  auto it = fwindow->m_key_callback.begin();

  while (it != fwindow->m_key_callback.end()) {

    fwindow->m_key_callback.start_safe_next(it);

    if (key != fwindow->m_key_callback[it].key || fwindow->m_key_callback[it].state == key_state::press) {
      it = fwindow->m_key_callback.end_safe_next();
      continue;
    }
    if (fwindow->m_key_callback[it].function) {
      fwindow->m_key_callback[it].function(fwindow, key, fwindow->m_key_callback[it].user_ptr);
    }

    it = fwindow->m_key_callback.end_safe_next();
  }
}

void fan::window_t::window_input_action_reset(fan::window_handle_t window, uint16_t key)
{

}

#ifdef fan_platform_windows

static void handle_special(WPARAM wparam, LPARAM lparam, uint16_t& key, bool down) {
  if (wparam == 0x10 || wparam == 0x11) {
    if (down) {
      switch (lparam) {
        case fan::special_lparam::lshift_lparam_down:
        {
          key = fan::input::key_left_shift;
          break;
        }
        case fan::special_lparam::rshift_lparam_down:
        {
          key = fan::input::key_right_shift;
          break;
        }
        case fan::special_lparam::lctrl_lparam_down:
        {
          key = fan::input::key_left_control;
          break;
        }
        case fan::special_lparam::rctrl_lparam_down:
        {
          key = fan::input::key_right_control;
          break;
        }
      }
    }
    else {
      switch (lparam) {
        case fan::special_lparam::lshift_lparam_up: // ?
        {
          key = fan::input::key_left_shift;
          break;
        }
        case fan::special_lparam::rshift_lparam_up:
        {
          key = fan::input::key_right_shift;
          break;
        }
        case fan::special_lparam::lctrl_lparam_up:
        {
          key = fan::input::key_left_control;
          break;
        }
        case fan::special_lparam::rctrl_lparam_up: // ? 
        {
          key = fan::input::key_right_control;
          break;
        }
      }
    }
  }
  else {
    key = fan::window_input::convert_keys_to_fan(wparam);
  }

}

LRESULT fan::window_t::window_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam)
{
  switch (msg) {
    case WM_MOUSEMOVE:
    {
      auto window = fan::get_window_by_id(hwnd);

      if (!window) {
        break;
      }

      const auto get_cursor_position = [&] {
        POINT p;
        GetCursorPos(&p);
        ScreenToClient(window->m_window_handle, &p);

        return fan::vec2i(p.x, p.y);
      };

      const fan::vec2i position(get_cursor_position());

      const auto cursor_in_range = [](const fan::vec2i& position, const fan::vec2& window_size) {
        return position.x >= 0 && position.x < window_size.x&&
          position.y >= 0 && position.y < window_size.y;
      };

      window->m_mouse_position = position;

      window->call_mouse_move_cb = true;

      break;
    }
    case WM_MOVE:
    {

      auto window = fan::get_window_by_id(hwnd);

      if (!window) {
        break;
      }

      window->m_position = fan::vec2i(
          static_cast<int>(static_cast<short>(LOWORD(lparam))),
          static_cast<int>(static_cast<short>(HIWORD(lparam)))
      );

      auto it = window->m_move_callback.begin();

      while (it != window->m_move_callback.end()) {
        it = window->m_move_callback.next(it);
      }

      break;
    }
    case WM_SIZE:
    {
      fan::window_t* fwindow = fan::get_window_by_id(hwnd);

      if (!fwindow) {
        break;
      }

      RECT rect;
      GetClientRect(hwnd, &rect);

      fwindow->m_previous_size = fwindow->m_size;
      fwindow->m_size = fan::vec2i(rect.right - rect.left, rect.bottom - rect.top);

      #if fan_renderer == fan_renderer_opengl
      wglMakeCurrent(fwindow->m_hdc, m_context);
      #endif

      auto it = fwindow->m_resize_callback.begin();

      while (it != fwindow->m_resize_callback.end()) {

        fwindow->m_resize_callback[it].first(fwindow, fwindow->m_size, fwindow->m_resize_callback[it].second);

        it = fwindow->m_resize_callback.next(it);
      }

      break;
    }
    case WM_SETFOCUS:
    {
      fan::window_t* fwindow = fan::get_window_by_id(hwnd);

      if (!fwindow) {
        break;
      }

      fwindow->m_focused = true;
      break;
    }
    case WM_KILLFOCUS:
    {
      fan::window_t* fwindow = fan::get_window_by_id(hwnd);
      if (!fwindow) {
        break;
      }

      fwindow->m_focused = false;
      break;
    }
    case WM_SYSCOMMAND:
    {
      //auto fwindow = get_window_storage<fan::window_t*>(m_window, stringify(this_window));
      // disable alt action for window
      if (wparam == SC_KEYMENU && (lparam >> 16) <= 0) {
        return 0;
      }

      break;
    }
    case WM_DESTROY:
    {

      PostQuitMessage(0);

      break;
    }
    case WM_CLOSE:
    {
      fan::window_t* fwindow = fan::get_window_by_id(hwnd);

      //if (fwindow->key_press(fan::key_alt)) {
      //	return 0;
      //}

      for (int i = 0; i < fwindow->m_close_callback.size(); i++) {
        fwindow->m_close_callback[i].first(fwindow, fwindow->m_close_callback[i].second);
      }

      fwindow->m_event_flags |= fan::window_t::events::close;

      break;
    }
  }

  return DefWindowProc(hwnd, msg, wparam, lparam);
}
#endif

#ifdef fan_platform_windows

#elif defined(fan_platform_unix)

static bool isExtensionSupported(const char* extList, const char* extension) {
  const char* start;
  const char* where, * terminator;

  where = strchr(extension, ' ');
  if (where || *extension == '\0') {
    return false;
  }

  for (start = extList;;) {
    where = strstr(start, extension);

    if (!where) {
      break;
    }

    terminator = where + strlen(extension);

    if (where == start || *(where - 1) == ' ') {
      if (*terminator == ' ' || *terminator == '\0') {
        return true;
      }
    }

    start = terminator;
  }

  return false;
}
#endif

void fan::window_t::initialize_window(const std::string& name, const fan::vec2i& window_size, uint64_t flags)
{
  m_keys_callback.open();
	m_key_callback.open();
	m_key_combo_callback.open();

	m_text_callback.open();
	m_move_callback.open();
	m_resize_callback.open();
	m_close_callback.open();
	m_mouse_position_callback.open();

  window_id_storage.open();

  #ifdef fan_platform_windows

  auto instance = GetModuleHandle(NULL);

  WNDCLASS wc = { 0 };

  auto str = fan::random::string(10);

  wc.lpszClassName = str.c_str();

  wc.lpfnWndProc = fan::window_t::window_proc;

  wc.hCursor = LoadCursor(NULL, IDC_ARROW);

  wc.hInstance = instance;

  RegisterClass(&wc);

  const bool full_screen = flag_values::m_size_mode == fan::window_t::mode::full_screen;
  const bool borderless = flag_values::m_size_mode == fan::window_t::mode::borderless;

  RECT rect = { 0, 0, window_size.x, window_size.y };
  AdjustWindowRect(&rect, full_screen || borderless ? WS_POPUP : WS_OVERLAPPEDWINDOW, FALSE);

  const fan::vec2i position = fan::get_screen_resolution() / 2 - window_size / 2;

  if (full_screen) {
    this->set_resolution(window_size, fan::window_t::mode::full_screen);
  }

  m_window_handle = CreateWindow(str.c_str(), name.c_str(),
      (flag_values::m_no_resize ? ((full_screen || borderless ? WS_POPUP : (WS_OVERLAPPED | WS_MINIMIZEBOX | WS_SYSMENU)) | WS_SYSMENU) :
    (full_screen || borderless ? WS_POPUP : WS_OVERLAPPEDWINDOW)) | WS_VISIBLE,
      position.x, position.y,
      rect.right - rect.left, rect.bottom - rect.top,
      0, 0, 0, 0);

  if (!m_window_handle) {
    fan::throw_error("failed to initialize window:" + std::to_string(GetLastError()));
  }

  RAWINPUTDEVICE r_id;
  r_id.usUsagePage = HID_USAGE_PAGE_GENERIC;
  r_id.usUsage = HID_USAGE_GENERIC_MOUSE;
  r_id.dwFlags = RIDEV_INPUTSINK;
  r_id.hwndTarget = m_window_handle;

  BOOL result = RegisterRawInputDevices(&r_id, 1, sizeof(RAWINPUTDEVICE));

  if (!result) {
    fan::throw_error("failed to register raw input:" + std::to_string(result));
  }

  ShowCursor(!flag_values::m_no_mouse);
  if (flag_values::m_no_mouse) {
    auto middle = this->get_position() + this->get_size() / 2;
    SetCursorPos(middle.x, middle.y);
  }

  #elif defined(fan_platform_unix)

  m_xim = 0;
  m_xic = 0;

  // if vulkan
  XInitThreads();

  if (!fan::sys::m_display) {
    fan::sys::m_display = XOpenDisplay(NULL);
    if (!fan::sys::m_display) {
      throw std::runtime_error("failed to initialize window");
    }

  }

  static bool init_once = true;

  if (init_once) {
    fan::sys::m_screen = DefaultScreen(fan::sys::m_display);
    init_once = false;
  }

  void* lib_handle;
  open_lib_handle(&lib_handle);
  fan::opengl::glx::PFNGLXGETPROCADDRESSPROC glXGetProcAddress = (decltype(glXGetProcAddress))get_lib_proc(&lib_handle, "glXGetProcAddress");
  if (glXGetProcAddress == nullptr) {
    fan::throw_error("failed to initialize glxGetprocAddress");
  }
  close_lib_handle(&lib_handle);
  static fan::opengl::glx::PFNGLXMAKECURRENTPROC glXMakeCurrent = (decltype(glXMakeCurrent))glXGetProcAddress((const fan::opengl::GLubyte*)"glXMakeCurrent");
  static fan::opengl::glx::PFNGLXGETCURRENTDRAWABLEPROC glXGetCurrentDrawable = (decltype(glXGetCurrentDrawable))glXGetProcAddress((const fan::opengl::GLubyte*)"glXGetCurrentDrawable");
  static fan::opengl::glx::PFNGLXSWAPINTERVALEXTPROC glXSwapIntervalEXT = (decltype(glXSwapIntervalEXT))glXGetProcAddress((const fan::opengl::GLubyte*)"glXSwapIntervalEXT");
  static fan::opengl::glx::PFNGLXDESTROYCONTEXTPROC glXDestroyContext = (decltype(glXDestroyContext))glXGetProcAddress((const fan::opengl::GLubyte*)"glXDestroyContext");
  static fan::opengl::glx::PFNGLXCHOOSEFBCONFIGPROC glXChooseFBConfig = (decltype(glXChooseFBConfig))glXGetProcAddress((const fan::opengl::GLubyte*)"glXChooseFBConfig");
  static fan::opengl::glx::PFNGLXGETVISUALFROMFBCONFIGPROC glXGetVisualFromFBConfig = (decltype(glXGetVisualFromFBConfig))glXGetProcAddress((const fan::opengl::GLubyte*)"glXGetVisualFromFBConfig");
  static fan::opengl::glx::PFNGLXQUERYVERSIONPROC glXQueryVersion = (decltype(glXQueryVersion))glXGetProcAddress((const fan::opengl::GLubyte*)"glXQueryVersion");
  static fan::opengl::glx::PFNGLXGETFBCONFIGATTRIBPROC glXGetFBConfigAttrib = (decltype(glXGetFBConfigAttrib))glXGetProcAddress((const fan::opengl::GLubyte*)"glXGetFBConfigAttrib");
  static fan::opengl::glx::PFNGLXQUERYEXTENSIONSSTRINGPROC glXQueryExtensionsString = (decltype(glXQueryExtensionsString))glXGetProcAddress((const fan::opengl::GLubyte*)"glXQueryExtensionsString");
  static fan::opengl::glx::PFNGLXGETCURRENTCONTEXTPROC glXGetCurrentContext = (decltype(glXGetCurrentContext))glXGetProcAddress((const fan::opengl::GLubyte*)"glXGetCurrentContext");
  static fan::opengl::glx::PFNGLXSWAPBUFFERSPROC glXSwapBuffers = (decltype(glXSwapBuffers))glXGetProcAddress((const fan::opengl::GLubyte*)"glXSwapBuffers");
  static fan::opengl::glx::PFNGLXCREATENEWCONTEXTPROC glXCreateNewContext = (decltype(glXCreateNewContext))glXGetProcAddress((const fan::opengl::GLubyte*)"glXCreateNewContext");
  static fan::opengl::glx::PFNGLXCREATECONTEXTATTRIBSARBPROC glXCreateContextAttribsARB = (decltype(glXCreateContextAttribsARB))glXGetProcAddress((const fan::opengl::GLubyte*)"glXCreateContextAttribsARB");


  int minor_glx = 0, major_glx = 0;
  glXQueryVersion(fan::sys::m_display, &major_glx, &minor_glx);

  constexpr auto major = 2;
  constexpr auto minor = 1;

  if (minor_glx < minor && major_glx <= major) {
    fan::print("fan window error: too low glx version");
    XCloseDisplay(fan::sys::m_display);
    exit(1);
  }

  constexpr uint32_t samples = 0;

  int pixel_format_attribs[] = {
    GLX_X_RENDERABLE, True,
    GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
    GLX_RENDER_TYPE, GLX_RGBA_BIT,
    GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
    GLX_RED_SIZE, 8,
    GLX_GREEN_SIZE, 8,
    GLX_BLUE_SIZE, 8,
    GLX_ALPHA_SIZE, 8,
    GLX_DEPTH_SIZE, 24,
    GLX_STENCIL_SIZE, 8,
    GLX_DOUBLEBUFFER, True,
    GLX_SAMPLE_BUFFERS, 1,
    GLX_SAMPLES, samples,
    None
  };

  if (!samples) {
    // set back to zero to disable antialising
    for (int i = 0; i < 4; i++) {
      pixel_format_attribs[22 + i] = 0;
    }
  }

  int fbcount;

  auto fbc = glXChooseFBConfig(fan::sys::m_display, fan::sys::m_screen, pixel_format_attribs, &fbcount);

  if (!fbc) {
    fan::print("fan window error: failed to retreive framebuffer");
    XCloseDisplay(fan::sys::m_display);
    exit(1);
  }

  int best_fbc = -1, worst_fbc = -1, best_num_samp = -1, worst_num_samp = 999;
  for (int i = 0; i < fbcount; ++i) {
    XVisualInfo* vi = glXGetVisualFromFBConfig(fan::sys::m_display, fbc[i]);
    if (vi != 0) {
      int samp_buf, samples;
      if (!glXGetFBConfigAttrib) {
        exit(1);
      }
      glXGetFBConfigAttrib(fan::sys::m_display, fbc[i], GLX_SAMPLE_BUFFERS, &samp_buf);
      glXGetFBConfigAttrib(fan::sys::m_display, fbc[i], GLX_SAMPLES, &samples);

      if (best_fbc < 0 || (samp_buf && samples > best_num_samp)) {
        best_fbc = i;
        best_num_samp = samples;
      }
      if (worst_fbc < 0 || !samp_buf || samples < worst_num_samp)
        worst_fbc = i;
      worst_num_samp = samples;
    }
    XFree(vi);
  }

  fan::opengl::glx::GLXFBConfig bestFbc = fbc[best_fbc];

  XFree(fbc);

  m_visual = glXGetVisualFromFBConfig(fan::sys::m_display, bestFbc);

  if (!m_visual) {
    fan::print("fan window error: failed to create visual");
    XCloseDisplay(fan::sys::m_display);
    exit(1);
  }

  if (fan::sys::m_screen != m_visual->screen) {
    fan::print("fan window error: screen doesn't match with visual screen");
    XCloseDisplay(fan::sys::m_display);
    exit(1);
  }

  std::memset(&m_window_attribs, 0, sizeof(m_window_attribs));

  std::memset(&m_window_attribs, 0, sizeof(m_window_attribs));

  m_window_attribs.border_pixel = BlackPixel(fan::sys::m_display, fan::sys::m_screen);
  m_window_attribs.background_pixel = WhitePixel(fan::sys::m_display, fan::sys::m_screen);
  m_window_attribs.override_redirect = True;
  m_window_attribs.colormap = XCreateColormap(fan::sys::m_display, RootWindow(fan::sys::m_display, fan::sys::m_screen), m_visual->visual, AllocNone);
  m_window_attribs.event_mask = ExposureMask | KeyPressMask | ButtonPress |
    StructureNotifyMask | ButtonReleaseMask |
    KeyReleaseMask | EnterWindowMask | LeaveWindowMask |
    PointerMotionMask | Button1MotionMask | VisibilityChangeMask |
    ColormapChangeMask;


  const fan::vec2i position = fan::get_screen_resolution() / 2 - window_size / 2;

  m_window_handle = XCreateWindow(
      fan::sys::m_display,
      RootWindow(fan::sys::m_display, fan::sys::m_screen),
      position.x,
      position.y,
      window_size.x,
      window_size.y,
      0,
      m_visual->depth,
      InputOutput,
      m_visual->visual,
      CWBackPixel | CWColormap | CWBorderPixel | CWEventMask | CWCursor,
      &m_window_attribs
  );

  if (flags & fan::window_t::flags::no_resize) {
    auto sh = XAllocSizeHints();
    sh->flags = PMinSize | PMaxSize;
    sh->min_width = sh->max_width = window_size.x;
    sh->min_height = sh->max_height = window_size.y;
    XSetWMSizeHints(fan::sys::m_display, m_window_handle, sh, XA_WM_NORMAL_HINTS);
    XFree(sh);
  }

  this->set_name(name);

  if (!m_atom_delete_window) {
    m_atom_delete_window = XInternAtom(fan::sys::m_display, "WM_DELETE_WINDOW", False);
  }

  XSetWMProtocols(fan::sys::m_display, m_window_handle, &m_atom_delete_window, 1);

  //TODO FIX
  int gl_attribs[] = {
    GLX_CONTEXT_MINOR_VERSION_ARB, 1,
    GLX_CONTEXT_MAJOR_VERSION_ARB, 2,
    GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
    0
  };

  bool initialize_context = !m_context;

  const char* glxExts = glXQueryExtensionsString(fan::sys::m_display, fan::sys::m_screen);
  if (!isExtensionSupported(glxExts, "GLX_ARB_create_context") && initialize_context) {
    std::cout << "GLX_ARB_create_context not supported\n";
    m_context = glXCreateNewContext(fan::sys::m_display, bestFbc, GLX_RGBA_TYPE, 0, True);
  }
  else if (initialize_context) {
    m_context = glXCreateContextAttribsARB(fan::sys::m_display, bestFbc, 0, true, gl_attribs);
  }

  initialize_context = false;

  XSync(fan::sys::m_display, True);

  #if fan_renderer == fan_renderer_opengl
  glXMakeCurrent(fan::sys::m_display, m_window_handle, m_context);
  #endif

  XClearWindow(fan::sys::m_display, m_window_handle);
  XMapRaised(fan::sys::m_display, m_window_handle);
  XAutoRepeatOn(fan::sys::m_display);

  m_xim = XOpenIM(fan::sys::m_display, 0, 0, 0);

  if (!m_xim) {
    // fallback to internal input method
    XSetLocaleModifiers("@im=none");
    m_xim = XOpenIM(fan::sys::m_display, 0, 0, 0);
  }

  m_xic = XCreateIC(m_xim,
      XNInputStyle, XIMPreeditNothing | XIMStatusNothing,
      XNClientWindow, m_window_handle,
      XNFocusWindow, m_window_handle,
      NULL);

  XSetICFocus(m_xic);

  #endif

  m_position = position;

  m_previous_size = m_size;

  set_window_by_id(m_window_handle, this);
}

uint32_t fan::window_t::handle_events() {

  this->calculate_delta_time();

  if (m_max_fps) {

    uint64_t dt = fan::time::clock::now() - m_fps_next_tick;

    uint64_t goal_fps = m_max_fps;

    int64_t frame_time = std::pow(10, 9) * (1.0 / goal_fps);
    frame_time -= dt;

    fan::delay(fan::time::nanoseconds(std::max((int64_t)0, frame_time)));

    m_fps_next_tick = fan::time::clock::now();

  }

  if (call_mouse_move_cb) {

    auto it = m_mouse_position_callback.begin();
    while (it != m_mouse_position_callback.end()) {

      m_mouse_position_callback.start_safe_next(it);

      m_mouse_position_callback[it].first(this, m_mouse_position, m_mouse_position_callback[it].second);

      it = m_mouse_position_callback.end_safe_next();
    }
  }

  m_previous_mouse_position = m_mouse_position;
  call_mouse_move_cb = false;

  #ifdef fan_platform_windows

  MSG msg{};

  while (PeekMessageW(&msg, 0, 0, 0, PM_REMOVE))
  {
   // fan::print(msg.message);
    switch (msg.message) {
      case WM_SYSKEYDOWN:
      case WM_KEYDOWN:
      {
        auto window = fan::get_window_by_id(msg.hwnd);

        if (!window) {
          break;
        }

        uint16_t key;

        handle_special(msg.wParam, msg.lParam, key, true);

        bool repeat = msg.lParam & (1 << 30);

        fan::window_t::window_input_action(window->m_window_handle, key);

        window->m_current_key = key;

        auto it = window->m_keys_callback.begin();

        while (it != window->m_keys_callback.end()) {

          window->m_keys_callback.start_safe_next(it);

          window->m_keys_callback[it].first(window, key, repeat ? fan::key_state::repeat : fan::key_state::press, window->m_keys_callback[it].second);

          it = window->m_keys_callback.end_safe_next();
        }

        break;
      }
      case WM_CHAR:
      {

        auto window = fan::get_window_by_id(msg.hwnd);

        if (!window) {
          break;
        }

        if (msg.wParam < 8) {
          window->m_reserved_flags |= msg.wParam;
        }
        else {

          bool found = false;

          if (!found) {

            uint16_t fan_key = fan::window_input::convert_utfkeys_to_fan(msg.wParam);

            bool found = false;

            for (auto i : banned_keys) {
              if (fan_key == i) {

                found = true;
                break;
              }
            }

            if (!found) {

              auto src = msg.wParam + (window->m_reserved_flags << 8);

              auto utf8_str = fan::utf16_to_utf8((wchar_t*)&src);

              uint32_t value = 0;

              for (int i = 0, j = 0; i < utf8_str.size(); i++, j += 0x08) {
                value |= (uint8_t)utf8_str[i] << j;
              }

              auto it = window->m_text_callback.begin();

              while (it != window->m_text_callback.end()) {

                window->m_text_callback[it].first(window, value, window->m_text_callback[it].second);

                it = window->m_text_callback.next(it);
              }
            }

            window->m_reserved_flags = 0;
          }

        }

        break;
      }
      case WM_LBUTTONDOWN:
      {
        auto window = fan::get_window_by_id(msg.hwnd);

        if (!window) {
          break;
        }

        const uint16_t key = fan::input::mouse_left;

        fan::window_t::window_input_mouse_action(window->m_window_handle, key);

        auto it = window->m_keys_callback.begin();

        while (it != window->m_keys_callback.end()) {

          window->m_keys_callback[it].first(window, key, key_state::press, window->m_keys_callback[it].second);

          it = window->m_keys_callback.next(it);
        }

        break;
      }
      case WM_RBUTTONDOWN:
      {
        auto window = fan::get_window_by_id(msg.hwnd);

        if (!window) {
          break;
        }

        const uint16_t key = fan::input::mouse_right;

        fan::window_t::window_input_mouse_action(window->m_window_handle, key);

        auto it = window->m_keys_callback.begin();

        while (it != window->m_keys_callback.end()) {

          window->m_keys_callback[it].first(window, key, key_state::press, window->m_keys_callback[it].second);

          it = window->m_keys_callback.next(it);
        }

        break;
      }
      case WM_MBUTTONDOWN:
      {
        auto window = fan::get_window_by_id(msg.hwnd);

        if (!window) {
          break;
        }

        const uint16_t key = fan::input::mouse_middle;

        fan::window_t::window_input_mouse_action(window->m_window_handle, key);

        auto it = window->m_keys_callback.begin();

        while (it != window->m_keys_callback.end()) {

          window->m_keys_callback[it].first(window, key, key_state::press, window->m_keys_callback[it].second);

          it = window->m_keys_callback.next(it);
        }

        break;
      }
      case WM_SYSKEYUP:
      case WM_KEYUP:
      {
        auto window = fan::get_window_by_id(msg.hwnd);

        if (!window) {
          break;
        }

        uint16_t key = 0;

        handle_special(msg.wParam, msg.lParam, key, false);

        window_input_up(window->m_window_handle, key);

        auto it = window->m_keys_callback.begin();

        while (it != window->m_keys_callback.end()) {
          window->m_keys_callback.start_safe_next(it);

          window->m_keys_callback[it].first(window, key, key_state::release, window->m_keys_callback[it].second);

          it = window->m_keys_callback.end_safe_next();
        }

        break;
      }
      case WM_MOUSEWHEEL:
      {
        auto fwKeys = GET_KEYSTATE_WPARAM(msg.wParam);
        auto zDelta = GET_WHEEL_DELTA_WPARAM(msg.wParam);

        auto window = fan::get_window_by_id(msg.hwnd);

        fan::window_t::window_input_mouse_action(window->m_window_handle, zDelta < 0 ? fan::input::mouse_scroll_down : fan::input::mouse_scroll_up);

        auto it = window->m_keys_callback.begin();

        while (it != window->m_keys_callback.end()) {

          window->m_keys_callback[it].first(window, zDelta < 0 ? fan::input::mouse_scroll_down : fan::input::mouse_scroll_up, key_state::press, window->m_keys_callback[it].second);

          it = window->m_keys_callback.next(it);
        }

        break;
      }

      case WM_INPUT:
      {
        auto window = fan::get_window_by_id(msg.hwnd);

        if (!window) {
          break;
        }

        if (!window->focused()) {
          break;
        }

        UINT size = sizeof(RAWINPUT);
        BYTE data[sizeof(RAWINPUT)];

        GetRawInputData((HRAWINPUT)msg.lParam, RID_INPUT, data, &size, sizeof(RAWINPUTHEADER));

        RAWINPUT* raw = (RAWINPUT*)data;

        static bool allow_outside = false;

        const auto cursor_in_range = [](const fan::vec2i& position, const fan::vec2& window_size) {
          return position.x >= 0 && position.x < window_size.x&&
            position.y >= 0 && position.y < window_size.y;
        };

        if (raw->header.dwType == RIM_TYPEMOUSE)
        {

          const auto get_cursor_position = [&] {
            POINT p;
            GetCursorPos(&p);
            ScreenToClient(window->m_window_handle, &p);

            return fan::vec2i(p.x, p.y);
          };

          if (fan::is_flag(raw->data.mouse.usButtonFlags, RI_MOUSE_LEFT_BUTTON_DOWN) ||
              fan::is_flag(raw->data.mouse.usButtonFlags, RI_MOUSE_MIDDLE_BUTTON_DOWN) ||
              fan::is_flag(raw->data.mouse.usButtonFlags, RI_MOUSE_RIGHT_BUTTON_DOWN)
              ) {

            const fan::vec2i position(get_cursor_position());

            if (cursor_in_range(position, window->get_size())) {
              allow_outside = true;
            }

            if (fan::window_t::flag_values::m_no_mouse) {
              RECT rect;
              GetClientRect(window->m_window_handle, &rect);

              POINT ul;
              ul.x = rect.left;
              ul.y = rect.top;

              POINT lr;
              lr.x = rect.right;
              lr.y = rect.bottom;

              MapWindowPoints(window->m_window_handle, nullptr, &ul, 1);
              MapWindowPoints(window->m_window_handle, nullptr, &lr, 1);

              rect.left = ul.x;
              rect.top = ul.y;

              rect.right = lr.x;
              rect.bottom = lr.y;

              ClipCursor(&rect);
            }
            else {
              window->m_previous_mouse_position = window->m_mouse_position;
              window->m_mouse_position = position;
            }

          }

          else if (fan::is_flag(raw->data.mouse.usButtonFlags, RI_MOUSE_LEFT_BUTTON_UP)) {

            auto it = window->m_keys_callback.begin();

            while (it != window->m_keys_callback.end()) {

              window->m_keys_callback.start_safe_next(it);

              window->m_keys_callback[it].first(window, fan::mouse_left, key_state::release, window->m_keys_callback[it].second);

              it = window->m_keys_callback.end_safe_next();
            }

            window_input_up(window->m_window_handle, fan::input::mouse_left); allow_outside = false;
          }

          else if (fan::is_flag(raw->data.mouse.usButtonFlags, RI_MOUSE_MIDDLE_BUTTON_UP)) {

            auto it = window->m_keys_callback.begin();

            while (it != window->m_keys_callback.end()) {

              window->m_keys_callback.start_safe_next(it);

              window->m_keys_callback[it].first(window, fan::mouse_middle, key_state::release, window->m_keys_callback[it].second);

              it = window->m_keys_callback.end_safe_next();
            }

            window_input_up(window->m_window_handle, fan::input::mouse_middle); allow_outside = false;
          }

          else if (fan::is_flag(raw->data.mouse.usButtonFlags, RI_MOUSE_RIGHT_BUTTON_UP)) {

            auto it = window->m_keys_callback.begin();

            while (it != window->m_keys_callback.end()) {

              window->m_keys_callback.start_safe_next(it);

              window->m_keys_callback[it].first(window, fan::mouse_right, key_state::release, window->m_keys_callback[it].second);

              it = window->m_keys_callback.end_safe_next();
            }

            window_input_up(window->m_window_handle, fan::input::mouse_right); allow_outside = false;
          }

          else if ((raw->data.mouse.usFlags & MOUSE_MOVE_RELATIVE) == MOUSE_MOVE_RELATIVE) {

            const fan::vec2i position(get_cursor_position());

            window->m_raw_mouse_offset = fan::vec2i(raw->data.mouse.lLastX, raw->data.mouse.lLastY);

            if ((!cursor_in_range(position, window->get_size()) && !allow_outside)) {
              break;
            }

            if (fan::window_t::flag_values::m_no_mouse) {
              RECT rect;
              GetClientRect(window->m_window_handle, &rect);

              POINT ul;
              ul.x = rect.left;
              ul.y = rect.top;

              POINT lr;
              lr.x = rect.right;
              lr.y = rect.bottom;

              MapWindowPoints(window->m_window_handle, nullptr, &ul, 1);
              MapWindowPoints(window->m_window_handle, nullptr, &lr, 1);

              rect.left = ul.x;
              rect.top = ul.y;

              rect.right = lr.x;
              rect.bottom = lr.y;

              ClipCursor(&rect);
            }
          }

        }
        break;
      }
    }

    TranslateMessage(&msg);
    DispatchMessage(&msg);
  }



  #elif defined(fan_platform_unix)

  XEvent event;

  int nevents = XEventsQueued(fan::sys::m_display, QueuedAfterReading);

  while (nevents--) {
    XNextEvent(fan::sys::m_display, &event);
    // if (XFilterEvent(&m_event, m_window))
    // 	continue;

    switch (event.type) {

      case Expose:
      {
        auto window = fan::get_window_by_id(event.xexpose.window);

        if (!window) {
          break;
        }

        XWindowAttributes attribs;
        XGetWindowAttributes(fan::sys::m_display, window->m_window_handle, &attribs);

        window->m_previous_size = window->m_size;
        window->m_size = fan::vec2i(attribs.width, attribs.height);


        auto it = window->m_resize_callback.begin();

        while (it != window->m_resize_callback.end()) {

          window->m_resize_callback[it].first(window, window->m_size, window->m_resize_callback[it].second);

          it = window->m_resize_callback.next(it);
        }

        break;
      }
      // case ConfigureNotify:
      // {

      // 	for (const auto& i : window.m_move_callback) {
      // 		if (i) {
      // 			i();
      // 		}
      // 	}

      // 	break;
      // }
      case ClientMessage:
      {
        auto window = fan::get_window_by_id(event.xclient.window);

        if (!window) {
          break;
        }

        if (event.xclient.data.l[0] == (long)m_atom_delete_window) {
          for (uintptr_t i = 0; i < window->m_close_callback.size(); i++) {
              window->m_close_callback[i].first(window, window->m_close_callback[i].second);
          }

          window->m_event_flags |= window_t::events::close;

        }

        break;
      }
      case KeyPress:
      {

        auto window = fan::get_window_by_id(event.xkey.window);

        if (!window) {
          break;
        }

        uint16_t key = fan::window_input::convert_keys_to_fan(event.xkey.keycode);

        fan::window_t::window_input_action(window->m_window_handle, key);

        bool repeat = 0;

        window->m_current_key = key;

        auto it = window->m_keys_callback.begin();

        while (it != window->m_keys_callback.end()) {

          window->m_keys_callback.start_safe_next(it);

          window->m_keys_callback[it].first(window, key, repeat ? fan::key_state::repeat : fan::key_state::press, window->m_keys_callback[it].second);

          it = window->m_keys_callback.end_safe_next();
        }

        KeySym keysym;

        char text[32] = {};

        Status status;
        std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
        std::wstring str;

        Xutf8LookupString(window->m_xic, &event.xkey, text, sizeof(text) - 1, &keysym, &status);

        str = converter.from_bytes(text);

        bool found = false;

        for (auto i : banned_keys) {
          if (key == i) {
            found = true;
            break;
          }
        }

        if (str.size() && !found) {
          if (!str.size()) {

            auto it = window->m_text_callback.begin();

            while (it != window->m_text_callback.end()) {

              window->m_text_callback.start_safe_next(it);

              window->m_text_callback[it].first(window, key, window->m_text_callback[it].second);

              it = window->m_text_callback.end_safe_next();
            }
          }
          else {

            auto utf8_str = fan::utf16_to_utf8((wchar_t*)str.data());

            auto it = window->m_text_callback.begin();

            while (it != window->m_text_callback.end()) {

              window->m_text_callback[it].first(window, utf8_str.get_character(0), window->m_text_callback[it].second);

              it = window->m_text_callback.next(it);
            }
          }
        }

        break;
      }
      case KeyRelease:
      {

        auto window = fan::get_window_by_id(event.xkey.window);

        if (!window) {
          break;
        }

        if (XEventsQueued(fan::sys::m_display, QueuedAfterReading)) {
          XEvent nev;
          XPeekEvent(fan::sys::m_display, &nev);

          if (nev.type == KeyPress && nev.xkey.time == event.xkey.time &&
              nev.xkey.keycode == event.xkey.keycode) {
            break;
          }
        }

        const uint16_t key = fan::window_input::convert_keys_to_fan(event.xkey.keycode);

        window->window_input_up(window->m_window_handle, key);


        auto it = window->m_keys_callback.begin();

        while (it != window->m_keys_callback.end()) {
          window->m_keys_callback.start_safe_next(it);

          window->m_keys_callback[it].first(window, key, key_state::release, window->m_keys_callback[it].second);

          it = window->m_keys_callback.end_safe_next();
        }

        break;
      }
      case MotionNotify:
      {
        auto window = fan::get_window_by_id(event.xmotion.window);

        if (!window) {
          break;
        }

        const fan::vec2i position(event.xmotion.x, event.xmotion.y);

        auto mouse_move_position_callback = window->m_mouse_position_callback;

        auto it = mouse_move_position_callback.begin();

        while (it != mouse_move_position_callback.end()) {

          mouse_move_position_callback.start_safe_next(it);

          mouse_move_position_callback[it].first(window, position, mouse_move_position_callback[it].second);

          it = mouse_move_position_callback.end_safe_next();
        }

        window->m_previous_mouse_position = window->m_mouse_position;

        window->m_mouse_position = position;

        break;
      }
      case ButtonPress:
      {

        auto window = fan::get_window_by_id(event.xbutton.window);

        if (!window) {
          break;
        }

        uint16_t key = fan::window_input::convert_keys_to_fan(event.xbutton.button);

        window->window_input_mouse_action(window->m_window_handle, key);

        auto it = window->m_keys_callback.begin();

        while (it != window->m_keys_callback.end()) {
          window->m_keys_callback.start_safe_next(it);

          window->m_keys_callback[it].first(window, key, key_state::press, window->m_keys_callback[it].second);

          it = window->m_keys_callback.end_safe_next();
        }

        break;
      }
      case ButtonRelease:
      {

        auto window = fan::get_window_by_id(event.xbutton.window);

        if (!window) {
          break;
        }

        if (XEventsQueued(fan::sys::m_display, QueuedAfterReading)) {
          XEvent nev;
          XPeekEvent(fan::sys::m_display, &nev);

          if (nev.type == ButtonPress && nev.xbutton.time == event.xbutton.time &&
              nev.xbutton.button == event.xbutton.button) {
            break;
          }
        }

        auto key = fan::window_input::convert_keys_to_fan(event.xbutton.button);
        window->window_input_up(window->m_window_handle, key);

        auto it = window->m_keys_callback.begin();

        while (it != window->m_keys_callback.end()) {

          window->m_keys_callback.start_safe_next(it);
          window->m_keys_callback[it].first(window, key, key_state::release, window->m_keys_callback[it].second);
          it = window->m_keys_callback.end_safe_next();
        }

        break;
      }
      case FocusOut:
      {
        fan::window_t* window = fan::get_window_by_id(event.xfocus.window);

        if (!window) {
          break;
        }

        window->m_focused = false;
        break;
      }
    }
  }

  #endif

  return m_event_flags;
}

void* fan::window_t::get_user_data() const
{
  return m_user_data;
}

void fan::window_t::set_user_data(void* user_data)
{
  m_user_data = user_data;
}

bool fan::window_t::key_pressed(uint16_t key) const
{
  #if defined(fan_platform_windows)

    return GetKeyState(fan::window_input::convert_fan_to_keys(key)) & 0x8000;
  #elif defined(fan_platform_unix)

  return 1;

  #endif
}
