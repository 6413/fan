#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(window/window.h)
#include _FAN_PATH(math/random.h)
#include _FAN_PATH(types/utf_string.h)

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
#elif defined(fan_platform_unix) || defined(fan_platform_android)
static constexpr const char* shared_library = "libGL.so.1";

#endif

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

void fan::erase_window_id(fan::window_handle_t wid)
{
  uint32_t it = window_id_storage.begin();
  while(it != window_id_storage.end()) {
    if (window_id_storage[it].window_handle == wid) {
      window_id_storage.erase(it);
      return;
    }
    it = window_id_storage.next(it);
  }
}

fan::string fan::window_t::get_name() const
{
  return m_name;
}

void fan::window_t::set_name(const fan::string& name)
{

  m_name = name;

  #ifdef fan_platform_windows

  SetWindowTextA(m_window_handle, m_name.c_str());

  #elif defined(fan_platform_unix)

  XStoreName(fan::sys::m_display, m_window_handle, m_name.c_str());
  XSetIconName(fan::sys::m_display, m_window_handle, m_name.c_str());

  #endif

}

void fan::window_t::calculate_delta_time() {
  m_delta_time = (f64_t)fan::time::clock::elapsed(m_current_frame) / 1000000000;
  m_current_frame = fan::time::clock::now();
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
    new_size = fan::sys::get_screen_resolution();
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
    new_size = fan::sys::get_screen_resolution();
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

  const fan::vec2i position = fan::sys::get_screen_resolution() / 2 - new_size / 2;

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
    fan::sys::set_screen_resolution(size);
  }
  else {
    fan::sys::reset_screen_resolution();
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

fan::window_t::buttons_callback_NodeReference_t fan::window_t::add_buttons_callback(mouse_buttons_cb_t function)
{
  auto nr = m_buttons_callback.NewNodeLast();

  m_buttons_callback[nr].data = function;
  return nr;
}

void fan::window_t::remove_buttons_callback(fan::window_t::buttons_callback_NodeReference_t id)
{
  m_buttons_callback.Unlink(id);
  m_buttons_callback.Recycle(id);
}

fan::window_t::keys_callback_NodeReference_t fan::window_t::add_keys_callback(keyboard_keys_cb_t function)
{
  auto nr = m_keys_callback.NewNodeLast();
  m_keys_callback[nr].data = function;
  return nr;
}

void fan::window_t::remove_keys_callback(fan::window_t::keys_callback_NodeReference_t id)
{
  m_keys_callback.Unlink(id);
  m_keys_callback.Recycle(id);
}

fan::window_t::key_callback_NodeReference_t fan::window_t::add_key_callback(uint16_t key, keyboard_state state, keyboard_key_cb_t function)
{
  auto nr = m_key_callback.NewNodeLast();
  m_key_callback[nr].data = keyboard_cb_store_t{ key, state, function, };
  return nr;
}

void fan::window_t::edit_key_callback(fan::window_t::key_callback_NodeReference_t id, uint16_t key, keyboard_state state)
{
  m_key_callback[id].data.key = key;
  m_key_callback[id].data.state = state;
}

void fan::window_t::remove_key_callback(fan::window_t::key_callback_NodeReference_t id)
{
  m_key_callback.Unlink(id);
  m_key_callback.Recycle(id);
}

fan::window_t::text_callback_NodeReference_t fan::window_t::add_text_callback(text_cb_t function)
{
  auto nr = m_text_callback.NewNodeLast();
  m_text_callback[nr].data = function;
  return nr;
}

void fan::window_t::remove_text_callback(fan::window_t::text_callback_NodeReference_t id)
{
  m_text_callback.Unlink(id);
  m_text_callback.Recycle(id);
}

fan::window_t::close_callback_NodeReference_t fan::window_t::add_close_callback(close_cb_t function)
{
  auto nr = m_close_callback.NewNodeLast();
  m_close_callback[nr].data = function;
  return nr;
}

void fan::window_t::remove_close_callback(fan::window_t::close_callback_NodeReference_t id)
{
  m_close_callback.Unlink(id);
  m_close_callback.Recycle(id);
}

fan::window_t::mouse_position_callback_NodeReference_t fan::window_t::add_mouse_move_callback(mouse_move_cb_t function)
{
  auto nr = m_mouse_position_callback.NewNodeLast();
  m_mouse_position_callback[nr].data = function;
  return nr;
}

void fan::window_t::remove_mouse_move_callback(fan::window_t::mouse_position_callback_NodeReference_t id)
{
  m_mouse_position_callback.Unlink(id);
  m_mouse_position_callback.Recycle(id);
}

fan::window_t::resize_callback_NodeReference_t fan::window_t::add_resize_callback(resize_cb_t function) {
  auto nr = m_resize_callback.NewNodeLast();
  m_resize_callback[nr].data = function;
  return nr;
}

void fan::window_t::remove_resize_callback(fan::window_t::resize_callback_NodeReference_t id) {
  m_resize_callback.Unlink(id);
  m_resize_callback.Recycle(id);
}

fan::window_t::move_callback_NodeReference_t fan::window_t::add_move_callback(move_cb_t function)
{
  auto nr = m_move_callback.NewNodeLast();
  m_move_callback[nr].data = function;
  return nr;
}

void fan::window_t::remove_move_callback(fan::window_t::move_callback_NodeReference_t id)
{
  m_move_callback.Unlink(id);
  m_move_callback.Recycle(id);
}

void fan::window_t::set_background_color(const fan::color& color)
{
  m_background_color = color;
}

fan::window_handle_t fan::window_t::get_handle() const
{
  return m_window_handle;
}

// ms
uintptr_t fan::window_t::get_fps(uint32_t frame_update, bool window_name, bool print) {
  auto time_diff = (m_current_frame - m_last_frame) / 1e+9;
  if (time_diff >= 1.0 / frame_update) {
    fan::string fps_info;
    if (window_name || print) {
      f64_t fps, frame_time;
      fps = (1.0 / time_diff) * m_fps_counter;
      frame_time = time_diff / m_fps_counter;
      fps_info.append(
          fan::string("fps: ") +
          fan::to_string(fps) +
          fan::string(" frame time: ") +
          fan::to_string(frame_time) +
          fan::string(" ms")
      );
      m_last_frame = m_current_frame;
      m_fps_counter = 0;
    }
    if (window_name) {
      this->set_name(fps_info.c_str());
    }
    if (print) {
      fan::print(fps_info);
    }
    return m_fps_counter;
  }

  m_fps_counter++;
  return 0;
}

fan::window_t::window_t(const fan::vec2i& window_size, const fan::string& name, uint64_t flags)
{
  m_size = window_size;
  m_mouse_position = 0;
  m_max_fps = 0;
  m_fps_counter = 0;
  m_last_frame = fan::time::clock::now();
  m_current_frame = fan::time::clock::now();
  m_delta_time = 0;
  m_name = name;
  m_flags = flags;
  m_current_key = 0;
  m_reserved_flags = 0;
  m_focused = true;
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

  m_buttons_callback.Open();
  m_keys_callback.Open();
	m_key_callback.Open();
	m_text_callback.Open();
	m_move_callback.Open();
	m_resize_callback.Open();
	m_close_callback.Open();
	m_mouse_position_callback.Open();

  window_id_storage.open();

  initialize_window(name, window_size, flags);

  this->calculate_delta_time();
}

fan::window_t::~window_t()
{
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

void fan::window_t::destroy_window_internal(){
  fan::erase_window_id(this->m_window_handle);

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
}

void fan::window_t::destroy_window()
{
  destroy_window_internal();

  m_buttons_callback.Close();
  m_keys_callback.Close();
	m_key_callback.Close();

	m_text_callback.Close();
  m_move_callback.Close();
  m_resize_callback.Close();
  m_close_callback.Close();
  m_mouse_position_callback.Close();
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

  auto it = fwindow->m_key_callback.GetNodeFirst();

  while (it != fwindow->m_key_callback.dst) {

    fwindow->m_key_callback.StartSafeNext(it);

    if (key != fwindow->m_key_callback[it].data.key || fwindow->m_key_callback[it].data.state == keyboard_state::release) {
      it = fwindow->m_key_callback.EndSafeNext();
      continue;
    }

    keyboard_key_cb_data_t cbd;
    cbd.window = fwindow;
    cbd.key = key;
    fwindow->m_key_callback[it].data.function(cbd);

    it = fwindow->m_key_callback.EndSafeNext();
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

  auto it = fwindow->m_key_callback.GetNodeFirst();

  while (it != fwindow->m_key_callback.dst) {

    fwindow->m_key_callback.StartSafeNext(it);

    if (key != fwindow->m_key_callback[it].data.key || fwindow->m_key_callback[it].data.state == keyboard_state::release) {
      it = fwindow->m_key_callback.EndSafeNext();
      continue;
    }

    keyboard_key_cb_data_t cbd;
    cbd.window = fwindow;
    cbd.key = key;
    fwindow->m_key_callback[it].data.function(cbd);

    it = fwindow->m_key_callback.EndSafeNext();
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

  auto it = fwindow->m_key_callback.GetNodeFirst();

  while (it != fwindow->m_key_callback.dst) {

    fwindow->m_key_callback.StartSafeNext(it);

    if (key != fwindow->m_key_callback[it].data.key || fwindow->m_key_callback[it].data.state == keyboard_state::press) {
      it = fwindow->m_key_callback.EndSafeNext();
      continue;
    }
    
    keyboard_key_cb_data_t cbd;
    cbd.window = fwindow;
    cbd.key = key;
    fwindow->m_key_callback[it].data.function(cbd);

    it = fwindow->m_key_callback.EndSafeNext();
  }
}

void fan::window_t::window_input_action_reset(fan::window_handle_t window, uint16_t key)
{

}

#ifdef fan_platform_windows

static void handle_special(WPARAM wparam, LPARAM lparam, uint16_t& key, bool down) {
 // if (wparam == 0x10 || wparam == 0x11) {
  //  if (down) {
  //    switch (lparam) {
  //      case fan::special_lparam::lshift_lparam_down:
  //      {
  //        key = fan::input::key_left_shift;
  //        break;
  //      }
  //      case fan::special_lparam::rshift_lparam_down:
  //      {
  //        key = fan::input::key_right_shift;
  //        break;
  //      }
  //      case fan::special_lparam::lctrl_lparam_down:
  //      {
  //        key = fan::input::key_left_control;
  //        break;
  //      }
  //      case fan::special_lparam::rctrl_lparam_down:
  //      {
  //        key = fan::input::key_right_control;
  //        break;
  //      }
  //    }
  //  }
  //  else {
  //    switch (lparam) {
  //      case fan::special_lparam::lshift_lparam_up: // ?
  //      {
  //        key = fan::input::key_left_shift;
  //        break;
  //      }z
  //      case fan::special_lparam::rshift_lparam_up:
  //      {
  //        key = fan::input::key_right_shift;
  //        break;
  //      }
  //      case fan::special_lparam::lctrl_lparam_up:
  //      {
  //        key = fan::input::key_left_control;
  //        break;
  //      }
  //      case fan::special_lparam::rctrl_lparam_up: // ? 
  //      {
  //        key = fan::input::key_right_control;
  //        break;
  //      }
  //    }
  //  }
  //}
  //else {
    //fan::print(key, (int)wparam);
    key = fan::window_input::convert_keys_to_fan(wparam);
 // }

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

      auto it = window->m_move_callback.GetNodeFirst();

      while (it != window->m_move_callback.dst) {
        it = it.Next(&window->m_move_callback);
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

      auto it = fwindow->m_resize_callback.GetNodeFirst();

      while (it != fwindow->m_resize_callback.dst) {
        
        resize_cb_data_t cbd;
        cbd.window = fwindow;
        cbd.size = fwindow->m_size;
        fwindow->m_resize_callback[it].data(cbd);

        it = it.Next(&fwindow->m_resize_callback);
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

      for (uint16_t i = fan::first; i != fan::last; i++) {
        if (GetAsyncKeyState(fan::window_input::convert_fan_to_keys(i))) {
          if (i >= fan::mouse_left) {
            auto it = fwindow->m_buttons_callback.GetNodeFirst();
            while (it != fwindow->m_buttons_callback.dst) {
              fwindow->m_buttons_callback.StartSafeNext(it);

              mouse_buttons_cb_data_t cbd;
              cbd.window = fwindow;
              cbd.button = i;
              cbd.state = fan::mouse_state::release;
              fwindow->m_buttons_callback[it].data(cbd);

              it = fwindow->m_buttons_callback.EndSafeNext();
            }
          }
          else {
            auto it = fwindow->m_keys_callback.GetNodeFirst();
            while (it != fwindow->m_keys_callback.dst) {
              fwindow->m_keys_callback.StartSafeNext(it);

              keyboard_keys_cb_data_t cbd;
              cbd.window = fwindow;
              cbd.key = i;
              cbd.state = fan::keyboard_state::release;
              fwindow->m_keys_callback[it].data(cbd);

              it = fwindow->m_keys_callback.EndSafeNext();
            }
          }
        }
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

      auto it = fwindow->m_close_callback.GetNodeFirst();

      while (it != fwindow->m_close_callback.dst) {
        close_cb_data_t cbd;
        cbd.window = fwindow;
        fwindow->m_close_callback[it].data(cbd);
        it = it.Next(&fwindow->m_close_callback);
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

void fan::window_t::initialize_window(const fan::string& name, const fan::vec2i& window_size, uint64_t flags)
{
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

  const fan::vec2i position = fan::sys::get_screen_resolution() / 2 - window_size / 2;

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
    fan::throw_error("failed to initialize window:" + fan::to_string(GetLastError()));
  }

  RAWINPUTDEVICE r_id[2];
  r_id[0].usUsagePage = HID_USAGE_PAGE_GENERIC;
  r_id[0].usUsage = HID_USAGE_GENERIC_MOUSE;
  r_id[0].dwFlags = RIDEV_INPUTSINK;
  r_id[0].hwndTarget = m_window_handle;

  r_id[1].usUsagePage = HID_USAGE_PAGE_GENERIC;
	r_id[1].usUsage = HID_USAGE_GENERIC_KEYBOARD;
	r_id[1].dwFlags = RIDEV_INPUTSINK;
	r_id[1].hwndTarget = m_window_handle;

  BOOL result = RegisterRawInputDevices(r_id, 2, sizeof(RAWINPUTDEVICE));

  if (!result) {
    fan::throw_error("failed to register raw input:" + fan::to_string(result));
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
  fan::sys::open_lib_handle(shared_library, &lib_handle);
  fan::opengl::glx::PFNGLXGETPROCADDRESSPROC glXGetProcAddress = (decltype(glXGetProcAddress))fan::sys::get_lib_proc(&lib_handle, "glXGetProcAddress");
  if (glXGetProcAddress == nullptr) {
    fan::throw_error("failed to initialize glxGetprocAddress");
  }
  fan::sys::close_lib_handle(&lib_handle);
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

  constexpr auto major = 3;
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


  const fan::vec2i position = fan::sys::get_screen_resolution() / 2 - window_size / 2;

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
    GLX_CONTEXT_MINOR_VERSION_ARB, minor,
    GLX_CONTEXT_MAJOR_VERSION_ARB, major,
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

    auto it = m_mouse_position_callback.GetNodeFirst();
    while (it != m_mouse_position_callback.dst) {

      m_mouse_position_callback.StartSafeNext(it);

      mouse_move_cb_data_t cbd;
      cbd.window = this;
      cbd.position = m_mouse_position;
      m_mouse_position_callback[it].data(cbd);

      it = m_mouse_position_callback.EndSafeNext();
    }
  }

  m_previous_mouse_position = m_mouse_position;
  call_mouse_move_cb = false;

  #ifdef fan_platform_windows

  MSG msg{};

  while (PeekMessageW(&msg, m_window_handle, 0, 0, PM_REMOVE))
  {
    switch (msg.message) {
      case WM_SYSKEYDOWN:
      case WM_KEYDOWN:
      {
        auto window = fan::get_window_by_id(msg.hwnd);

        if (!window) {
          break;
        }

        fan::print((msg.lParam >> 16) & 0x1ff);

        uint16_t key;
        //fan::print((int)msg.wParam);
        handle_special(msg.wParam, msg.lParam, key, true);

        bool repeat = msg.lParam & (1 << 30);

        fan::window_t::window_input_action(window->m_window_handle, key);

        window->m_current_key = key;

        auto it = window->m_keys_callback.GetNodeFirst();

        while (it != window->m_keys_callback.dst) {

          window->m_keys_callback.StartSafeNext(it);

          keyboard_keys_cb_data_t cdb;
          cdb.window = window;
          cdb.key = key;
          cdb.state = repeat ? fan::keyboard_state::repeat : fan::keyboard_state::press;
          window->m_keys_callback[it].data(cdb);

          it = window->m_keys_callback.EndSafeNext();
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

            found = false;

            for (auto i : banned_keys) {
              if (fan_key == i) {

                found = true;
                break;
              }
            }

            if (!found) {

              auto src = msg.wParam + (window->m_reserved_flags << 8);
              
              UINT u = VkKeyScan(src);
              UINT high_byte = HIBYTE(u);
              UINT low_byte = LOBYTE(u);
              if (m_prev_text_flag == u) {
                auto it = window->m_text_callback.GetNodeFirst();

                while (it != window->m_text_callback.dst) {

                  fan::window_t::text_cb_data_t d;
                  d.character = src;
                  d.window = window;
                  d.state = fan::keyboard_state::repeat;
                  window->m_text_callback[it].data(d);

                  it = it.Next(&window->m_text_callback);
                }
                break;
              }
              if (m_prev_text_flag) {
                window->m_keymap[fan::key_shift] = 0;
                window->m_keymap[fan::key_control] = 0;
                window->m_keymap[fan::key_alt] = 0;
                window->m_keymap[LOBYTE(m_prev_text_flag)] = false;
              }
              m_prev_text_flag = u;
              window->m_keymap[fan::key_shift] = high_byte & 0x1;
              window->m_keymap[fan::key_control] = high_byte & 0x2;
              window->m_keymap[fan::key_alt] = high_byte & 0x4;
              window->m_keymap[fan::window_input::convert_keys_to_fan(low_byte)] = true;
              m_prev_text = src;
              
             // UTF-8
             // auto utf8_str = fan::utf16_to_utf8((wchar_t*)&src);

             /* uint32_t value = 0;

              for (int i = 0, j = 0; i < utf8_str.size(); i++, j += 0x08) {
                value |= (uint8_t)utf8_str[i] << j;
              }*/

              auto it = window->m_text_callback.GetNodeFirst();

              while (it != window->m_text_callback.dst) {

                fan::window_t::text_cb_data_t d;
                d.character = src;
                d.window = window;
                d.state = fan::keyboard_state::press;
                window->m_text_callback[it].data(d);

                it = it.Next(&window->m_text_callback);
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

        const uint16_t button = fan::input::mouse_left;

        fan::window_t::window_input_mouse_action(window->m_window_handle, button);

        auto it = window->m_buttons_callback.GetNodeFirst();

        while (it != window->m_buttons_callback.dst) {

          mouse_buttons_cb_data_t cbd;
          cbd.window = window;
          cbd.button = button;
          cbd.state = fan::mouse_state::press;
          window->m_buttons_callback[it].data(cbd);

          it = it.Next(&window->m_buttons_callback);
        }

        break;
      }
      case WM_RBUTTONDOWN:
      {
        auto window = fan::get_window_by_id(msg.hwnd);

        if (!window) {
          break;
        }

        const uint16_t button = fan::input::mouse_right;

        fan::window_t::window_input_mouse_action(window->m_window_handle, button);

        auto it = window->m_buttons_callback.GetNodeFirst();

        while (it != window->m_buttons_callback.dst) {

          mouse_buttons_cb_data_t cbd;
          cbd.window = window;
          cbd.button = button;
          cbd.state = fan::mouse_state::press;
          window->m_buttons_callback[it].data(cbd);

          it = it.Next(&window->m_buttons_callback);
        }

        break;
      }
      case WM_MBUTTONDOWN:
      {
        auto window = fan::get_window_by_id(msg.hwnd);

        if (!window) {
          break;
        }

        const uint16_t button = fan::input::mouse_middle;

        fan::window_t::window_input_mouse_action(window->m_window_handle, button);

        auto it = window->m_buttons_callback.GetNodeFirst();

        while (it != window->m_buttons_callback.dst) {

          mouse_buttons_cb_data_t cbd;
          cbd.window = window;
          cbd.button = button;
          cbd.state = fan::mouse_state::press;
          window->m_buttons_callback[it].data(cbd);

          it = it.Next(&window->m_buttons_callback);
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

        do {
            if (key == fan::key_control || key == fan::key_alt || key == fan::key_shift) {
              break;
            }
            auto it = window->m_text_callback.GetNodeFirst();

            while (it != window->m_text_callback.dst) {

              fan::window_t::text_cb_data_t d;
              d.character = m_prev_text;
              d.window = window;
              d.state = fan::keyboard_state::release;
              // reset flag
              m_prev_text_flag = 0;
              window->m_text_callback[it].data(d);

              it = it.Next(&window->m_text_callback);
            }
          } while (0);

        window_input_up(window->m_window_handle, key);

        auto it = window->m_keys_callback.GetNodeFirst();

        while (it != window->m_keys_callback.dst) {
          window->m_keys_callback.StartSafeNext(it);

          keyboard_keys_cb_data_t cbd;
          cbd.window = window;
          cbd.key = key;
          cbd.state = fan::keyboard_state::release;
          window->m_keys_callback[it].data(cbd);

          it = window->m_keys_callback.EndSafeNext();
        }

        break;
      }
      case WM_MOUSEWHEEL:
      {
        auto zDelta = GET_WHEEL_DELTA_WPARAM(msg.wParam);

        auto window = fan::get_window_by_id(msg.hwnd);

        fan::window_t::window_input_mouse_action(window->m_window_handle, zDelta < 0 ? fan::input::mouse_scroll_down : fan::input::mouse_scroll_up);

        auto it = window->m_buttons_callback.GetNodeFirst();

        while (it != window->m_buttons_callback.dst) {

          mouse_buttons_cb_data_t cbd;
          cbd.window = window;
          cbd.button = zDelta < 0 ? fan::input::mouse_scroll_down : fan::input::mouse_scroll_up;
          cbd.state = fan::mouse_state::press;
          window->m_buttons_callback[it].data(cbd);

          it = it.Next(&window->m_buttons_callback);
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

            auto it = window->m_buttons_callback.GetNodeFirst();

            while (it != window->m_buttons_callback.dst) {

              mouse_buttons_cb_data_t cbd;
              cbd.window = window;
              cbd.button = fan::input::mouse_left;
              cbd.state = fan::mouse_state::release;
              window->m_buttons_callback[it].data(cbd);

              it = it.Next(&window->m_buttons_callback);
            }

            window_input_up(window->m_window_handle, fan::input::mouse_left); allow_outside = false;
          }

          else if (fan::is_flag(raw->data.mouse.usButtonFlags, RI_MOUSE_MIDDLE_BUTTON_UP)) {

            auto it = window->m_buttons_callback.GetNodeFirst();

            while (it != window->m_buttons_callback.dst) {

              mouse_buttons_cb_data_t cbd;
              cbd.window = window;
              cbd.button = fan::input::mouse_middle;
              cbd.state = fan::mouse_state::release;
              window->m_buttons_callback[it].data(cbd);

              it = it.Next(&window->m_buttons_callback);
            }

            window_input_up(window->m_window_handle, fan::input::mouse_middle); allow_outside = false;
          }

          else if (fan::is_flag(raw->data.mouse.usButtonFlags, RI_MOUSE_RIGHT_BUTTON_UP)) {

            auto it = window->m_buttons_callback.GetNodeFirst();

            while (it != window->m_buttons_callback.dst) {

              mouse_buttons_cb_data_t cbd;
              cbd.window = window;
              cbd.button = fan::input::mouse_right;
              cbd.state = fan::mouse_state::release;
              window->m_buttons_callback[it].data(cbd);

              it = it.Next(&window->m_buttons_callback);
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

        auto it = window->m_resize_callback.GetNodeFirst();

        while (it != window->m_resize_callback.dst) {

          resize_cb_data_t cdb;
          cdb.window = window;
          cdb.size = window->m_size;
          window->m_resize_callback[it].data(cdb);

          it = it.Next(&window->m_resize_callback);
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

          auto it = window->m_close_callback.GetNodeFirst();
          while (it != window->m_close_callback.dst) {
            close_cb_data_t cbd;
            cbd.window = window;
            window->m_close_callback[it].data(cbd);
            it = it.Next(&window->m_close_callback);
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

        fan::print((int)event.xkey.keycode);

        uint16_t key = fan::window_input::convert_keys_to_fan(event.xkey.keycode);
        fan::print(key, event.xkey.type);
        fan::window_t::window_input_action(window->m_window_handle, key);

        bool repeat = 0;

        window->m_current_key = key;

        auto it = window->m_keys_callback.GetNodeFirst();

        while (it != window->m_keys_callback.dst) {

          window->m_keys_callback.StartSafeNext(it);

          keyboard_keys_cb_data_t cdb;
          cdb.window = window;
          cdb.key = key;
          cdb.state = repeat ? fan::keyboard_state::repeat : fan::keyboard_state::press;
          window->m_keys_callback[it].data(cdb);

          it = window->m_keys_callback.EndSafeNext();
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

            auto it = window->m_text_callback.GetNodeFirst();

            while (it != window->m_text_callback.dst) {

              window->m_text_callback.StartSafeNext(it);

              text_cb_data_t cdb;
              cdb.window = window;
              cdb.character = str[0];
              window->m_text_callback[it].data(cdb);

              it = window->m_text_callback.EndSafeNext();
            }
          }
          else {

            //auto utf8_str = fan::utf16_to_utf8((wchar_t*)str.data());
            //?
            auto it = window->m_text_callback.GetNodeFirst();

            while (it != window->m_text_callback.dst) {

              fan::window_t::text_cb_data_t d;
              d.window = window;
              d.character = str[0];
              window->m_text_callback[it].data(d);

              it = it.Next(&window->m_text_callback);
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


        auto it = window->m_keys_callback.GetNodeFirst();

        while (it != window->m_keys_callback.dst) {
          window->m_keys_callback.StartSafeNext(it);

          keyboard_keys_cb_data_t cdb;
          cdb.window = window;
          cdb.key = key;
          cdb.state = fan::keyboard_state::release;
          window->m_keys_callback[it].data(cdb);

          it = window->m_keys_callback.EndSafeNext();
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

        auto it = mouse_move_position_callback.GetNodeFirst();

        while (it != mouse_move_position_callback.dst) {

          mouse_move_position_callback.StartSafeNext(it);

          mouse_move_cb_data_t cdb;
          cdb.window = window;
          cdb.position = position;
          mouse_move_position_callback[it].data(cdb);

          it = mouse_move_position_callback.EndSafeNext();
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

        uint16_t button = fan::window_input::convert_keys_to_fan(event.xbutton.button);

        window->window_input_mouse_action(window->m_window_handle, button);

        auto it = window->m_buttons_callback.GetNodeFirst();

        while (it != window->m_buttons_callback.dst) {

          mouse_buttons_cb_data_t cbd;
          cbd.window = window;
          cbd.button = button;
          cbd.state = fan::mouse_state::press;
          window->m_buttons_callback[it].data(cbd);

          it = it.Next(&window->m_buttons_callback);
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

        auto button = fan::window_input::convert_keys_to_fan(event.xbutton.button);
        window->window_input_up(window->m_window_handle, button);

        auto it = window->m_buttons_callback.GetNodeFirst();

        while (it != window->m_buttons_callback.dst) {

          mouse_buttons_cb_data_t cbd;
          cbd.window = window;
          cbd.button = button;
          cbd.state = fan::mouse_state::release;
          window->m_buttons_callback[it].data(cbd);

          it = it.Next(&window->m_buttons_callback);
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

bool fan::window_t::key_pressed(uint16_t key) const
{
  #if defined(fan_platform_windows)

    return GetKeyState(fan::window_input::convert_fan_to_keys(key)) & 0x8000;
  #elif defined(fan_platform_unix)

  return 1;

  #endif
}

#ifdef fan_platform_windows

int main();

int WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd) {
  return main();
}

#endif