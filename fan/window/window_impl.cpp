module;

#include <fan/utility.h>

#if defined(FAN_VULKAN)
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
#include <string>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <chrono>

module fan.window;

namespace fan {

  bool& init_manager_t::initialized() {
    static bool instance = false;
    return instance;
  }

  void init_manager_t::initialize() {
    if (initialized()) {
      return;
    }
    if (!glfwInit()) {
      fan::throw_error("failed to initialize context");
    }
    initialized() = true;
  }

  void init_manager_t::uninitialize() {
    if (initialized() == false) {
      return;
    }
    glfwTerminate();
    initialized() = false;
  }

  init_manager_t::cleaner_t::cleaner_t() {
    init_manager_t::initialize();
  }

  init_manager_t::cleaner_t::~cleaner_t() {
    init_manager_t::uninitialize();
  }

  init_manager_t::cleaner_t& init_manager_t::cleaner() {
    static cleaner_t instance;
    return instance;
  }

  fan::init_manager_t::cleaner_t& _cleaner = fan::init_manager_t::cleaner();
}

namespace fan::window {
  void mouse_button_callback(GLFWwindow* wnd, int button, int action, int mods) {
    fan::window_t* window = (fan::window_t*)glfwGetWindowUserPointer(wnd);
    window->key_states[button] = action;
    auto it = window->m_buttons_callback.GetNodeFirst();

    while (it != window->m_buttons_callback.dst) {
      fan::window_t::buttons_data_t cbd;
      cbd.window = window;
      cbd.button = button;
      cbd.state = action;
      cbd.position = window->get_mouse_position();
      window->m_buttons_callback[it](cbd);

      it = it.Next(&window->m_buttons_callback);
    }
  }

  void keyboard_keys_callback(GLFWwindow* wnd, int key, int scancode, int action, int mods) {
    if (key == -1) {
      return;
    }

    fan::window_t* window = (fan::window_t*)glfwGetWindowUserPointer(wnd);
    window->key_states[key] = action;

    {
      auto it = window->m_keys_callback.GetNodeFirst();

      while (it != window->m_keys_callback.dst) {
        fan::window_t::keys_data_t cbd;
        cbd.window = window;
        cbd.key = key;
        cbd.state = static_cast<fan::keyboard_state_t>(action);
        cbd.scancode = scancode;
        window->m_keys_callback[it](cbd);
        it = it.Next(&window->m_keys_callback);
      }
    }
    {
      auto it = window->m_key_callback.GetNodeFirst();

      while (it != window->m_key_callback.dst) {

        fan::window_t::key_data_t cbd;
        cbd.window = window;
        window->m_key_callback[it](cbd);
        it = it.Next(&window->m_key_callback);
      }
    }
  }

  void text_callback(GLFWwindow* wnd, unsigned int codepoint) {
    fan::window_t* window = (fan::window_t*)glfwGetWindowUserPointer(wnd);
    auto it = window->m_text_callback.GetNodeFirst();

    while (it != window->m_text_callback.dst) {
      fan::window_t::text_data_t cbd;
      cbd.window = window;
      cbd.character = codepoint;
      cbd.state = fan::keyboard_state::press;
      window->m_text_callback[it](cbd);

      it = it.Next(&window->m_text_callback);
    }
  }

  void mouse_position_callback(GLFWwindow* wnd, double xpos, double ypos) {
    fan::window_t* window = (fan::window_t*)glfwGetWindowUserPointer(wnd);
    {
      auto it = window->m_mouse_position_callback.GetNodeFirst();

      fan::window_t::mouse_move_data_t cbd;
      cbd.window = window;
      cbd.position = fan::vec2d(xpos, ypos);

      while (it != window->m_mouse_position_callback.dst) {
        window->m_mouse_position_callback[it](cbd);

        it = it.Next(&window->m_mouse_position_callback);
      }
    }
    if (window->previous_mouse_position.x == -0xfff && window->previous_mouse_position.y == -0xfff) {
      window->previous_mouse_position = fan::vec2d(xpos, ypos);
    }
    {
      fan::window_t::mouse_motion_data_t cbd;
      cbd.window = window;
      cbd.motion = fan::vec2d(xpos, ypos) - window->previous_mouse_position;
      auto it = window->m_mouse_motion_callback.GetNodeFirst();
      while (it != window->m_mouse_motion_callback.dst) {
        window->m_mouse_motion_callback[it](cbd);

        it = it.Next(&window->m_mouse_motion_callback);
      }
    }
    window->previous_mouse_position = fan::vec2d(xpos, ypos);
  }

  void close_callback(GLFWwindow* wnd) {
    fan::window_t* window = (fan::window_t*)glfwGetWindowUserPointer(wnd);
    auto it = window->m_close_callback.GetNodeFirst();

    while (it != window->m_close_callback.dst) {
      fan::window_t::close_data_t cbd;
      cbd.window = window;
      window->m_close_callback[it](cbd);

      it = it.Next(&window->m_close_callback);
    }
  }

  void resize_callback(GLFWwindow* wnd, int width, int height) {
    fan::window_t* window = (fan::window_t*)glfwGetWindowUserPointer(wnd);

    auto it = window->m_resize_callback.GetNodeFirst();
    while (it != window->m_resize_callback.dst) {
      fan::window_t::resize_data_t cbd;
      cbd.window = window;
      cbd.size = fan::vec2i(width, height);
      window->m_resize_callback[it](cbd);

      it = it.Next(&window->m_resize_callback);
    }
  }

  void move_callback(GLFWwindow* wnd, int xpos, int ypos) {
    fan::window_t* window = (fan::window_t*)glfwGetWindowUserPointer(wnd);

    auto it = window->m_move_callback.GetNodeFirst();
    while (it != window->m_move_callback.dst) {
      fan::window_t::move_data_t cbd;
      cbd.window = window;
      cbd.position = fan::vec2i(xpos, ypos);
      window->m_move_callback[it](cbd);
      it = it.Next(&window->m_move_callback);
    }
  }


  void scroll_callback(GLFWwindow* wnd, double xoffset, double yoffset) {
    fan::window_t* window = (fan::window_t*)glfwGetWindowUserPointer(wnd);

    auto it = window->m_buttons_callback.GetNodeFirst();

    int button = yoffset < 0 ? fan::input::mouse_scroll_down : fan::input::mouse_scroll_up;

    window->key_states[button] = (int)fan::mouse_state::press;

    while (it != window->m_buttons_callback.dst) {
      fan::window_t::buttons_data_t cbd;
      cbd.window = window;
      cbd.button = button;
      cbd.state = fan::mouse_state::press;
      cbd.position = window->get_mouse_position();
      window->m_buttons_callback[it](cbd);

      it = it.Next(&window->m_buttons_callback);
    }
  }

  void window_focus_callback(GLFWwindow* wnd, int focused) {
    fan::window_t* window = (fan::window_t*)glfwGetWindowUserPointer(wnd);
    for (int i = 0; i < GLFW_KEY_LAST; i++) {
      if (window->key_states[i] != -1) {
        window->key_states[i] = GLFW_RELEASE;
      }
    }
  }
}

namespace fan {

  void window_t::open(std::uint64_t flags) {
    fan::window_t::open(default_window_size, -1, default_window_name, flags, mode::windowed);
  }

  void window_t::open(fan::vec2i window_size, fan::vec2i window_pos, const std::string& name, std::uint64_t flags, int open_mode) {
    this->flags = flags;
    std::fill(key_states, key_states + std::size(key_states), -1);
    std::fill(prev_key_states, prev_key_states + std::size(prev_key_states), -1);

    if (window_size.x == -1 && window_size.y == -1) {
      const GLFWvidmode* mode0 = glfwGetVideoMode(glfwGetPrimaryMonitor());
      window_size = resolutions[current_resolution];
    }

  #if FAN_DEBUG >= fan_debug_high
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
  #endif

    if (m_antialiasing_samples > 0) {
      glfwWindowHint(GLFW_SAMPLES, m_antialiasing_samples);
    }

    glfwWindowHint(GLFW_VISIBLE, !(flags & flags::hidden));

    if (flags & flags::transparent) {
      glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
      glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE);
    }

    if (renderer == renderer_t::vulkan) {
      glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    }
    else if (renderer == renderer_t::opengl) {
      glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    }

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);

    int mx, my;
    glfwGetMonitorPos(monitor, &mx, &my);

    int w = window_size.x;
    int h = window_size.y;
    int x = mx;
    int y = my;
    GLFWmonitor* use_mon = nullptr;

    if (open_mode == mode::windowed) {

      if (window_pos.x != -1 && window_pos.y != -1) {
        x = window_pos.x;
        y = window_pos.y;
      }
      else {
        x = mx + (mode->width - w) / 2;
        y = my + (mode->height - h) / 2;
      }

    #if defined(fan_platform_windows)
      if (y < my + 31) {
        y = my + 31;
      }
    #endif
      x = std::clamp(x, mx, mx + mode->width - w);
      y = std::clamp(y, my + 31, my + mode->height - h);
    }
    else if (open_mode == mode::full_screen) {
      x = 0;
      y = 0;
      w = mode->width;
      h = mode->height;
      use_mon = monitor;
    }
    else if (open_mode == mode::borderless) {
      glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
      w = mode->width;
      h = mode->height;
      x = mx;
      y = my;
    }

    glfw_window = glfwCreateWindow(w, h, name.c_str(), use_mon, nullptr);

    if (use_mon == nullptr) {
      glfwSetWindowPos(glfw_window, x, y);
    }

    if (glfw_window == nullptr) {
      glfwTerminate();
      fan::throw_error("failed to create window");
    }

    glfwSetWindowUserPointer(glfw_window, this);

  #if defined(fan_platform_windows)
    apply_window_theme();
    if (flags & flags::topmost) {
      set_topmost();
    }
    if (flags & flags::click_through) {
      make_click_through();
    }
  #endif

    GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();
    fan::vec2 screen_size = fan::vec2(mode->width, mode->height);
    if (renderer == renderer_t::opengl) {
      glfwMakeContextCurrent(glfw_window);
    }
    glfwSetMouseButtonCallback(glfw_window, fan::window::mouse_button_callback);
    glfwSetKeyCallback(glfw_window, fan::window::keyboard_keys_callback);
    glfwSetCharCallback(glfw_window, fan::window::text_callback);
    glfwSetWindowPosCallback(glfw_window, fan::window::move_callback);
    glfwSetWindowSizeCallback(glfw_window, fan::window::resize_callback);
    glfwSetWindowCloseCallback(glfw_window, fan::window::close_callback);
    glfwSetCursorPosCallback(glfw_window, fan::window::mouse_position_callback);
    glfwSetScrollCallback(glfw_window, fan::window::scroll_callback);
    glfwInitHint(GLFW_JOYSTICK_HAT_BUTTONS, GLFW_TRUE);
    display_mode = open_mode;
  }

  void window_t::close() {
    if (glfw_window) {
      glfwMakeContextCurrent(nullptr);
      glfwDestroyWindow(glfw_window);
      glfw_window = nullptr;
    }
  }

  void window_t::make_context_current() {
    glfwMakeContextCurrent(*this);
  }

  void window_t::handle_key_states(){
    if (key_states[fan::mouse_scroll_up] == GLFW_PRESS) {
      key_states[fan::mouse_scroll_up] = GLFW_RELEASE;
    }
    if (key_states[fan::mouse_scroll_down] == GLFW_PRESS) {
      key_states[fan::mouse_scroll_down] = GLFW_RELEASE;
    }
    for (std::size_t i = 0; i < std::size(key_states); ++i) {
      int curr = key_states[i];
      int prev = prev_key_states[i];
      if (curr == GLFW_PRESS) {
        if (prev == GLFW_PRESS || prev == GLFW_REPEAT) {
          key_states[i] = GLFW_REPEAT;
        }
        else {
          key_states[i] = GLFW_PRESS;
        }
      }
      else if (curr == GLFW_RELEASE) {
        if (prev == GLFW_PRESS || prev == GLFW_REPEAT) {
          key_states[i] = GLFW_RELEASE;
        }
        else {
          key_states[i] = -1;
        }
      }
    }

    int present = glfwJoystickPresent(GLFW_JOYSTICK_1);
    if (present) {
      int button_count;
      const unsigned char* buttons = glfwGetJoystickButtons(GLFW_JOYSTICK_1, &button_count);

      for (int i = fan::gamepad_a; i <= fan::gamepad_left; ++i) {
        int index = i - fan::gamepad_a;
        if (index >= button_count) {
          continue;
        }

        int prev = prev_key_states[i];
        int physical = buttons[index];

        if (physical == GLFW_PRESS) {
          if (prev == GLFW_PRESS || prev == GLFW_REPEAT) {
            key_states[i] = GLFW_REPEAT;
          }
          else {
            key_states[i] = GLFW_PRESS;
          }
        }
        else {
          if (prev == GLFW_PRESS || prev == GLFW_REPEAT) {
            key_states[i] = GLFW_RELEASE;
          }
          else {
            key_states[i] = -1;
          }
        }
      }
    }
    {
      auto it = m_mouse_down_callbacks.GetNodeFirst();
      while (it != m_mouse_down_callbacks.dst) {
        m_mouse_down_callbacks.StartSafeNext(it);

        for (int b = 0; b < 3; ++b) {
          int st = key_states[b];
          if (st == fan::keyboard_state::press || st == fan::keyboard_state::repeat) {
            buttons_data_t cbd;
            cbd.window = this;
            cbd.position = get_mouse_position();
            cbd.button = b;
            cbd.state = st;
            m_mouse_down_callbacks[it](cbd);
          }
        }

        it = m_mouse_down_callbacks.EndSafeNext();
      }
    }
    {
      auto it = m_key_down_callbacks.GetNodeFirst();
      while (it != m_key_down_callbacks.dst) {
        m_key_down_callbacks.StartSafeNext(it);

        for (int k = fan::key_first; k <= fan::input_last; ++k) {
          int st = key_states[k];
          if (st == fan::keyboard_state::press || st == fan::keyboard_state::repeat) {
            fan::window_t::keys_data_t cbd;
            cbd.window = this;
            cbd.scancode = fan::window::input::convert_fan_to_scancode(k);
            cbd.key = k;
            cbd.state = (decltype(cbd.state))st;
            m_key_down_callbacks[it](cbd);
          }
        }

        it = m_key_down_callbacks.EndSafeNext();
      }
    }
  }


  uint32_t window_t::handle_events() {
    f64_t current_frame_time = glfwGetTime();
    m_delta_time = current_frame_time - last_frame_time;
    /*if (m_delta_time >= 0.3) {
    #if FAN_DEBUG >= 4
        fan::print("framerate too low, overriding delta time");
    #endif
        m_delta_time = 1.0 / 256.0;
    }*/
    last_frame_time = current_frame_time;

    std::memcpy(prev_key_states, key_states, sizeof(key_states));
    glfwPollEvents();
    handle_key_states();
    return 0;
  }

  window_t::key_handle_t window_t::add_key_callback(int key, keyboard_state_t st,
    std::function<void(const key_data_t&)> fn)
  {
    using handle_t = window_t::key_handle_t;
    using fn_t = typename handle_t::fn_t;
    using add_fn = typename handle_t::add_fn;
    using rem_fn = typename handle_t::remove_fn;

    add_fn add = [key, st](window_t* w, fn_t cb) {
      auto nr = w->m_key_callback.NewNodeLast();
      w->m_key_callback[nr] = [cb, key, st](const key_data_t& d) {
        if (d.window->key_state(key) == st) {
          cb(d.window, d);
        }
      };
      return nr;
    };
    rem_fn remove = [](window_t* w, key_callback_NodeReference_t nr) {
      w->m_key_callback.unlrec(nr);
    };
    return handle_t(
      this,
      std::move(add),
      std::move(remove),
      [fn](window_t*, const key_data_t& d) { fn(d); }
    );
  }

  window_t::buttons_handle_t window_t::on_mouse_click(uint16_t button, buttons_cb_t fn) {
    return add_buttons_callback([=](const buttons_data_t& d) {
      if (d.button == button && d.state == fan::mouse_state::press) {
        fn(d);
      }
    });
  }

  window_t::mouse_down_handle_t window_t::on_mouse_down(uint16_t button, buttons_cb_t fn) {
    using handle_t = window_t::mouse_down_handle_t;
    using fn_t = typename handle_t::fn_t;
    using add_fn = typename handle_t::add_fn;
    using rem_fn = typename handle_t::remove_fn;

    add_fn add = [button](window_t* w, fn_t cb) {
      auto nr = w->m_mouse_down_callbacks.NewNodeLast();
      w->m_mouse_down_callbacks[nr] = [cb, button](const buttons_data_t& d) {
        int ks = d.window->key_state(button);
        if (ks == fan::mouse_state::press || ks == fan::mouse_state::repeat) {
          cb(d.window, d);
        }
        };
      return nr;
      };
    rem_fn remove = [](window_t* w, mouse_down_callbacks_NodeReference_t nr) {
      w->m_mouse_down_callbacks.unlrec(nr);
      };
    return handle_t(
      this,
      std::move(add),
      std::move(remove),
      [fn](window_t*, const buttons_data_t& d) { fn(d); }
    );
  }

  window_t::buttons_handle_t window_t::on_mouse_up(uint16_t button, buttons_cb_t fn) {
    return add_buttons_callback([=](const buttons_data_t& d) {
      if (d.button == button && d.state == fan::mouse_state::release) {
        fn(d);
      }
      });
  }

  window_t::key_handle_t window_t::on_key_click(int key, key_cb_t fn) {
    return add_key_callback(key, fan::keyboard_state::press, fn);
  }

  window_t::key_handle_t window_t::on_key_down(int key, key_cb_t fn) {
    return add_key_callback(key, fan::keyboard_state::repeat, fn);
  }

  window_t::key_handle_t window_t::on_key_up(int key, key_cb_t fn) {
    return add_key_callback(key, fan::keyboard_state::release, fn);
  }

  window_t::mouse_move_handle_t window_t::on_mouse_move(mouse_move_cb_t fn) {
    return add_mouse_move_callback(fn);
  }

  window_t::resize_handle_t window_t::on_resize(resize_cb_t fn) {
    return add_resize_callback(fn);
  }

  fan::vec2i window_t::get_size() const {
    fan::vec2i window_size;
    glfwGetWindowSize(glfw_window, &window_size.x, &window_size.y);
    return window_size;
  }

  void window_t::set_size(const fan::vec2i& window_size) {
    glfwSetWindowSize(glfw_window, window_size.x, window_size.y);
  }

  fan::vec2 window_t::get_position() const {
    int xpos, ypos;
    glfwGetWindowPos(glfw_window, &xpos, &ypos);
    return fan::vec2(xpos, ypos);
  }

  void window_t::set_position(const fan::vec2& position) {
    glfwSetWindowPos(glfw_window, position.x, position.y);
  }

  GLFWmonitor* window_t::get_current_monitor() {
    int window_x, window_y, window_width, window_height;
    glfwGetWindowPos(glfw_window, &window_x, &window_y);
    glfwGetWindowSize(glfw_window, &window_width, &window_height);

    int monitor_count;
    GLFWmonitor** monitors = glfwGetMonitors(&monitor_count);

    GLFWmonitor* best_monitor = nullptr;
    int best_area = 0;

    for (int i = 0; i < monitor_count; i++) {
      const GLFWvidmode* mode = glfwGetVideoMode(monitors[i]);
      int monitor_x, monitor_y;
      glfwGetMonitorPos(monitors[i], &monitor_x, &monitor_y);

      int overlap_x = std::max(0, std::min(window_x + window_width, monitor_x + mode->width) - std::max(window_x, monitor_x));
      int overlap_y = std::max(0, std::min(window_y + window_height, monitor_y + mode->height) - std::max(window_y, monitor_y));
      int overlap_area = overlap_x * overlap_y;

      if (overlap_area > best_area) {
        best_area = overlap_area;
        best_monitor = monitors[i];
      }
    }

    return best_monitor ? best_monitor : glfwGetPrimaryMonitor();
  }

  fan::vec2 window_t::get_primary_monitor_resolution() {
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    return { mode->width, mode->height };
  }

  fan::vec2 window_t::get_current_monitor_resolution() {
    GLFWmonitor* monitor = get_current_monitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    return { mode->width, mode->height };
  }

  void window_t::set_windowed() {
    GLFWmonitor* m = get_current_monitor();
    const GLFWvidmode* vm = glfwGetVideoMode(m);

    fan::vec2i size = get_size();

    int mx, my;
    glfwGetMonitorPos(m, &mx, &my);

    glfwSetWindowAttrib(glfw_window, GLFW_DECORATED, GLFW_TRUE);
    glfwPollEvents();

    int l, top, r, b;
    glfwGetWindowFrameSize(glfw_window, &l, &top, &r, &b);
    if (top == 0) top = 31;

    // shrink if too large to fit with title bar
    size.y = std::min(size.y, vm->height - top);
    size.x = std::min(size.x, vm->width);

    // center
    int x = mx + (vm->width - size.x) / 2;
    int y = my + (vm->height - size.y) / 2;

    x = std::clamp(x, mx, mx + vm->width - size.x);
    y = std::clamp(y, my + top, my + vm->height - size.y);

    glfwSetWindowMonitor(
      glfw_window,
      nullptr,
      x,
      y,
      size.x,
      size.y,
      vm->refreshRate
    );

    display_mode = mode::windowed;
  }

  void window_t::set_fullscreen() {
    if (display_mode == (uint8_t)mode::full_screen) {
      return;
    }

    GLFWmonitor* monitor = get_current_monitor();
    const GLFWvidmode* vm = glfwGetVideoMode(monitor);

    glfwSetWindowMonitor(
      glfw_window,
      monitor,
      0,
      0,
      vm->width,
      vm->height,
      vm->refreshRate
    );

    glfwSetWindowAttrib(glfw_window, GLFW_DECORATED, GLFW_FALSE);

    display_mode = (uint8_t)mode::full_screen;
  }

  void window_t::set_borderless() {
    if (display_mode == (uint8_t)mode::borderless) {
      return;
    }

    glfwSetWindowAttrib(glfw_window, GLFW_DECORATED, GLFW_FALSE);

    fan::vec2i size = get_size();
    fan::vec2 pos = get_position();

    GLFWmonitor* monitor = get_current_monitor();
    const GLFWvidmode* vm = glfwGetVideoMode(monitor);

    int mx, my;
    glfwGetMonitorPos(monitor, &mx, &my);

    glfwSetWindowMonitor(
      glfw_window,
      nullptr,
      mx,
      my,
      vm->width,
      vm->height,
      vm->refreshRate
    );


    display_mode = (uint8_t)mode::borderless;
  }


  void window_t::set_cursor(int flag) {
    glfwSetInputMode(*this, GLFW_CURSOR, flag == 0 ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
  }

  void window_t::toggle_cursor() {
    bool disabled = glfwGetInputMode(*this, GLFW_CURSOR) != GLFW_CURSOR_DISABLED;
    set_cursor(disabled);
  }

  void window_t::set_display_mode(int mode) {
    switch (mode) {
    case mode::windowed: {
      set_windowed();
      break;
    }
    case mode::full_screen: {
      set_fullscreen();
      break;
    }
    case mode::borderless: {
      set_borderless();
      break;
    }
    default: {
      fan::throw_error("invalid mode");
      break;
    }
    }
  }

  fan::vec2d window_t::get_mouse_position() const {
    fan::vec2d mouse_pos;
    glfwGetCursorPos(glfw_window, &mouse_pos.x, &mouse_pos.y);
    return mouse_pos;
  }

  int window_t::key_state(int key) const {
    return key_states[key];
  }

  bool window_t::key_pressed(int key, int press) const {
    return glfwGetKey(glfw_window, key) == press;
  }

  fan::vec2 window_t::get_gamepad_axis(int key) const {
    fan::vec2 axis = 0;

    int axes_count;
    const float* axes = glfwGetJoystickAxes(GLFW_JOYSTICK_1, &axes_count);
    if (!axes || axes_count == 0) {
      return axis;
    }

    auto norm_trigger = [](f32_t v) {
      if (v <= -1.f) {
        return 0.f;
      }
      return (v + 1.f) * 0.5f;
    };

    switch (key) {
    case fan::gamepad_left_thumb:
    {
      if (axes_count > 1) {
        axis.x = axes[0];
        axis.y = axes[1];
      }
      break;
    }
    case fan::gamepad_right_thumb:
    {
      if (axes_count > 3) {
        axis.x = axes[2];
        axis.y = axes[3];
      }
      break;
    }
    case fan::gamepad_l2:
    {
      if (axes_count > 4) {
        axis.x = norm_trigger(axes[4]);
      }
      break;
    }
    case fan::gamepad_r2:
    {
      if (axes_count > 5) {
        axis.x = norm_trigger(axes[5]);
      }
      break;
    }
    }

    return axis;
  }

  uint8_t window_t::get_antialiasing() const {
    return m_antialiasing_samples;
  }
  void window_t::set_antialiasing(int samples) {
    if (samples < 0) {
      samples = 0;
    }

    m_antialiasing_samples = samples;

    if (glfw_window != nullptr) {
      fan::throw_error_impl("Call before making window");
    }
  }
  void window_t::set_name(const std::string& name) {
    glfwSetWindowTitle(glfw_window, name.c_str());
  }
  void window_t::set_icon(const fan::image::info_t& icon_info) {
    GLFWimage icon;
    icon.width = icon_info.size.x;
    icon.height = icon_info.size.y;
    icon.pixels = (decltype(icon.pixels))icon_info.data;
    glfwSetWindowIcon(glfw_window, 1, &icon);
  }

  void window_t::swap_buffers() {
    glfwSwapBuffers(glfw_window);
  }

#if defined(fan_platform_windows)
  //---------------------------Windows specific code---------------------------

  HWND window_t::get_win32_handle() {
    return glfwGetWin32Window(glfw_window);
  }

  void window_t::set_topmost() {
    SetWindowPos(get_win32_handle(), HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
  }
  void window_t::make_click_through() {
    auto handle = get_win32_handle();
    LONG exStyle = GetWindowLong(handle, GWL_EXSTYLE);
    SetWindowLong(handle, GWL_EXSTYLE, exStyle | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW);
  }

  void window_t::initialize_dark_mode() {
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
  bool window_t::is_high_contrast() const {
    HIGHCONTRASTW high_contrast = {sizeof(high_contrast)};
    if (SystemParametersInfoW(SPI_GETHIGHCONTRAST, sizeof(high_contrast), &high_contrast, FALSE)) {
      return high_contrast.dwFlags & HCF_HIGHCONTRASTON;
    }
    return false;
  }

  void window_t::apply_window_theme() {
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
}

bool fan::window_t::is_key_clicked(int key) {
  if (!this) return false;
  if (key < 0 || key > fan::input_last) return false;
  return key_states[key] == fan::keyboard_state::press &&
    prev_key_states[key] != fan::keyboard_state::press;
}
bool fan::window_t::is_key_down(int key) {
  if (!this) return false;
  if (key < 0 || key > fan::input_last) return false;
  return key_states[key] == fan::keyboard_state::press ||
    key_states[key] == fan::keyboard_state::repeat;
}
bool fan::window_t::is_key_released(int key) {
  if (!this) return false;
  if (key < 0 || key > fan::input_last) return false;
  return key_states[key] != fan::keyboard_state::press &&
    prev_key_states[key] == fan::keyboard_state::press;
}

bool fan::window_t::is_mouse_clicked(int button) {
  if (!this) return false;
  if (button < fan::mouse_first || button > fan::mouse_last) return false;
  return key_states[button] == fan::keyboard_state::press &&
         prev_key_states[button] != fan::keyboard_state::press;
}
bool fan::window_t::is_mouse_down(int button) {
  if (!this) return false;
  if (button < fan::mouse_first || button > fan::mouse_last) return false;
  return key_states[button] == fan::keyboard_state::press ||
    key_states[button] == fan::keyboard_state::repeat;
}
bool fan::window_t::is_gamepad_button_down(int key) {
  int jid = 0;
  if (!glfwJoystickPresent(jid)) return false;

  int count;
  const unsigned char* buttons = glfwGetJoystickButtons(jid, &count);
  if (!buttons) return false;

  int idx = key - fan::gamepad_a;
  if (idx < 0 || idx >= count) return false;

  return buttons[idx] == GLFW_PRESS;
}
bool fan::window_t::is_gamepad_axis_active(int key) {
  fan::vec2 axis = get_gamepad_axis(key);
  if (key == fan::gamepad_l2 || key == fan::gamepad_r2) {
    return axis.x > gamepad_axis_deadzone;
  }
  return axis.length() > gamepad_axis_deadzone;
}
fan::vec2 fan::window_t::get_current_gamepad_axis(int key) {
  if (!this) return fan::vec2(0, 0);
  return get_gamepad_axis(key);
}