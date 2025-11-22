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
#include <algorithm>

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
    fan::window_t::open(default_window_size, default_window_name, flags);
  }

  void window_t::open(fan::vec2i window_size, const std::string& name, std::uint64_t flags) {
    this->flags = flags;
    std::fill(key_states, key_states + std::size(key_states), -1);
    std::fill(prev_key_states, prev_key_states + std::size(prev_key_states), -1);
    if (window_size.x == -1 && window_size.y == -1) {
      const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
      window_size = resolutions[current_resolution];
    }
  #if fan_debug >= fan_debug_high
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
  #endif
    if (m_antialiasing_samples > 0) {
      glfwWindowHint(GLFW_SAMPLES, m_antialiasing_samples);
    }
    if (!(flags & flags::hidden)) {
      glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
    }
    else {
      glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    }
    if (flags & flags::transparent) {
      glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
      glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE);
    }
    using namespace fan::window;
    if (renderer == renderer_t::vulkan) {
      glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    }
    else if (renderer == renderer_t::opengl) {
      glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    }
    glfw_window = glfwCreateWindow(window_size.x, window_size.y, name.c_str(), NULL, NULL);
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
    const GLFWvidmode* mode = glfwGetVideoMode(primaryMonitor);
    fan::vec2 screen_size = fan::vec2(mode->width, mode->height);
    fan::vec2 window_pos = (screen_size - window_size) / 2;
    //glfwSetWindowPos(glfw_window, window_pos.x, window_pos.y);
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

  void window_t::handle_key_states() {
    // can be 1 or 2 aka press or repeat
    if (key_state(fan::mouse_left) == 1 || key_state(fan::mouse_middle) == 1 || key_state(fan::mouse_right) == 1) {
      drag_delta_start = get_mouse_position(); // requires manual reset = -1 because button callbacks are processed before this
    }
    else if (key_state(fan::mouse_left) == 0 || key_state(fan::mouse_middle) == 0 || key_state(fan::mouse_right) == 0) {
      drag_delta_start = -1;
    }
    if (key_states[fan::mouse_scroll_up] == 1) {
      key_states[fan::mouse_scroll_up] = 0;
    }
    if (key_states[fan::mouse_scroll_down] == 1) {
      key_states[fan::mouse_scroll_down] = 0;
    }
    memcpy(prev_key_states, key_states, sizeof(key_states));
    for (std::size_t i = 0; i < std::size(key_states); ++i) {
      if (key_states[i] == GLFW_PRESS && prev_key_states[i] == GLFW_PRESS) {
        key_states[i] = GLFW_REPEAT;
      }
      if (key_states[i] == GLFW_RELEASE && prev_key_states[i] == GLFW_RELEASE) {
        key_states[i] = -1;
      }
    }
    // gamepad
    for (int i = fan::gamepad_a; i <= fan::gamepad_left; ++i) {
      // fix hardcoded joystick
      int present = glfwJoystickPresent(GLFW_JOYSTICK_1);
      if (present) {
        int button_count;
        const unsigned char* buttons = glfwGetJoystickButtons(GLFW_JOYSTICK_1, &button_count);
        //if (i - fan::gamepad_a >= button_count) {
        //  fan::print("i bigger or equal than button_count");
        //}
        if (key_states[i] == GLFW_PRESS && prev_key_states[i] == GLFW_PRESS) {
          key_states[i] = GLFW_REPEAT;
        }
        else if (key_states[i] == -1 && buttons[i - fan::gamepad_a]) {
          key_states[i] = GLFW_PRESS;
        }
        else if (buttons[i - fan::gamepad_a] == GLFW_RELEASE && (prev_key_states[i] == GLFW_PRESS || prev_key_states[i] == GLFW_REPEAT)) {
          key_states[i] = GLFW_RELEASE;
        }
        else if (buttons[i - fan::gamepad_a] == GLFW_RELEASE && (key_states[i] == GLFW_PRESS || key_states[i] == GLFW_REPEAT)) {
          key_states[i] = -1;
        }
      }
    }
    {
      auto it = m_mouse_down_callbacks.GetNodeFirst();
      while (it != m_mouse_down_callbacks.dst) {
        m_mouse_down_callbacks.StartSafeNext(it);
        for (int b = 0; b < 3; ++b) {
          int state = key_state(b);
          if (state == fan::keyboard_state::press || state == fan::keyboard_state::repeat) {
            buttons_data_t cbd;
            cbd.window = this;
            cbd.position = get_mouse_position();
            cbd.button = b;
            cbd.state = state;
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
        for (int k = fan::key_first; k <= fan::key_last; ++k) {
          int ks = key_state(k);
          if (ks == fan::keyboard_state::press || ks == fan::keyboard_state::repeat) {
            fan::window_t::keys_data_t cbd;
            cbd.window = this;
            cbd.scancode = fan::window::input::convert_fan_to_scancode(k);
            cbd.key = k;
            cbd.state = (decltype(cbd.state))ks;
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
    #if fan_debug >= 4
        fan::print("framerate too low, overriding delta time");
    #endif
        m_delta_time = 1.0 / 256.0;
    }*/
    last_frame_time = current_frame_time;
    handle_key_states();
    glfwPollEvents();
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
    glfwSetWindowAttrib(glfw_window, GLFW_DECORATED, true);

    GLFWmonitor* monitor = get_current_monitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);

    int monitor_x, monitor_y;
    glfwGetMonitorPos(monitor, &monitor_x, &monitor_y);

    int window_width = mode->width / 2;
    int window_height = mode->height / 2;
    int window_x = monitor_x + mode->width / 8;
    int window_y = monitor_y + mode->height / 8;

    glfwSetWindowMonitor(glfw_window, NULL, window_x, window_y, window_width, window_height, mode->refreshRate);
    display_mode = (uint8_t)mode::windowed;
  }

  void window_t::set_fullscreen() {
    GLFWmonitor* monitor = get_current_monitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    glfwSetWindowMonitor(glfw_window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
    display_mode = (uint8_t)mode::full_screen;
  }

  void window_t::set_windowed_fullscreen() {
    GLFWmonitor* monitor = get_current_monitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);

    int monitor_x, monitor_y;
    glfwGetMonitorPos(monitor, &monitor_x, &monitor_y);

    int work_x, work_y, work_width, work_height;
    glfwGetMonitorWorkarea(monitor, &work_x, &work_y, &work_width, &work_height);

    glfwSetWindowMonitor(glfw_window, NULL, work_x, work_y, work_width, work_height, mode->refreshRate);
    display_mode = (uint8_t)mode::windowed_fullscreen;
  }

  void window_t::set_borderless() {
    GLFWmonitor* monitor = get_current_monitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);

    int monitor_x, monitor_y;
    glfwGetMonitorPos(monitor, &monitor_x, &monitor_y);

    glfwSetWindowAttrib(glfw_window, GLFW_DECORATED, false);
    set_position(fan::vec2(monitor_x, monitor_y));
    set_size(fan::vec2(mode->width, mode->height));
    display_mode = (uint8_t)mode::borderless;
  }

  void window_t::set_cursor(int flag) {
    glfwSetInputMode(*this, GLFW_CURSOR, flag == 0 ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
  }

  void window_t::toggle_cursor() {
    bool disabled = glfwGetInputMode(*this, GLFW_CURSOR) != GLFW_CURSOR_DISABLED;
    set_cursor(disabled);
  }

  void window_t::set_display_mode(const mode& mode) {
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
    case mode::windowed_fullscreen: {
      set_windowed_fullscreen();
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
    if (axes_count == 0) {
      return axis;
    }
    switch (key) {
    case fan::gamepad_left_thumb: {
      if (axes_count > 1) {
        axis = fan::vec2(axes[0], axes[1]);
      }
      break;
    }
    case fan::gamepad_right_thumb: {
      if (axes_count > 3) {
        axis = fan::vec2(axes[2], axes[3]);
      }
      break;
    }
    case fan::gamepad_l2: {
      if (axes_count > 5) {
        axis = fan::vec2(axes[4], axes[5]);
      }
      break;
    }
    case fan::gamepad_r2: {
      if (axes_count > 5) {
        axis = fan::vec2(axes[5], axes[4]);
      }
      break;
    }
    }
    return axis;
  }
}