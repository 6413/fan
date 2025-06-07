module;

#include <fan/types/types.h>

#if defined(fan_vulkan)
#include <vulkan/vulkan.h>
#endif
#if defined(fan_platform_windows)
  #define WIN32_LEAN_AND_MEAN
  #include <Windows.h>
  #define GLFW_EXPOSE_NATIVE_WIN32
  #define GLFW_EXPOSE_NATIVE_WGL
  #define GLFW_NATIVE_INCLUDE_NONE
#endif
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <string>

export module fan.window;

import fan.print;
export import fan.window.input_common;

export import fan.types.vector;

export namespace fan {
  struct init_manager_t {
    static bool& initialized() {
      static bool instance = false;
      return instance;
    }

    static void initialize() {
      if (initialized()) {
        return;
      }
      if (!glfwInit()) {
        fan::throw_error("failed to initialize context");
      }
      initialized() = true;
    }
    static void uninitialize() {
      if (initialized() == false) {
        return;
      }
      glfwTerminate();
      initialized() = false;
    }

    struct cleaner_t {
      cleaner_t() {
        initialize();
      }

      ~cleaner_t() {
        uninitialize();
      }
    };

    static cleaner_t& cleaner() {
      static cleaner_t instance;
      return instance;
    }
  };
  fan::init_manager_t::cleaner_t& _cleaner = fan::init_manager_t::cleaner();
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

    struct mouse_buttons_cb_data_t {
      fan::window_t* window;
      uint16_t button;
      fan::mouse_state state;
    };
    using mouse_buttons_cb_t = std::function<void(const mouse_buttons_cb_data_t&)>;

    struct keyboard_keys_cb_data_t {
      fan::window_t* window;
      int key;
      fan::keyboard_state_t state;
      uint16_t scancode;
    };
    using keyboard_keys_cb_t = std::function<void(const keyboard_keys_cb_data_t&)>;

    struct keyboard_key_cb_data_t {
      fan::window_t* window;
      int key;
    };
    using keyboard_key_cb_t = std::function<void(const keyboard_key_cb_data_t&)>;

    struct text_cb_data_t {
      fan::window_t* window;
      uint32_t character;
      fan::keyboard_state_t state;
    };
    using text_cb_t = std::function<void(const text_cb_data_t&)>;

    struct mouse_move_cb_data_t {
      fan::window_t* window;
      fan::vec2d position;
    };
    using mouse_move_cb_t = std::function<void(const mouse_move_cb_data_t&)>;

    struct mouse_motion_cb_data_t {
      fan::window_t* window;
      fan::vec2d motion;
    };
    using mouse_motion_cb_t = std::function<void(const mouse_motion_cb_data_t&)>;

    struct close_cb_data_t {
      fan::window_t* window;
    };
    using close_cb_t = std::function<void(const close_cb_data_t&)>;

    struct resize_cb_data_t {
      fan::window_t* window;
      fan::vec2i size;
    };
    using resize_cb_t = std::function<void(const resize_cb_data_t&)>;

    struct move_cb_data_t {
      fan::window_t* window;
    };
    using move_cb_t = std::function<void(const move_cb_data_t&)>;

    struct keyboard_cb_store_t {
      int key;
      keyboard_state_t state;

      keyboard_key_cb_t function;
    };

    #define BLL_set_prefix buttons_callback
    #define BLL_set_NodeData mouse_buttons_cb_t data;
    #include "cb_list_builder_settings.h"
    #include <BLL/BLL.h>

    #define BLL_set_prefix keys_callback
    #define BLL_set_NodeData keyboard_keys_cb_t data;
    #include "cb_list_builder_settings.h"
    #include <BLL/BLL.h>

    #define BLL_set_prefix key_callback
    #define BLL_set_NodeData keyboard_cb_store_t data;
    #include "cb_list_builder_settings.h"
    #include <BLL/BLL.h>

    #define BLL_set_prefix text_callback
    #define BLL_set_NodeData text_cb_t data;
    #include "cb_list_builder_settings.h"
    #include <BLL/BLL.h>

    #define BLL_set_prefix move_callback
    #define BLL_set_NodeData move_cb_t data;
    #include "cb_list_builder_settings.h"
    #include <BLL/BLL.h>

    #define BLL_set_prefix resize_callback
    #define BLL_set_NodeData resize_cb_t data;
    #include "cb_list_builder_settings.h"
    #include <BLL/BLL.h>

    #define BLL_set_prefix close_callback
    #define BLL_set_NodeData close_cb_t data;
    #include "cb_list_builder_settings.h"
    #include <BLL/BLL.h>

    #define BLL_set_prefix mouse_position_callback
    #define BLL_set_NodeData mouse_move_cb_t data;
    #include "cb_list_builder_settings.h"
    #include <BLL/BLL.h>

    #define BLL_set_prefix mouse_motion_callback
    #define BLL_set_NodeData mouse_motion_cb_t data;
    #include "cb_list_builder_settings.h"
    #include <BLL/BLL.h>

    struct flags {
      static constexpr uint64_t no_mouse = 1 << 0;
      static constexpr uint64_t no_resize = 1 << 1;
      static constexpr uint64_t mode = 1 << 2;
      static constexpr uint64_t borderless = 1 << 3;
      static constexpr uint64_t full_screen = 1 << 4;
      static constexpr uint64_t no_decorate = 1 << 5;
      static constexpr uint64_t transparent = 1 << 6;
      static constexpr uint64_t no_visible = 1 << 7;
    };

    struct renderer_t {
      static constexpr uint8_t opengl = 0;
      static constexpr uint8_t vulkan = 1;
    };
    uint8_t renderer = renderer_t::opengl;

   //window_t();
    void open(uint64_t flags) {
      fan::window_t::open(default_window_size, default_window_name, flags);
    }
    void open(fan::vec2i window_size, const std::string& name, uint64_t flags = 0) {
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

      if (!(flags & flags::no_visible)) {
        glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
      }
      else {
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
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

      GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();
      const GLFWvidmode* mode = glfwGetVideoMode(primaryMonitor);
      fan::vec2 screen_size = fan::vec2(mode->width, mode->height);
      fan::vec2 window_pos = (screen_size - window_size) / 2;
      glfwSetWindowPos(glfw_window, window_pos.x, window_pos.y);

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
    void close() {
      if (glfw_window) {
        glfwMakeContextCurrent(nullptr);
        glfwDestroyWindow(glfw_window);
        glfw_window = nullptr;
      }
    }

    void make_context_current() {
      glfwMakeContextCurrent(*this);
    }

    void handle_key_states() {
      // can be 1 or 2 aka press or repeat
      if (key_state(fan::mouse_left) == 1 || key_state(fan::mouse_middle) == 1 || key_state(fan::mouse_right) == 1) {
        drag_delta_start = get_mouse_position();
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
    }
    uint32_t handle_events() {
      f64_t current_frame_time = glfwGetTime();
      m_delta_time = current_frame_time - last_frame_time;
      last_frame_time = current_frame_time;

      handle_key_states();

      glfwPollEvents();

      return 0;
    }

    buttons_callback_NodeReference_t add_buttons_callback(mouse_buttons_cb_t function) {
      auto nr = m_buttons_callback.NewNodeLast();

      m_buttons_callback[nr].data = function;
      return nr;
    }
    void remove_buttons_callback(buttons_callback_NodeReference_t id) {
      m_buttons_callback.Unlink(id);
      m_buttons_callback.Recycle(id);
    }

    keys_callback_NodeReference_t add_keys_callback(keyboard_keys_cb_t function) {
      auto nr = m_keys_callback.NewNodeLast();
      m_keys_callback[nr].data = function;
      return nr;
    }
    void remove_keys_callback(keys_callback_NodeReference_t id) {
      m_keys_callback.Unlink(id);
      m_keys_callback.Recycle(id);
    }

    key_callback_NodeReference_t add_key_callback(int key, keyboard_state_t state, keyboard_key_cb_t function) {
      auto nr = m_key_callback.NewNodeLast();
      m_key_callback[nr].data = keyboard_cb_store_t{ key, state, function, };
      return nr;
    }
    void edit_key_callback(key_callback_NodeReference_t id, int key, keyboard_state_t state) {
      m_key_callback[id].data.key = key;
      m_key_callback[id].data.state = state;
    }
    void remove_key_callback(key_callback_NodeReference_t id) {
      m_key_callback.unlrec(id);
    }

    text_callback_NodeReference_t add_text_callback(text_cb_t function) {
      auto nr = m_text_callback.NewNodeLast();
      m_text_callback[nr].data = function;
      return nr;
    }
    void remove_text_callback(text_callback_NodeReference_t id) {
      m_text_callback.Unlink(id);
      m_text_callback.Recycle(id);
    }

    close_callback_NodeReference_t add_close_callback(close_cb_t function) {
      auto nr = m_close_callback.NewNodeLast();
      m_close_callback[nr].data = function;
      return nr;
    }
    void remove_close_callback(close_callback_NodeReference_t id) {
      m_close_callback.Unlink(id);
      m_close_callback.Recycle(id);
    }

    mouse_position_callback_NodeReference_t add_mouse_move_callback(mouse_move_cb_t function) {
      auto nr = m_mouse_position_callback.NewNodeLast();
      m_mouse_position_callback[nr].data = function;
      return nr;
    }
    void remove_mouse_move_callback(mouse_position_callback_NodeReference_t id) {
      m_mouse_position_callback.Unlink(id);
      m_mouse_position_callback.Recycle(id);
    }

    mouse_motion_callback_NodeReference_t add_mouse_motion(mouse_motion_cb_t function) {
      auto nr = m_mouse_motion_callback.NewNodeLast();
      m_mouse_motion_callback[nr].data = function;
      return nr;
    }

    void erase_mouse_motion_callback(mouse_motion_callback_NodeReference_t id) {
      m_mouse_motion_callback.Unlink(id);
      m_mouse_motion_callback.Recycle(id);
    }

    resize_callback_NodeReference_t add_resize_callback(resize_cb_t function) {
      auto nr = m_resize_callback.NewNodeLast();
      m_resize_callback[nr].data = function;
      return nr;
    }

    void remove_resize_callback(resize_callback_NodeReference_t id) {
      m_resize_callback.Unlink(id);
      m_resize_callback.Recycle(id);
    }

    move_callback_NodeReference_t add_move_callback(move_cb_t function) {
      auto nr = m_move_callback.NewNodeLast();
      m_move_callback[nr].data = function;
      return nr;
    }

    void remove_move_callback(move_callback_NodeReference_t idt) {
      m_move_callback.unlrec(idt);
    }

    fan::vec2i get_size() const {
      fan::vec2i window_size;
      glfwGetWindowSize(glfw_window, &window_size.x, &window_size.y);
      return window_size;
    }

    void set_size(const fan::vec2i& window_size) {
      glfwSetWindowSize(glfw_window, window_size.x, window_size.y);
    }

    fan::vec2 get_position() const {
      int xpos, ypos;
      glfwGetWindowPos(glfw_window, &xpos, &ypos);
      return fan::vec2(xpos, ypos);
    }

    void set_position(const fan::vec2& position) {
      glfwSetWindowPos(glfw_window, position.x, position.y);
    }

    void set_windowed() {
      glfwSetWindowAttrib(glfw_window, GLFW_DECORATED, true);
      using namespace fan::window;
      GLFWmonitor* monitor = glfwGetPrimaryMonitor();
      const GLFWvidmode* mode = glfwGetVideoMode(monitor);
      //glfwSetWindowSize(glfw_window, windowWidth, windowHeight);
      glfwSetWindowMonitor(glfw_window, NULL, mode->width / 8, mode->height / 8, mode->width / 2, mode->height / 2, mode->refreshRate);
      fan::vec2 screen_size = fan::vec2(mode->width, mode->height);
      fan::vec2 window_pos = (screen_size - get_size()) / 2;
      glfwSetWindowPos(glfw_window, window_pos.x, window_pos.y);
      display_mode = (uint8_t)mode::windowed;
    }

    void set_fullscreen() {
      using namespace fan::window;
      GLFWmonitor* monitor = glfwGetPrimaryMonitor();
      const GLFWvidmode* mode = glfwGetVideoMode(monitor);
      glfwSetWindowMonitor(glfw_window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
      display_mode = (uint8_t)mode::full_screen;
    }

    void set_windowed_fullscreen() {
      using namespace fan::window;
      GLFWmonitor* monitor = glfwGetPrimaryMonitor();
      const GLFWvidmode* mode = glfwGetVideoMode(monitor);

#if defined(fan_platform_windows)
      glfwSetWindowMonitor(glfw_window, NULL, 0, mode->height - mode->height / (1.04), mode->width, mode->height / (1.08), mode->refreshRate);
#else
      glfwSetWindowMonitor(glfw_window, NULL, 0, 0, mode->width, mode->height, mode->refreshRate);
#endif
      display_mode = (uint8_t)mode::windowed_fullscreen;
    }

    void set_borderless() {
      using namespace fan::window;

      GLFWmonitor* monitor = glfwGetPrimaryMonitor();
      const GLFWvidmode* mode = glfwGetVideoMode(monitor);

      set_position(0);
      glfwSetWindowAttrib(glfw_window, GLFW_DECORATED, false);
      set_size(fan::vec2(mode->width, mode->height));
      display_mode = (uint8_t)mode::borderless;
    }


    void set_cursor(int flag) {
      glfwSetInputMode(*this, GLFW_CURSOR, flag == 0 ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    }

    void toggle_cursor() {
      bool disabled = glfwGetInputMode(*this, GLFW_CURSOR) != GLFW_CURSOR_DISABLED;
      set_cursor(disabled);
    }

    void set_display_mode(const mode& mode) {
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

    fan::vec2d get_mouse_position() const {
      fan::vec2d mouse_pos;
      glfwGetCursorPos(glfw_window, &mouse_pos.x, &mouse_pos.y);
      return mouse_pos;
    }

    int key_state(int key) const {
      return key_states[key];
    }

    bool key_pressed(int key, int press = fan::keyboard_state::press) const {
      return glfwGetKey(glfw_window, key) == press;
    }

    fan::vec2 get_gamepad_axis(int key) const {
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

    double last_frame_time = glfwGetTime();
    f64_t m_delta_time = 0;
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
    uint64_t flags = 0;

    operator GLFWwindow* () {
      return glfw_window;
    }
    operator GLFWwindow* const () const {
      return glfw_window;
    }

    int prev_key_states[fan::last]{};
    int key_states[fan::last]{};
    GLFWwindow* glfw_window;

    fan::vec2d previous_mouse_position = -0xfff;

    fan::vec2 drag_delta_start = -1;
    uint8_t display_mode = (uint8_t)mode::windowed;
  };
}

void fan::window::mouse_button_callback(GLFWwindow* wnd, int button, int action, int mods) {
  fan::window_t* window = (fan::window_t*)glfwGetWindowUserPointer(wnd);
  window->key_states[button] = action;

  auto it = window->m_buttons_callback.GetNodeFirst();

  while (it != window->m_buttons_callback.dst) {
    fan::window_t::mouse_buttons_cb_data_t cbd;
    cbd.window = window;
    cbd.button = button;
    cbd.state = static_cast<fan::mouse_state>(action);
    window->m_buttons_callback[it].data(cbd);

    it = it.Next(&window->m_buttons_callback);
  }
}

void fan::window::keyboard_keys_callback(GLFWwindow* wnd, int key, int scancode, int action, int mods) {
  if (key == -1) {
    return;
  }

  fan::window_t* window = (fan::window_t*)glfwGetWindowUserPointer(wnd);
  window->key_states[key] = action;

  {
    auto it = window->m_keys_callback.GetNodeFirst();

    while (it != window->m_keys_callback.dst) {
      fan::window_t::keyboard_keys_cb_data_t cbd;
      cbd.window = window;
      cbd.key = key;
      cbd.state = static_cast<fan::keyboard_state_t>(action);
      cbd.scancode = scancode;
      window->m_keys_callback[it].data(cbd);
      it = it.Next(&window->m_keys_callback);
    }
  }
  {
    auto it = window->m_key_callback.GetNodeFirst();

    while (it != window->m_key_callback.dst) {
        
      fan::window_t::keyboard_key_cb_data_t cbd;
      cbd.window = window;
      cbd.key = key;
      if (window->m_key_callback[it].data.key == key && (int)window->m_key_callback[it].data.state == action) {
        window->m_key_callback[it].data.function(cbd);
      }
      it = it.Next(&window->m_key_callback);
    }
      
  }
}

void fan::window::text_callback(GLFWwindow* wnd, unsigned int codepoint) {
  fan::window_t* window = (fan::window_t*)glfwGetWindowUserPointer(wnd);
  auto it = window->m_text_callback.GetNodeFirst();

  while (it != window->m_text_callback.dst) {
    fan::window_t::text_cb_data_t cbd;
    cbd.window = window;
    cbd.character = codepoint;
    cbd.state = fan::keyboard_state::press;
    window->m_text_callback[it].data(cbd);

    it = it.Next(&window->m_text_callback);
  }
}

void fan::window::mouse_position_callback(GLFWwindow* wnd, double xpos, double ypos) {
  fan::window_t* window = (fan::window_t*)glfwGetWindowUserPointer(wnd);
  {
    auto it = window->m_mouse_position_callback.GetNodeFirst();

    fan::window_t::mouse_move_cb_data_t cbd;
    cbd.window = window;
    cbd.position = fan::vec2d(xpos, ypos);

    while (it != window->m_mouse_position_callback.dst) {
      window->m_mouse_position_callback[it].data(cbd);

      it = it.Next(&window->m_mouse_position_callback);
    }
  }
  if (window->previous_mouse_position.x == -0xfff && window->previous_mouse_position.y == -0xfff) {
    window->previous_mouse_position = fan::vec2d(xpos, ypos);
  }
  {
    fan::window_t::mouse_motion_cb_data_t cbd;
    cbd.window = window;
    cbd.motion = fan::vec2d(xpos, ypos) - window->previous_mouse_position;
    auto it = window->m_mouse_motion_callback.GetNodeFirst();
    while (it != window->m_mouse_motion_callback.dst) {
      window->m_mouse_motion_callback[it].data(cbd);

      it = it.Next(&window->m_mouse_motion_callback);
    }
  }
  window->previous_mouse_position = fan::vec2d(xpos, ypos);
}

void fan::window::close_callback(GLFWwindow* wnd) {
  fan::window_t* window = (fan::window_t*)glfwGetWindowUserPointer(wnd);
  auto it = window->m_close_callback.GetNodeFirst();

  while (it != window->m_close_callback.dst) {
    fan::window_t::close_cb_data_t cbd;
    cbd.window = window;
    window->m_close_callback[it].data(cbd);

    it = it.Next(&window->m_close_callback);
  }
}

void fan::window::resize_callback(GLFWwindow* wnd, int width, int height) {
  fan::window_t* window = (fan::window_t*)glfwGetWindowUserPointer(wnd);
  auto it = window->m_resize_callback.GetNodeFirst();

  while (it != window->m_resize_callback.dst) {
    fan::window_t::resize_cb_data_t cbd;
    cbd.window = window;
    cbd.size = fan::vec2i(width, height);
    window->m_resize_callback[it].data(cbd);

    it = it.Next(&window->m_resize_callback);
  }
}

void fan::window::move_callback(GLFWwindow* wnd, int xpos, int ypos) {
  fan::window_t* window = (fan::window_t*)glfwGetWindowUserPointer(wnd);
  auto it = window->m_move_callback.GetNodeFirst();

  while (it != window->m_move_callback.dst) {
    fan::window_t::move_cb_data_t cbd;
    cbd.window = window;
    window->m_move_callback[it].data(cbd);

    it = it.Next(&window->m_move_callback);
  }
}

void fan::window::scroll_callback(GLFWwindow* wnd, double xoffset, double yoffset) {
  fan::window_t* window = (fan::window_t*)glfwGetWindowUserPointer(wnd);

  auto it = window->m_buttons_callback.GetNodeFirst();

  int button = yoffset < 0 ? fan::input::mouse_scroll_down : fan::input::mouse_scroll_up;

  window->key_states[button] = (int)fan::mouse_state::press;

  while (it != window->m_buttons_callback.dst) {
    fan::window_t::mouse_buttons_cb_data_t cbd;
    cbd.window = window;
    cbd.button = button;
    cbd.state = fan::mouse_state::press;
    window->m_buttons_callback[it].data(cbd);

    it = it.Next(&window->m_buttons_callback);
  }
}

void fan::window::window_focus_callback(GLFWwindow* wnd, int focused) {
  fan::window_t* window = (fan::window_t*)glfwGetWindowUserPointer(wnd);
  for (int i = 0; i < GLFW_KEY_LAST; i++) {
    if (window->key_states[i] != -1) {
      window->key_states[i] = GLFW_RELEASE;
    }
  }
}