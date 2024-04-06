namespace fan {
  namespace window {
    #define GLFW_INCLUDE_NONE
    #include <GLFW/glfw3.h>

    static void error_callback(int error, const char* description)
    {
      //fan::print("window error:", description);
    }

    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void keyboard_keys_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void text_callback(GLFWwindow* window, unsigned int codepoint);
    static void move_callback(GLFWwindow* window, int x, int y);
    static void resize_callback(GLFWwindow* window, int width, int height);
    static void close_callback(GLFWwindow* window);
    static void mouse_position_callback(GLFWwindow* window, double xpos, double ypos);
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
    //static void mouse_motion_callback(GLFWwindow* window, double xoffset, double yoffset);
  }
}

#include _FAN_PATH(types/vector.h)
#include _FAN_PATH(types/matrix.h)
#include _FAN_PATH(window/window_input.h)

namespace fan {
  struct window_t {

    enum class mode {
      not_set,
      windowed,
      borderless,
      full_screen
    };

    inline static std::unordered_map<fan::window::GLFWwindow*, window_t*> window_map;

    static constexpr const char* default_window_name = "window";
    static constexpr fan::vec2i default_window_size = fan::vec2i(800, 600);
    static constexpr mode default_size_mode = mode::windowed;

    struct resolutions {
      constexpr static fan::vec2i r_800x600 = fan::vec2i(800, 600);
      constexpr static fan::vec2i r_1024x768 = fan::vec2i(1024, 768);
      constexpr static fan::vec2i r_1280x720 = fan::vec2i(1280, 720);
      constexpr static fan::vec2i r_1280x800 = fan::vec2i(1280, 800);
      constexpr static fan::vec2i r_1280x900 = fan::vec2i(1280, 900);
      constexpr static fan::vec2i r_1280x1024 = fan::vec2i(1280, 1024);
      constexpr static fan::vec2i r_1360x768 = fan::vec2(1360, 768);
      constexpr static fan::vec2i r_1440x900 = fan::vec2i(1440, 900);
      constexpr static fan::vec2i r_1600x900 = fan::vec2i(1600, 900);
      constexpr static fan::vec2i r_1600x1024 = fan::vec2i(1600, 1024);
      constexpr static fan::vec2i r_1680x1050 = fan::vec2i(1680, 1050);
      constexpr static fan::vec2i r_1920x1080 = fan::vec2i(1920, 1080);
    };

    struct mouse_buttons_cb_data_t {
      fan::window_t* window;
      uint16_t button;
      fan::mouse_state state;
    };
    using mouse_buttons_cb_t = fan::function_t<void(const mouse_buttons_cb_data_t&)>;

    struct keyboard_keys_cb_data_t {
      fan::window_t* window;
      int key;
      fan::keyboard_state state;
      uint16_t scancode;
    };
    using keyboard_keys_cb_t = fan::function_t<void(const keyboard_keys_cb_data_t&)>;

    struct keyboard_key_cb_data_t {
      fan::window_t* window;
      int key;
    };
    using keyboard_key_cb_t = fan::function_t<void(const keyboard_key_cb_data_t&)>;

    struct text_cb_data_t {
      fan::window_t* window;
      uint32_t character;
      fan::keyboard_state state;
    };
    using text_cb_t = fan::function_t<void(const text_cb_data_t&)>;

    struct mouse_move_cb_data_t {
      fan::window_t* window;
      fan::vec2i position;
    };
    using mouse_move_cb_t = fan::function_t<void(const mouse_move_cb_data_t&)>;

    struct mouse_motion_cb_data_t {
      fan::window_t* window;
      fan::vec2i motion;
    };
    using mouse_motion_cb_t = fan::function_t<void(const mouse_motion_cb_data_t&)>;

    struct close_cb_data_t {
      fan::window_t* window;
    };
    using close_cb_t = fan::function_t<void(const close_cb_data_t&)>;

    struct resize_cb_data_t {
      fan::window_t* window;
      fan::vec2i size;
    };
    using resize_cb_t = fan::function_t<void(const resize_cb_data_t&)>;

    struct move_cb_data_t {
      fan::window_t* window;
    };
    using move_cb_t = fan::function_t<void(const move_cb_data_t&)>;

    struct keyboard_cb_store_t {
      int key;
      keyboard_state state;

      keyboard_key_cb_t function;
    };

    #define BLL_set_prefix buttons_callback
    #define BLL_set_NodeData mouse_buttons_cb_t data;
    #include "cb_list_builder_settings.h"
    #include _FAN_PATH(BLL/BLL.h)

    #define BLL_set_prefix keys_callback
    #define BLL_set_NodeData keyboard_keys_cb_t data;
    #include "cb_list_builder_settings.h"
    #include _FAN_PATH(BLL/BLL.h)

    #define BLL_set_prefix key_callback
    #define BLL_set_NodeData keyboard_cb_store_t data;
    #include "cb_list_builder_settings.h"
    #include _FAN_PATH(BLL/BLL.h)

    #define BLL_set_prefix text_callback
    #define BLL_set_NodeData text_cb_t data;
    #include "cb_list_builder_settings.h"
    #include _FAN_PATH(BLL/BLL.h)

    #define BLL_set_prefix move_callback
    #define BLL_set_NodeData move_cb_t data;
    #include "cb_list_builder_settings.h"
    #include _FAN_PATH(BLL/BLL.h)

    #define BLL_set_prefix resize_callback
    #define BLL_set_NodeData resize_cb_t data;
    #include "cb_list_builder_settings.h"
    #include _FAN_PATH(BLL/BLL.h)

    #define BLL_set_prefix close_callback
    #define BLL_set_NodeData close_cb_t data;
    #include "cb_list_builder_settings.h"
    #include _FAN_PATH(BLL/BLL.h)

    #define BLL_set_prefix mouse_position_callback
    #define BLL_set_NodeData mouse_move_cb_t data;
    #include "cb_list_builder_settings.h"
    #include _FAN_PATH(BLL/BLL.h)

    #define BLL_set_prefix mouse_motion_callback
    #define BLL_set_NodeData mouse_motion_cb_t data;
    #include "cb_list_builder_settings.h"
    #include _FAN_PATH(BLL/BLL.h)

    struct flags {
      static constexpr int no_mouse = 1 << 0;
      static constexpr int no_resize = 1 << 1;
      static constexpr int mode = 1 << 2;
      static constexpr int borderless = 1 << 3;
      static constexpr int full_screen = 1 << 4;
      static constexpr int no_decorate = 1 << 5;
      static constexpr int transparent = 1 << 6;
    };


    inline static struct glfw_initialize_t {
      glfw_initialize_t() = default;
      void open() {
        using namespace fan::window;
        if (glfwInit() == false) {
          fan::throw_error("failed to initialize window manager context");
        }
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, loco_gl_major);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, loco_gl_minor);

        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, true);

        glfwSetErrorCallback(error_callback);
        initialized = true;
      }
      ~glfw_initialize_t() {
        using namespace fan::window;
        glfwTerminate();
      }
      static inline bool initialized = false;
    }intialize_glfw_var;

    window_t() : window_t(fan::window_t::default_window_size, fan::window_t::default_window_name, 0) {
    }
    window_t(const fan::vec2i& window_size = fan::window_t::default_window_size, const fan::string& name = default_window_name, uint64_t flags = 0) {
      if (intialize_glfw_var.initialized == false) {
        intialize_glfw_var.open();
      }

      using namespace fan::window;
      glfw_window = glfwCreateWindow(window_size.x, window_size.y, name.c_str(), NULL, NULL);
      if (glfw_window == nullptr) {
        glfwTerminate();
        fan::throw_error("failed to create window");
      }

      GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();
      const GLFWvidmode* mode = glfwGetVideoMode(primaryMonitor);
      fan::vec2 screen_size = fan::vec2(mode->width, mode->height);
      fan::vec2 window_pos = (screen_size - window_size) / 2;
      glfwSetWindowPos(glfw_window, window_pos.x, window_pos.y);

      glfwMakeContextCurrent(glfw_window);

      window_map[glfw_window] = this;

      glfwSetMouseButtonCallback(glfw_window, fan::window::mouse_button_callback);
      glfwSetKeyCallback(glfw_window, fan::window::keyboard_keys_callback);
      glfwSetCharCallback(glfw_window, fan::window::text_callback);
      glfwSetWindowPosCallback(glfw_window, fan::window::move_callback);
      glfwSetWindowSizeCallback(glfw_window, fan::window::resize_callback);
      glfwSetWindowCloseCallback(glfw_window, fan::window::close_callback);
      glfwSetCursorPosCallback(glfw_window, fan::window::mouse_position_callback);
      glfwSetScrollCallback(glfw_window, fan::window::scroll_callback);

      frame_timer.start(1e+9);
    }

    void close() {
      fan::window::glfwDestroyWindow(glfw_window);
    }

    uint32_t handle_events() {
      f64_t current_frame_time = fan::window::glfwGetTime();
      m_delta_time = current_frame_time - last_frame_time;
      last_frame_time = current_frame_time;

      fan::window::glfwPollEvents();
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

    key_callback_NodeReference_t add_key_callback(int key, keyboard_state state, keyboard_key_cb_t function) {
      auto nr = m_key_callback.NewNodeLast();
      m_key_callback[nr].data = keyboard_cb_store_t{ key, state, function, };
      return nr;
    }
    void edit_key_callback(key_callback_NodeReference_t id, int key, keyboard_state state) {
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
      fan::window::glfwGetWindowSize(glfw_window, &window_size.x, &window_size.y);
      return window_size;
    }

    void set_size(const fan::vec2i& window_size) {
      fan::window::glfwSetWindowSize(glfw_window, window_size.x, window_size.y);
    }

    void set_windowed() {
      using namespace fan::window;
      GLFWmonitor* monitor = glfwGetPrimaryMonitor();
      const GLFWvidmode* mode = glfwGetVideoMode(monitor);
      //glfwSetWindowSize(glfw_window, windowWidth, windowHeight);
      glfwSetWindowMonitor(glfw_window, NULL, mode->width / 8, mode->height / 8, mode->width / 2, mode->height / 2, mode->refreshRate);
      fan::vec2 screen_size = fan::vec2(mode->width, mode->height);
      fan::vec2 window_pos = (screen_size - get_size()) / 2;
      glfwSetWindowPos(glfw_window, window_pos.x, window_pos.y);
    }

    void set_full_screen() {
      using namespace fan::window;
      GLFWmonitor* monitor = glfwGetPrimaryMonitor();
      const GLFWvidmode* mode = glfwGetVideoMode(monitor);
      glfwSetWindowMonitor(glfw_window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
    }

    void set_windowed_full_screen() {
      using namespace fan::window;
      GLFWmonitor* monitor = glfwGetPrimaryMonitor();
      const GLFWvidmode* mode = glfwGetVideoMode(monitor);

      glfwSetWindowMonitor(glfw_window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
    }

    void set_size_mode(const mode& mode) {
      switch (mode) {
        case mode::windowed: {
          set_windowed();
          break;
        }
        case mode::full_screen: {
          set_full_screen();
          break;
        }
        case mode::borderless: {
          set_windowed_full_screen();
          break;
        }
      }
    }

    fan::vec2d get_mouse_position() const {
      fan::vec2d mouse_pos;
      fan::window::glfwGetCursorPos(glfw_window, &mouse_pos.x, &mouse_pos.y);
      return mouse_pos;
    }

    bool key_pressed(int key) const {
      return fan::window::glfwGetKey(glfw_window, key) == GLFW_PRESS;
    }

    uintptr_t get_fps(bool print = true) {
      ++m_frame_counter;
      if (frame_timer.finished()) {
        if (print) {
          fan::print("fps:", m_frame_counter);
        }
        uintptr_t fps = m_frame_counter;
        m_frame_counter = 0;
        frame_timer.restart();
        return fps;
      }
      return 0;
    }


    double last_frame_time = fan::window::glfwGetTime();
    f64_t m_delta_time = 0;
    uint32_t m_frame_counter = 0;
    fan::time::clock frame_timer;

    buttons_callback_t m_buttons_callback;
    keys_callback_t m_keys_callback;
    key_callback_t m_key_callback;
    text_callback_t m_text_callback;
    move_callback_t m_move_callback;
    resize_callback_t m_resize_callback;
    close_callback_t m_close_callback;
    mouse_position_callback_t m_mouse_position_callback;
    mouse_motion_callback_t m_mouse_motion_callback;

    fan::window::GLFWwindow* glfw_window;
  };
}
inline void fan::window::mouse_button_callback(GLFWwindow* wnd, int button, int action, int mods)
{
  auto found = fan::window_t::window_map.find(wnd);
  if (found != fan::window_t::window_map.end()) {
    fan::window_t* window = found->second;
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
}

inline void fan::window::keyboard_keys_callback(GLFWwindow* wnd, int key, int scancode, int action, int mods)
{
  auto found = fan::window_t::window_map.find(wnd);
  if (found != fan::window_t::window_map.end()) {
    fan::window_t* window = found->second;
    auto it = window->m_keys_callback.GetNodeFirst();

    while (it != window->m_keys_callback.dst) {
      fan::window_t::keyboard_keys_cb_data_t cbd;
      cbd.window = window;
      cbd.key = key;
      cbd.state = static_cast<fan::keyboard_state>(action);
      cbd.scancode = scancode;
      window->m_keys_callback[it].data(cbd);

      it = it.Next(&window->m_keys_callback);
    }
  }
}

//inline void fan::window::key_callback(GLFWwindow* wnd, int key, int scancode, int action, int mods)
//{
//  auto found = fan::window_t::window_map.find(wnd);
//  if (found != fan::window_t::window_map.end()) {
//    fan::window_t* window = found->second;
//    auto it = window->m_key_callback.GetNodeFirst();
//
//    while (it != window->m_key_callback.dst) {
//      fan::window_t::keyboard_key_cb_data_t cbd;
//      cbd.window = window;
//      cbd.key = key;
//      window->m_key_callback[it].data.function(cbd);
//
//      it = it.Next(&window->m_key_callback);
//    }
//  }
//}

inline void fan::window::text_callback(GLFWwindow* wnd, unsigned int codepoint)
{
  auto found = fan::window_t::window_map.find(wnd);
  if (found != fan::window_t::window_map.end()) {
    fan::window_t* window = found->second;
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
}

inline void fan::window::mouse_position_callback(GLFWwindow* wnd, double xpos, double ypos)
{
  auto found = fan::window_t::window_map.find(wnd);
  if (found != fan::window_t::window_map.end()) {
    fan::window_t* window = found->second;
    auto it = window->m_mouse_position_callback.GetNodeFirst();

    while (it != window->m_mouse_position_callback.dst) {
      fan::window_t::mouse_move_cb_data_t cbd;
      cbd.window = window;
      cbd.position = fan::vec2d(xpos, ypos);
      window->m_mouse_position_callback[it].data(cbd);

      it = it.Next(&window->m_mouse_position_callback);
    }
  }
}

//inline void fan::window::cursor_enter_callback(GLFWwindow* wnd, int entered)
//{
//  // Cursor enter callback implementation
//}
//
//inline void fan::window::scroll_callback(GLFWwindow* wnd, double xoffset, double yoffset)
//{
//  // Scroll callback implementation
//}

inline void fan::window::close_callback(GLFWwindow* wnd)
{
  auto found = fan::window_t::window_map.find(wnd);
  if (found != fan::window_t::window_map.end()) {
    fan::window_t* window = found->second;
    auto it = window->m_close_callback.GetNodeFirst();

    while (it != window->m_close_callback.dst) {
      fan::window_t::close_cb_data_t cbd;
      cbd.window = window;
      window->m_close_callback[it].data(cbd);

      it = it.Next(&window->m_close_callback);
    }
  }
}

inline void fan::window::resize_callback(GLFWwindow* wnd, int width, int height)
{
  auto found = fan::window_t::window_map.find(wnd);
  if (found != fan::window_t::window_map.end()) {
    fan::window_t* window = found->second;
    auto it = window->m_resize_callback.GetNodeFirst();

    while (it != window->m_resize_callback.dst) {
      fan::window_t::resize_cb_data_t cbd;
      cbd.window = window;
      cbd.size = fan::vec2i(width, height);
      window->m_resize_callback[it].data(cbd);

      it = it.Next(&window->m_resize_callback);
    }
  }
}

inline void fan::window::move_callback(GLFWwindow* wnd, int xpos, int ypos)
{
  auto found = fan::window_t::window_map.find(wnd);
  if (found != fan::window_t::window_map.end()) {
    fan::window_t* window = found->second;
    auto it = window->m_move_callback.GetNodeFirst();

    while (it != window->m_move_callback.dst) {
      fan::window_t::move_cb_data_t cbd;
      cbd.window = window;
      window->m_move_callback[it].data(cbd);

      it = it.Next(&window->m_move_callback);
    }
  }
}

inline void fan::window::scroll_callback(GLFWwindow* wnd, double xoffset, double yoffset) {
  auto found = fan::window_t::window_map.find(wnd);
  if (found != fan::window_t::window_map.end()) {
    fan::window_t* window = found->second;

    auto it = window->m_buttons_callback.GetNodeFirst();

    while (it != window->m_buttons_callback.dst) {
      fan::window_t::mouse_buttons_cb_data_t cbd;
      cbd.window = window;
      cbd.button = yoffset < 0 ? fan::input::mouse_scroll_down : fan::input::mouse_scroll_up;
      cbd.state = fan::mouse_state::press;
      window->m_buttons_callback[it].data(cbd);

      it = it.Next(&window->m_buttons_callback);
    }
  }
}

//inline void fan::window::window_focus_callback(GLFWwindow* wnd, int focused)
//{
//  // Window focus callback implementation
//}

//inline void fan::window::window_iconify_callback(GLFWwindow* wnd, int iconified)
//{
//  // Window iconify callback implementation
//}

//inline void fan::window::mouse_position_callback(GLFWwindow* wnd, double xpos, double ypos)
//{
//  auto found = fan::window_t::window_map.find(wnd);
//  if (found != fan::window_t::window_map.end()) {
//    fan::window_t* window = found->second;
//    auto it = window->m_mouse_motion_callback.GetNodeFirst();
//
//    while (it != window->m_mouse_motion_callback.dst) {
//      fan::window_t::mouse_motion_cb_data_t cbd;
//      cbd.window = window;
//      cbd.motion = fan::vec2i(static_cast<int>(xpos), static_cast<int>(ypos));
//      window->m_mouse_motion_callback[it].data(cbd);
//
//      it = it.Next(&window->m_mouse_motion_callback);
//    }
//  }
//}