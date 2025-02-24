#pragma once

#include <fan/types/function.h>
#include <fan/types/vector.h>
#include <fan/window/window_input.h>
#include <fan/window/window_input_common.h>
namespace fan {
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
        throw std::runtime_error("failed to initialize context");
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
}

namespace fan {
  namespace window {

    void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    void keyboard_keys_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    void text_callback(GLFWwindow* window, unsigned int codepoint);
    void move_callback(GLFWwindow* window, int x, int y);
    void resize_callback(GLFWwindow* window, int width, int height);
    void close_callback(GLFWwindow* window);
    void mouse_position_callback(GLFWwindow* window, double xpos, double ypos);
    void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
    void window_focus_callback(GLFWwindow* window, int focused);
    //static void mouse_motion_callback(GLFWwindow* window, double xoffset, double yoffset);
  }
}

namespace fan {
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
      fan::vec2d position;
    };
    using mouse_move_cb_t = fan::function_t<void(const mouse_move_cb_data_t&)>;

    struct mouse_motion_cb_data_t {
      fan::window_t* window;
      fan::vec2d motion;
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
    void open(uint64_t flags);
    void open(fan::vec2i window_size, const std::string& name, uint64_t flags = 0);
    void close();

    void handle_key_states();
    uint32_t handle_events();

    buttons_callback_NodeReference_t add_buttons_callback(mouse_buttons_cb_t function);
    void remove_buttons_callback(buttons_callback_NodeReference_t id);

    keys_callback_NodeReference_t add_keys_callback(keyboard_keys_cb_t function);
    void remove_keys_callback(keys_callback_NodeReference_t id);

    key_callback_NodeReference_t add_key_callback(int key, keyboard_state state, keyboard_key_cb_t function);
    void edit_key_callback(key_callback_NodeReference_t id, int key, keyboard_state state);
    void remove_key_callback(key_callback_NodeReference_t id);

    text_callback_NodeReference_t add_text_callback(text_cb_t function);
    void remove_text_callback(text_callback_NodeReference_t id);

    close_callback_NodeReference_t add_close_callback(close_cb_t function);
    void remove_close_callback(close_callback_NodeReference_t id);

    mouse_position_callback_NodeReference_t add_mouse_move_callback(mouse_move_cb_t function);
    void remove_mouse_move_callback(mouse_position_callback_NodeReference_t id);

    mouse_motion_callback_NodeReference_t add_mouse_motion(mouse_motion_cb_t function);
    void erase_mouse_motion_callback(mouse_motion_callback_NodeReference_t id);

    resize_callback_NodeReference_t add_resize_callback(resize_cb_t function);
    void remove_resize_callback(resize_callback_NodeReference_t id);

    move_callback_NodeReference_t add_move_callback(move_cb_t function);
    void remove_move_callback(move_callback_NodeReference_t idt);

    fan::vec2i get_size() const;

    void set_size(const fan::vec2i& window_size);

    fan::vec2 get_position() const;
    void set_position(const fan::vec2& position);

    void set_windowed();
    void set_fullscreen();
    void set_windowed_fullscreen();
    void set_borderless();

    // 0 disabled 1 enabled
    void set_cursor(int flag);
    void toggle_cursor();

    void set_display_mode(const mode& mode);

    fan::vec2d get_mouse_position() const;

    int key_state(int key) const;
    bool key_pressed(int key, int press = GLFW_PRESS) const;

    fan::vec2 get_gamepad_axis(int key) const;

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

    int prev_key_states[fan::last]{};
    int key_states[fan::last]{};
    GLFWwindow* glfw_window;

    fan::vec2d previous_mouse_position = -0xfff;

    fan::vec2 drag_delta_start = -1;
    uint8_t display_mode = (uint8_t)mode::windowed;
  };
  void handle_key_states();
}