#include "window.h"

fan::window_t::glfw_initialize_t intialize_glfw_var;

void fan::window::mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
  auto found = fan::window_t::window_map.find(window);
  if (found != fan::window_t::window_map.end()) {
    fan::window_t* pwindow = found->second;
    auto it = pwindow->m_buttons_callback.GetNodeFirst();

    while (it != pwindow->m_buttons_callback.dst) {
      fan::window_t::mouse_buttons_cb_data_t cbd;
      cbd.window = pwindow;
      cbd.button = button;
      cbd.state = static_cast<fan::mouse_state>(action);
      pwindow->m_buttons_callback[it].data(cbd);

      it = it.Next(&pwindow->m_buttons_callback);
    }
  }
}

void fan::window::keyboard_keys_callback(GLFWwindow* wnd, int key, int scancode, int action, int mods)
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

void fan::window::text_callback(GLFWwindow* wnd, unsigned int codepoint)
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

void fan::window::mouse_position_callback(GLFWwindow* wnd, double xpos, double ypos)
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

void fan::window::close_callback(GLFWwindow* wnd)
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

void fan::window::resize_callback(GLFWwindow* wnd, int width, int height)
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

void fan::window::move_callback(GLFWwindow* wnd, int xpos, int ypos)
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

void fan::window::scroll_callback(GLFWwindow* wnd, double xoffset, double yoffset) {
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

void fan::window_t::glfw_initialize_t::open() {
  using namespace fan::window;
  if (glfwInit() == false) {
    fan::throw_error("failed to initialize window manager context");
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, fan::window_t::gl_major);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, fan::window_t::gl_minor);

  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, true);

  glfwSetErrorCallback(error_callback);
  initialized = true;
}

fan::window_t::glfw_initialize_t::~glfw_initialize_t() {
  using namespace fan::window;
  glfwTerminate();
}

fan::window_t::window_t() : window_t(fan::window_t::default_window_size, fan::window_t::default_window_name, 0) {}

fan::window_t::window_t(const fan::vec2i& window_size, const fan::string& name, uint64_t flags) {
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

void fan::window_t::close() {
  glfwDestroyWindow(glfw_window);
}

uint32_t fan::window_t::handle_events() {
  f64_t current_frame_time = glfwGetTime();
  m_delta_time = current_frame_time - last_frame_time;
  last_frame_time = current_frame_time;

  glfwPollEvents();
  return 0;
}

fan::window_t::buttons_callback_NodeReference_t fan::window_t::add_buttons_callback(mouse_buttons_cb_t function) {
  auto nr = m_buttons_callback.NewNodeLast();

  m_buttons_callback[nr].data = function;
  return nr;
}

void fan::window_t::remove_buttons_callback(buttons_callback_NodeReference_t id) {
  m_buttons_callback.Unlink(id);
  m_buttons_callback.Recycle(id);
}

fan::window_t::keys_callback_NodeReference_t fan::window_t::add_keys_callback(keyboard_keys_cb_t function) {
  auto nr = m_keys_callback.NewNodeLast();
  m_keys_callback[nr].data = function;
  return nr;
}

void fan::window_t::remove_keys_callback(keys_callback_NodeReference_t id) {
  m_keys_callback.Unlink(id);
  m_keys_callback.Recycle(id);
}

fan::window_t::key_callback_NodeReference_t fan::window_t::add_key_callback(int key, keyboard_state state, keyboard_key_cb_t function) {
  auto nr = m_key_callback.NewNodeLast();
  m_key_callback[nr].data = keyboard_cb_store_t{ key, state, function, };
  return nr;
}

void fan::window_t::edit_key_callback(key_callback_NodeReference_t id, int key, keyboard_state state) {
  m_key_callback[id].data.key = key;
  m_key_callback[id].data.state = state;
}

void fan::window_t::remove_key_callback(key_callback_NodeReference_t id) {
  m_key_callback.unlrec(id);
}

fan::window_t::text_callback_NodeReference_t fan::window_t::add_text_callback(text_cb_t function) {
  auto nr = m_text_callback.NewNodeLast();
  m_text_callback[nr].data = function;
  return nr;
}

void fan::window_t::remove_text_callback(text_callback_NodeReference_t id) {
  m_text_callback.Unlink(id);
  m_text_callback.Recycle(id);
}

fan::window_t::close_callback_NodeReference_t fan::window_t::add_close_callback(close_cb_t function) {
  auto nr = m_close_callback.NewNodeLast();
  m_close_callback[nr].data = function;
  return nr;
}

void fan::window_t::remove_close_callback(close_callback_NodeReference_t id) {
  m_close_callback.Unlink(id);
  m_close_callback.Recycle(id);
}

fan::window_t::mouse_position_callback_NodeReference_t fan::window_t::add_mouse_move_callback(mouse_move_cb_t function) {
  auto nr = m_mouse_position_callback.NewNodeLast();
  m_mouse_position_callback[nr].data = function;
  return nr;
}

void fan::window_t::remove_mouse_move_callback(mouse_position_callback_NodeReference_t id) {
  m_mouse_position_callback.Unlink(id);
  m_mouse_position_callback.Recycle(id);
}

fan::window_t::mouse_motion_callback_NodeReference_t fan::window_t::add_mouse_motion(mouse_motion_cb_t function) {
  auto nr = m_mouse_motion_callback.NewNodeLast();
  m_mouse_motion_callback[nr].data = function;
  return nr;
}

void fan::window_t::erase_mouse_motion_callback(mouse_motion_callback_NodeReference_t id) {
  m_mouse_motion_callback.Unlink(id);
  m_mouse_motion_callback.Recycle(id);
}

fan::window_t::resize_callback_NodeReference_t fan::window_t::add_resize_callback(resize_cb_t function) {
  auto nr = m_resize_callback.NewNodeLast();
  m_resize_callback[nr].data = function;
  return nr;
}

void fan::window_t::remove_resize_callback(resize_callback_NodeReference_t id) {
  m_resize_callback.Unlink(id);
  m_resize_callback.Recycle(id);
}

fan::window_t::move_callback_NodeReference_t fan::window_t::add_move_callback(move_cb_t function) {
  auto nr = m_move_callback.NewNodeLast();
  m_move_callback[nr].data = function;
  return nr;
}

void fan::window_t::remove_move_callback(move_callback_NodeReference_t idt) {
  m_move_callback.unlrec(idt);
}

fan::vec2i fan::window_t::get_size() const {
  fan::vec2i window_size;
  glfwGetWindowSize(glfw_window, &window_size.x, &window_size.y);
  return window_size;
}

void fan::window_t::set_size(const fan::vec2i& window_size) {
  glfwSetWindowSize(glfw_window, window_size.x, window_size.y);
}

void fan::window_t::set_windowed() {
  using namespace fan::window;
  GLFWmonitor* monitor = glfwGetPrimaryMonitor();
  const GLFWvidmode* mode = glfwGetVideoMode(monitor);
  //glfwSetWindowSize(glfw_window, windowWidth, windowHeight);
  glfwSetWindowMonitor(glfw_window, NULL, mode->width / 8, mode->height / 8, mode->width / 2, mode->height / 2, mode->refreshRate);
  fan::vec2 screen_size = fan::vec2(mode->width, mode->height);
  fan::vec2 window_pos = (screen_size - get_size()) / 2;
  glfwSetWindowPos(glfw_window, window_pos.x, window_pos.y);
}

void fan::window_t::set_full_screen() {
  using namespace fan::window;
  GLFWmonitor* monitor = glfwGetPrimaryMonitor();
  const GLFWvidmode* mode = glfwGetVideoMode(monitor);
  glfwSetWindowMonitor(glfw_window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
}

void fan::window_t::set_windowed_full_screen() {
  using namespace fan::window;
  GLFWmonitor* monitor = glfwGetPrimaryMonitor();
  const GLFWvidmode* mode = glfwGetVideoMode(monitor);

  glfwSetWindowMonitor(glfw_window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
}

void fan::window_t::set_size_mode(const mode& mode) {
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

fan::vec2d fan::window_t::get_mouse_position() const {
  fan::vec2d mouse_pos;
  glfwGetCursorPos(glfw_window, &mouse_pos.x, &mouse_pos.y);
  return mouse_pos;
}

bool fan::window_t::key_pressed(int key) const {
  return glfwGetKey(glfw_window, key) == GLFW_PRESS;
}

uintptr_t fan::window_t::get_fps(bool print) {
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
