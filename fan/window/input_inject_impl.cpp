module;

#if defined(FAN_WINDOW)
  #include <cstdint>
  #include <cstring>
  #include <cstdlib>
  #if defined(_WIN32)
    #ifndef NOMINMAX
      #define NOMINMAX
    #endif
    #include <windows.h>
  #elif defined(__linux__)
    #include <dlfcn.h>
    #include <fcntl.h>
    #include <unistd.h>
    #include <sys/ioctl.h>
    #include <linux/uinput.h>
  #endif
#endif

module fan.window.input_inject;

#if defined(FAN_WINDOW)

namespace fan::window {
#if defined(__linux__)
  namespace {
    struct x11_api_t {
      bool load() {
        if (loaded) {
          return ok;
        }
        loaded = true;
        void* x11 = dlopen("libX11.so.6", RTLD_NOW | RTLD_LOCAL);
        void* xtst = dlopen("libXtst.so.6", RTLD_NOW | RTLD_LOCAL);
        if (!x11 || !xtst) {
          return false;
        }
        open_display = (void* (*)(const char*))dlsym(x11, "XOpenDisplay");
        close_display = (int (*)(void*))dlsym(x11, "XCloseDisplay");
        flush = (int (*)(void*))dlsym(x11, "XFlush");
        query_extension = (int (*)(void*, int*, int*, int*, int*))dlsym(xtst, "XTestQueryExtension");
        fake_motion = (int (*)(void*, int, int, int, unsigned long))dlsym(xtst, "XTestFakeMotionEvent");
        fake_relative_motion = (int (*)(void*, int, int, unsigned long))dlsym(xtst, "XTestFakeRelativeMotionEvent");
        fake_button = (int (*)(void*, unsigned int, int, unsigned long))dlsym(xtst, "XTestFakeButtonEvent");
        fake_key = (int (*)(void*, unsigned int, int, unsigned long))dlsym(xtst, "XTestFakeKeyEvent");
        ok = open_display && close_display && flush && query_extension && fake_motion && fake_relative_motion && fake_button && fake_key;
        return ok;
      }

      void* (*open_display)(const char*) = nullptr;
      int (*close_display)(void*) = nullptr;
      int (*flush)(void*) = nullptr;
      int (*query_extension)(void*, int*, int*, int*, int*) = nullptr;
      int (*fake_motion)(void*, int, int, int, unsigned long) = nullptr;
      int (*fake_relative_motion)(void*, int, int, unsigned long) = nullptr;
      int (*fake_button)(void*, unsigned int, int, unsigned long) = nullptr;
      int (*fake_key)(void*, unsigned int, int, unsigned long) = nullptr;
      bool loaded = false;
      bool ok = false;
    }x11;

    constexpr std::int32_t uinput_abs_max = 32767;

    void uinput_emit(int fd, std::uint16_t type, std::uint16_t code, std::int32_t value) {
      input_event event {};
      event.type = type;
      event.code = code;
      event.value = value;
      [[maybe_unused]] auto written = ::write(fd, &event, sizeof(event));
    }

    void uinput_sync(int fd) {
      uinput_emit(fd, EV_SYN, SYN_REPORT, 0);
    }

    int uinput_create(const char* name, bool pointer) {
      int fd = ::open("/dev/uinput", O_WRONLY | O_NONBLOCK | O_CLOEXEC);
      if (fd < 0) {
        return -1;
      }
      bool ok = ioctl(fd, UI_SET_EVBIT, EV_KEY) >= 0 && ioctl(fd, UI_SET_EVBIT, EV_SYN) >= 0;
      if (pointer) {
        for (int code : {BTN_LEFT, BTN_RIGHT, BTN_MIDDLE}) {
          ok = ok && ioctl(fd, UI_SET_KEYBIT, code) >= 0;
        }
        ok = ok && ioctl(fd, UI_SET_EVBIT, EV_REL) >= 0 && ioctl(fd, UI_SET_RELBIT, REL_WHEEL) >= 0;
        ok = ok && ioctl(fd, UI_SET_EVBIT, EV_ABS) >= 0;
        for (int code : {ABS_X, ABS_Y}) {
          uinput_abs_setup abs {};
          abs.code = code;
          abs.absinfo.minimum = 0;
          abs.absinfo.maximum = uinput_abs_max;
          ok = ok && ioctl(fd, UI_SET_ABSBIT, code) >= 0 && ioctl(fd, UI_ABS_SETUP, &abs) >= 0;
        }
      }
      else {
        for (int code = KEY_ESC; code <= KEY_MICMUTE; ++code) {
          ok = ok && ioctl(fd, UI_SET_KEYBIT, code) >= 0;
        }
      }
      uinput_setup setup {};
      setup.id.bustype = BUS_VIRTUAL;
      setup.id.vendor = 0x6413;
      setup.id.product = pointer ? 1 : 2;
      std::strncpy(setup.name, name, UINPUT_MAX_NAME_SIZE - 1);
      if (!ok || ioctl(fd, UI_DEV_SETUP, &setup) < 0 || ioctl(fd, UI_DEV_CREATE) < 0) {
        ::close(fd);
        return -1;
      }
      return fd;
    }
  }
#endif

  input_injector_t::~input_injector_t() {
    close();
  }

  std::uint16_t input_injector_t::scancode_to_evdev(std::uint16_t scancode) {
    std::uint8_t code = scancode & 0xff;
    if (!(scancode & 0xff00)) {
      return code >= 0x01 && code <= 0x58 ? code : 0;
    }
    switch (code) {
      case 0x1c: return 96;
      case 0x1d: return 97;
      case 0x35: return 98;
      case 0x37: return 99;
      case 0x38: return 100;
      case 0x47: return 102;
      case 0x48: return 103;
      case 0x49: return 104;
      case 0x4b: return 105;
      case 0x4d: return 106;
      case 0x4f: return 107;
      case 0x50: return 108;
      case 0x51: return 109;
      case 0x52: return 110;
      case 0x53: return 111;
      case 0x5b: return 125;
      case 0x5c: return 126;
      case 0x5d: return 127;
      default: return 0;
    }
  }

  bool input_injector_t::open() {
    if (is_open()) {
      return true;
    }
  #if defined(_WIN32)
    backend = backend_e::win32;
    return true;
  #elif defined(__linux__)
    auto open_x11 = [this] {
      int event_base, error_base, major, minor;
      if (!std::getenv("DISPLAY") || !x11.load() || !(x11_display = x11.open_display(nullptr))) {
        return false;
      }
      if (!x11.query_extension(x11_display, &event_base, &error_base, &major, &minor)) {
        x11.close_display(x11_display);
        x11_display = nullptr;
        return false;
      }
      backend = backend_e::x11;
      return true;
    };
    auto open_uinput = [this] {
      uinput_pointer = uinput_create("fan virtual pointer", true);
      uinput_keyboard = uinput_pointer < 0 ? -1 : uinput_create("fan virtual keyboard", false);
      if (uinput_keyboard < 0) {
        close();
        return false;
      }
      backend = backend_e::uinput;
      return true;
    };
    return std::getenv("WAYLAND_DISPLAY") ? open_uinput() || open_x11() : open_x11() || open_uinput();
  #else
    return false;
  #endif
  }

  void input_injector_t::close() {
    if (is_open()) {
      release_all();
    }
  #if defined(__linux__)
    if (x11_display) {
      x11.close_display(x11_display);
      x11_display = nullptr;
    }
    for (int* fd : {&uinput_pointer, &uinput_keyboard}) {
      if (*fd >= 0) {
        ioctl(*fd, UI_DEV_DESTROY);
        ::close(*fd);
        *fd = -1;
      }
    }
  #endif
    backend = backend_e::none;
  }

  bool input_injector_t::is_open() const {
    return backend != backend_e::none;
  }

  input_injector_t::backend_e input_injector_t::get_backend() const {
    return backend;
  }

  const char* input_injector_t::get_backend_name() const {
    static constexpr const char* names[] = {"none", "win32", "x11", "uinput"};
    return names[(std::uint8_t)backend];
  }

  void input_injector_t::set_desktop(std::int32_t x, std::int32_t y, std::int32_t width, std::int32_t height) {
    desktop_x = x;
    desktop_y = y;
    desktop_width = width;
    desktop_height = height;
  }

  bool input_injector_t::move(std::int32_t x, std::int32_t y) {
    last_x = x;
    last_y = y;
    switch (backend) {
    #if defined(_WIN32)
      case backend_e::win32: {
        return SetCursorPos(x, y) != 0;
      }
    #elif defined(__linux__)
      case backend_e::x11: {
        x11.fake_motion(x11_display, -1, x, y, 0);
        x11.flush(x11_display);
        return true;
      }
      case backend_e::uinput: {
        if (desktop_width <= 0 || desktop_height <= 0) {
          return false;
        }
        auto scale = [](std::int64_t v, std::int64_t size) {
          return (std::int32_t)(std::min(std::max(v, (std::int64_t)0), size - 1) * uinput_abs_max / std::max(size - 1, (std::int64_t)1));
        };
        uinput_emit(uinput_pointer, EV_ABS, ABS_X, scale(x - desktop_x, desktop_width));
        uinput_emit(uinput_pointer, EV_ABS, ABS_Y, scale(y - desktop_y, desktop_height));
        uinput_sync(uinput_pointer);
        return true;
      }
    #endif
      default: {
        return false;
      }
    }
  }

  bool input_injector_t::move_relative(std::int32_t dx, std::int32_t dy) {
    switch (backend) {
    #if defined(_WIN32)
      case backend_e::win32: {
        INPUT input {};
        input.type = INPUT_MOUSE;
        input.mi.dx = dx;
        input.mi.dy = dy;
        input.mi.dwFlags = MOUSEEVENTF_MOVE;
        return SendInput(1, &input, sizeof(input)) == 1;
      }
    #elif defined(__linux__)
      case backend_e::x11: {
        x11.fake_relative_motion(x11_display, dx, dy, 0);
        x11.flush(x11_display);
        return true;
      }
      case backend_e::uinput: {
        return move(last_x + dx, last_y + dy);
      }
    #endif
      default: {
        return false;
      }
    }
  }

  bool input_injector_t::button(std::uint8_t button, bool state) {
    if (button > 4 || !is_open()) {
      return false;
    }
    bool wheel = button >= 3;
    if (!wheel) {
      auto it = std::find(held_buttons.begin(), held_buttons.end(), button);
      if (state == (it != held_buttons.end())) {
        return true;
      }
      state ? held_buttons.push_back(button) : (void)held_buttons.erase(it);
    }
    else if (!state) {
      return true;
    }
    switch (backend) {
    #if defined(_WIN32)
      case backend_e::win32: {
        static constexpr DWORD down[] = {MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_MIDDLEDOWN};
        static constexpr DWORD up[] = {MOUSEEVENTF_LEFTUP, MOUSEEVENTF_RIGHTUP, MOUSEEVENTF_MIDDLEUP};
        INPUT input {};
        input.type = INPUT_MOUSE;
        input.mi.dwFlags = wheel ? MOUSEEVENTF_WHEEL : state ? down[button] : up[button];
        input.mi.mouseData = wheel ? (button == 3 ? WHEEL_DELTA : -WHEEL_DELTA) : 0;
        return SendInput(1, &input, sizeof(input)) == 1;
      }
    #elif defined(__linux__)
      case backend_e::x11: {
        static constexpr unsigned int x11_buttons[] = {1, 3, 2, 4, 5};
        x11.fake_button(x11_display, x11_buttons[button], state, 0);
        if (wheel) {
          x11.fake_button(x11_display, x11_buttons[button], false, 0);
        }
        x11.flush(x11_display);
        return true;
      }
      case backend_e::uinput: {
        static constexpr std::uint16_t evdev_buttons[] = {BTN_LEFT, BTN_RIGHT, BTN_MIDDLE};
        if (wheel) {
          uinput_emit(uinput_pointer, EV_REL, REL_WHEEL, button == 3 ? 1 : -1);
        }
        else {
          uinput_emit(uinput_pointer, EV_KEY, evdev_buttons[button], state);
        }
        uinput_sync(uinput_pointer);
        return true;
      }
    #endif
      default: {
        return false;
      }
    }
  }

  bool input_injector_t::key(std::uint16_t scancode, bool state) {
    auto it = std::find(held_keys.begin(), held_keys.end(), scancode);
    if (!state && it == held_keys.end()) {
      return true;
    }
    bool sent = false;
    switch (backend) {
    #if defined(_WIN32)
      case backend_e::win32: {
        if (!(scancode & 0xff)) {
          return false;
        }
        INPUT input {};
        input.type = INPUT_KEYBOARD;
        input.ki.wScan = scancode & 0xff;
        input.ki.dwFlags = KEYEVENTF_SCANCODE | ((scancode & 0xff00) ? KEYEVENTF_EXTENDEDKEY : 0) | (state ? 0 : KEYEVENTF_KEYUP);
        sent = SendInput(1, &input, sizeof(input)) == 1;
        break;
      }
    #elif defined(__linux__)
      case backend_e::x11:
      case backend_e::uinput: {
        std::uint16_t code = scancode_to_evdev(scancode);
        if (!code) {
          return false;
        }
        if (backend == backend_e::x11) {
          x11.fake_key(x11_display, code + 8, state, 0);
          x11.flush(x11_display);
        }
        else {
          uinput_emit(uinput_keyboard, EV_KEY, code, state);
          uinput_sync(uinput_keyboard);
        }
        sent = true;
        break;
      }
    #endif
      default: {
        return false;
      }
    }
    if (sent && state && it == held_keys.end()) {
      held_keys.push_back(scancode);
    }
    else if (sent && !state) {
      held_keys.erase(it);
    }
    return sent;
  }

  void input_injector_t::release_all() {
    for (auto scancode : std::vector<std::uint16_t>(held_keys)) {
      key(scancode, false);
    }
    for (auto held : std::vector<std::uint8_t>(held_buttons)) {
      button(held, false);
    }
    held_keys.clear();
    held_buttons.clear();
  }
}

#endif