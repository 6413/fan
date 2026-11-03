module;

export module fan.window.input_inject;

#if defined(FAN_WINDOW)

import std;

export namespace fan::window {
  struct input_injector_t {
    enum class backend_e : std::uint8_t {
      none,
      win32,
      x11,
      uinput
    };

    input_injector_t() = default;
    input_injector_t(const input_injector_t&) = delete;
    input_injector_t& operator=(const input_injector_t&) = delete;
    ~input_injector_t();

    bool open();
    void close();
    bool is_open() const;
    backend_e get_backend() const;
    const char* get_backend_name() const;

    void set_desktop(std::int32_t x, std::int32_t y, std::int32_t width, std::int32_t height);
    bool move(std::int32_t x, std::int32_t y);
    bool move_relative(std::int32_t dx, std::int32_t dy);
    bool button(std::uint8_t button, bool state);
    bool key(std::uint16_t scancode, bool state);
    void release_all();

    static std::uint16_t scancode_to_evdev(std::uint16_t scancode);

    backend_e backend = backend_e::none;
    void* x11_display = nullptr;
    int uinput_pointer = -1;
    int uinput_keyboard = -1;
    std::int32_t desktop_x = 0;
    std::int32_t desktop_y = 0;
    std::int32_t desktop_width = 0;
    std::int32_t desktop_height = 0;
    std::int32_t last_x = 0;
    std::int32_t last_y = 0;
    std::vector<std::uint16_t> held_keys;
    std::vector<std::uint8_t> held_buttons;
  };
}

#endif