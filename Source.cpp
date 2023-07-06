// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(system.h)


int main() {

  fan::sys::input input;
  input.listen_keyboard([&](uint16_t key, fan::keyboard_state state, bool somethign) {
    switch (key) {
      case fan::key_f4: {
        if (!somethign) {
          return;
        }
        fan::sys::input::send_string("hello", 1);
        break;
      }
    }
  });

  input.thread_loop();

  //
}