// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 1
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(system.h)

int main() {
  while (!GetAsyncKeyState(VK_SPACE)) { Sleep(100); }
  fan::sys::input input;
  input.send_string("connect 54f84ac8a8f5\n", 1);
  input.send_string("register 1\n", 1);
  input.send_string("123\n", 1);
  input.send_string("login 1 123\n", 1);
  input.send_string("createlobby 0\n", 1);
  input.send_string("joinlobby 0\n", 1);
  input.send_string("createchannel screenshare\n", 1);
  input.send_string("joinchannel 0\n", 1);
  input.send_string("share\n", 1);
  input.send_string("inputcontrol 1\n", 1);
}