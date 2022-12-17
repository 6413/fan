#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 1
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

struct a_t {

};

struct : a_t {

}x;

int main() {
  Sleep(2000);

  INPUT input[2];
  uint8_t i = 0;
  input[i].type = INPUT_KEYBOARD;
  input[i].ki.wVk = 0;
    
  input[i].ki.wScan = 0;
  input[i].ki.dwFlags = KEYEVENTF_SCANCODE;
  input[i].ki.time = 0;
  input[i].ki.dwExtraInfo = 0;
  i++;
  input[i].type = INPUT_KEYBOARD;
  input[i].ki.wVk = 0;
    
  input[i].ki.wScan = 0xe04b; // 73
  input[i].ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY;
  input[i].ki.time = 0;
  input[i].ki.dwExtraInfo = 0;

  SendInput(std::size(input), &input[0], sizeof(INPUT));
}