#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

using namespace fan;

int main() {
  for (int i = 0; i < 2; ++i) {
    print(int(int16_t(65535 + i)), (65535 + i) % 65536);
  }
}