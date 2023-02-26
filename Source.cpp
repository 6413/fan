#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(types/masterpiece.h)

#include <WITCH/WITCH.h>

#include <iostream>
#include <stdio.h>



struct EmptyStruct {
  static constexpr const char& zero() { return reinterpret_cast<const char&>(0); }
};


int main() {
 // sizeof(empt);
  //fan::print(EmptyStruct);
  //printf("%lu\n", __sizeof(__sizeof0_struct));
}
