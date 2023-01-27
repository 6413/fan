#include <stdlib.h>


#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)


int main() {
  int luvut[] = { 1, 2, 3 };
  int pienin = luvut[0];
  int* pienin_p2 = &pienin;

  fan::print(&luvut[0], pienin_p2);
  system("pause");
}