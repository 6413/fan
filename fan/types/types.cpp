#include "types.h"

import fan.types.print;

void fan::assert_test(bool test) {
  if (!test) {
    fan::throw_error("assert failed");
  }
}