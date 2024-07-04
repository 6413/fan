#include <pch.h>
#include "types.h"

#include <fan/types/print.h>

void fan::assert_test(bool test) {
  if (!test) {
    fan::throw_error("assert failed");
  }
}