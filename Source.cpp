#include <iostream>

uint64_t v() {
  uint64_t total = 1;
  for (uint32_t i = 1; i < 0xfffff; i++) {
    total += total * i;
  }
  return total;
}

int main() {
  return v();
}