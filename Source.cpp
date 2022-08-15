#include <fan/types/types.h>
#include <fan/types/vector.h>

int main() {
  for (uint32_t i = 1; i--; ) {
    fan::print(i);
  }
}