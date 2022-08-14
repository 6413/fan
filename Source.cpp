#include <fan/types/types.h>
#include <fan/types/vector.h>

int main() {
  fan::vec2 x(0, -1);
  fan::print(cos(2.1 - x.x));
}