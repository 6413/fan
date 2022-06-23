#include <fan/types/types.h>

struct instance_t {
  int color;
  int size;
};

instance_t instances[10];

template <typename T>
void get(uint32_t i, T instance_t::*a, T* b) {
  fan::print(*(uintptr_t*)&a, typeid(a).name());
  *b = instances[i].*a;
  //*a = A_.color;
}


int main() {
  instances[0].color = 5;
  int a = 0;
  fan::print(& ((instance_t*)0)->color);
 // a = get(0, &instance_t::color);
  fan::print(a);
}