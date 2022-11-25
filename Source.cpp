#include <fan/types/masterpiece.h>
#include <fan/types/types.h>
#include <fan/types/vector.h>
#include <tuple>
struct base_t{
  int x;
};

template <typename ...T>
auto f(T ...args){
  struct{
    base_t base;
    uint8_t x[(sizeof(T) + ... + 0)]{0};
  }pack;
  int i = 0;
  constexpr auto l = [](auto& pack, auto&i, auto args) {
    for (uint32_t j = 0; j < sizeof(args); ++j) {
      pack.x[i] = *(uint8_t*)&args;
    }
    i += sizeof(args);
  };
  (l(pack, i, args), ...);
}

int main(){
  fan::vec2ui m0 = 5;
  f32_t m1 = 6;
  f(m0, m1);
}