#include <fan/pch.h>

struct mark_export {};

struct custom_t {
  int y;
  int x;
};

struct aaa_t {
  int y;
};
struct st_t {
  aaa_t x;
  fan::mark<mark_export, uint32_t> y;
  aaa_t z;
};

int main() {
  st_t st;
  fan::iterate_struct(st, []<auto i, typename T>(T & v) {
    if constexpr (fan::is_marked<mark_export, T>) {
      v = 10;
      typename T::type_t x = v;
      fan::print(typeid(v).name(), v, sizeof(v));
    }
  });

}