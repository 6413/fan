#include <fan/pch.h>

struct a_t {
  int x;
  f32_t y;
  fan::string str;
};

struct b_t {
  uint8_t x;
};

struct c_t {
  a_t a;
  b_t b;
};

union g_t {
  int x;
  float y;
  c_t c;
};

int main() {
  std::vector<fan::union_mp<c_t>> mp;
  mp.resize(mp.size() + 1);
  a_t a{.x = 10000, .str = "hello world"};
  /*mp.back().get<a_t>() = a;

  mp[0].current([&]<typename T>(T & v) {
    if constexpr (std::is_same_v<T, a_t>) {
      fan::print(v);
    }
  });*/
  return 0;
}
