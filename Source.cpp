#include fan_pch

struct a_t {
  int y = 10;
  float z = 15;
};

struct b_t {
  int x = 5;
  a_t a;
};

int main() {
  fan::mp_t<b_t> mp;
  a_t* some_value;
  mp.get_value(1, [&]<typename T>(T& v) {
    if constexpr (std::is_same_v<T, a_t>) {
      some_value = &v;
    }
  });
}