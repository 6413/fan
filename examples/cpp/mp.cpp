#include <global_pch.h>

struct temp_t {
  int x;
};

struct a_t {
  int x;
  float y;
  temp_t c;
};

struct b_t {
  a_t a;
  int* b;
  int c[2];
};

struct recursive0_t {
  b_t b;
};

struct recursive1_t {
  recursive0_t r;
};

struct recursive2_t {
  recursive1_t r;
};

int main() {
  ////
  {
    fan::mp_t<a_t> mp_a{1, 2.2};
    mp_a.iterate([]<auto i>(auto & v) {
     // fan::print("before addition", v);
   //   v += 10;
     // fan::print("after addition", v);
    });
    //std::cout << mp_a;
    fan::print(mp_a);
    //fan::print()
  }
  fan::print("\n");
  {
    fan::mp_t<b_t> mp_b{a_t{1, 2.2}, nullptr, {1, 2}};
    fan::print(mp_b);
  }
  fan::print("\n");
  {
    fan::print(recursive2_t{});
  }

}