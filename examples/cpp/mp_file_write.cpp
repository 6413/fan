#include <fan/pch.h>

struct ga_t {
  struct a_t {
    int a = 5;
    f32_t z = 10;
    fan::string b = "a";
  }a;
  struct b_t {
    int a = 10;
    f32_t z = 15;
    fan::string b = "b";
  }b;
};

int main() {

  fan::mp_t<ga_t> a;

  fan::string out;
  a.iterate([&]<auto i, typename T>(T & a) {
    fan::mp_t<T> inner;
    inner.iterate([&]<auto i0, typename T2>(T2 & b) {
      fan::write_to_string(out, b);
    });
  });
  fan::io::file::write("data2", out, std::ios_base::binary);

  out.clear();
  fan::io::file::read("data2", &out);

  fan::mp_t<ga_t> b;
  uint64_t seek = 0;
  b.iterate([&]<auto i, typename T>(T & a) {
    fan::mp_t<T> inner;
    inner.iterate([&]<auto i0, typename T2>(T2 & b) {
      fan::read_from_string(out, seek, b);
    });
  });

  fan::print(b);
}