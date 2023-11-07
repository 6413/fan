#include fan_pch

struct l_t;

template <typename l_t>
struct shapes_t {
  struct r_t {
    int x;
    void f(int) {
      l_t::type_t x;
      fan::print("a");
    }
  };
  r_t r;
  struct s_t {
    int x;
    void f() {
      fan::print("b");
    }
  };
  s_t s;
};

struct l_t : shapes_t<l_t> {
  using type_t = int;
  fan::mp_t<shapes_t> shapes;
};

int main() {
  l_t l;
}