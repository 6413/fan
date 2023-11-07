#include fan_pch

struct l_t {
  struct shapes_t {
    struct r_t {
      int x;
      void f(int) {
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
  fan::mp_t<shapes_t> shapes;
};


int main() {
  l_t l;
  l.shapes.iterate([]<auto i>(auto & v) {
    fan_if_has_function(&v, f, (5));
  });
}