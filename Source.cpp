struct a_t {
  int x;
};

struct b_t {

};

void my_function(a_t& a) {
  a.x = 5;
}

template <void(*T)(a_t&)>
struct gt {
  void f(a_t& a) {
    T(a);
  }
};

int main() {
  a_t a;
  gt<my_function>().f(a);
}