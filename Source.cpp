#include <iostream>

struct a_t {
  struct b_t {
    void f() {
      // how to get a_t here without passing to parameter?
      a_t& a = *(a_t*)((unsigned char*)this - offsetof(a_t, b));
      a.b.f();
    }
  }b;
};

int main() {
  a_t a;
  a.b.f();
}