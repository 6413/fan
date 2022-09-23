#include <string.h>

struct a_t {
  int x;
};

struct b_t {
  int x;
  b_t(a_t a) {
    x = a.x;
  }
  b_t(a_t* a) {
    //(b_t*)(a_t*)(this) = 
    //this = a;
  }
  b_t* operator=(a_t* a) {
    return (b_t*)a;
  }
};

void func(a_t* a) {
  b_t* bptr = *bptr = a;
  bptr->x = 5;
}

int main() {
  a_t a;
  func(&a);
  return a.x;
}