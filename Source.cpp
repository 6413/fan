#include <cstdio>
#include <iostream>

#include <fan/types/types.h>


lstd_defstruct(a_t)

  void open() {

  }
  static void f(void *ptr) {
    auto* a = (lstd_current_type*)ptr;
    a->open();
  }
};
int main(){
    a_t a;
    a.f(&a);
}