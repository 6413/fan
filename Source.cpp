#include <iostream>
#include <type_traits>

#define xy \
  int x = 5; \
  int y = 10;

struct a_t {

  union {
    struct { xy } vi;
    
  };
};


int main() {
 /* a_t* a = (a_t*)malloc(sizeof(a_t));
  std::construct_at(a);
  std::cout << (a.vi.x);*/
}
