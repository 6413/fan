#include <stdio.h>
#include <utility>
#include <stdlib.h>
#include <vector>

struct sub0_t {
  sub0_t& operator=(sub0_t&&) {
    printf("hi\n");
    return *this;
  }
};
struct sub1_t {
  sub1_t& operator=(sub1_t&&) {
    return *this;
  }
};

struct st_t {

};

void func(st_t* st) {

}

int main() {
  st_t st;
  while (1) {
    func(&st);
  }
}