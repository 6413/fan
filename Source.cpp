#include fan_pch

struct a_t {
  int y = 10;
  float z = 15;
};

struct b_t {
  int x = 5;
  a_t a;
};

template<typename T>
concept Addable = 1 == 0;

template <Addable T>
void f(T x) {
  // Your function implementation
}

struct {
  int y;
}a;

void f() {
  if constexpr (false)  {
    
  }
}

int main() {


  //if (false) {}
  //else {}
  //switch (6) {
  //  if (5) { printf("10"); }
  //  printf("10");
  //}
}