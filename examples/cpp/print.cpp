#include fan_pch

struct a_t {
  int x = 1;
  int y = 2;
};

struct b_t {
  int z = 3;
  a_t a;
};

int main() {
  fan::print(b_t());
}