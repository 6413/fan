#include <compare>
#include <iostream>
#include <stack>

struct point {
  uint8_t bit : 1000;
  auto operator<=>(const point&) const = default;
  int x, y, z;
};

int main(){
  point a = point(1, 2, 3);
  point b = point(3, 2, 1);
  //std::cout << a > b;

  return 0;
}
