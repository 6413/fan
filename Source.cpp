#include <fan/types/types.h>
#include <fan/types/vector.h>

struct A{
  static constexpr auto a = 0;
  static constexpr auto b = 1;

};

struct R{
  static constexpr auto c = 0;
  static constexpr auto d = 1;
};

struct g{
  A a;
  R r;
};


void f() {

}

int main() {
  g g;
}