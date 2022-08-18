#include <fan/types/masterpiece.h>
#include <fan/types/types.h>

struct s_t {
  using type = int;
  type x;
};

int main() {
  fan::masterpiece_t<int, double> x;
  using b = decltype(x)::get_type<0>();
  decltype(decltype(x)::get_type<1>())::type
  fan::print(typeid(b).name());
}