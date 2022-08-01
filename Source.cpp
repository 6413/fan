#include <utility>
#include <type_traits>


struct x_t{
  float operator=(int);
};

template<typename T> class C : public T
{
  using type_t = std::invoke_result_t<decltype(&T::operator=), T>;
};

decltype(((x_t*)nullptr)->operator=({})) x_t::operator=(int x) {
  return 10;
}

int main() {
  x_t x;
  return (x = 5);
}