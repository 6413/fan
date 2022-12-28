#include <iostream>
#include <functional>

void foo() {
  std::cout << "nothing" << std::endl;
}

void foo(int value) {
  std::cout << value << std::endl;
}

void f() {
  std::cout << "nothing" << std::endl;
}

void f(int value, int value2) {
  std::cout << value << value2 << std::endl;
}

template<typename Callable>
using return_type_of_t = typename decltype(std::function{ std::declval<Callable>() })::result_type;

template<typename T = void, typename... Args>
constexpr auto get_type(std::remove_pointer_t<return_type_of_t<decltype([](Args&&...) -> auto {
  using _t = void(Args...);
  return (_t*)0;
  })>> f, Args&&... args) {
  return static_cast<decltype(f(std::declval<Args>()...))(*)(Args...)>(f);
}

int main() {
  auto func = get_type(foo, 5);
  func(5);

  auto func2 = get_type(foo);
  func2();

  auto func3 = get_type(f);
  func3();

  auto func4 = get_type(f, 5, 10);
  func4(5, 10);

  return 0;
}
