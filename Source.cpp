#include <iostream>
#include <functional>

struct s_t {
  void mf(float) {
    std::cout << "member function" << std::endl;
  }
};

void f(int)
{
  std::cout << "int" << std::endl;
}

void f(std::string)
{
  std::cout << "string" << std::endl;
}

#define get_type(_func, ...) [&] (auto&&... args)  { return _func(args...); }

int main()
{
  auto lf = get_type(f, 5);
  lf(5);

  auto lf2 = get_type(f, "");
  lf2("");

  s_t s;

  auto lf3 = get_type(s.mf, 5);
  lf3(3.0);
}