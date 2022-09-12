#include <fan/types/types.h>

#include <any>
#include <functional>

#include <fan/time/time.h>
#include <type_traits>

//#define test_pushback 0
//#define test_assign 1

void lambda(int x) {

}

int main(int arg) {

  fan::time::clock c;

  constexpr auto count = 1;

#if defined(test_pushback) && test_pushback == 0
  c.start();
  std::vector<std::function<int()>> cpp;
  uint32_t x = 0;
  for (uint32_t i = 0; i < count; i++) {
    cpp.push_back([&] { return i; });
    x += arg * cpp[i]();
  }
  fan::print("pushback 0", c.elapsed());
#elif defined(test_pushback) && test_pushback == 1
  c.start();
  std::vector<std::function<int()>> cpp;
  uint32_t x = 0;
  for (uint32_t i = 0; i < count; i++) {
    cpp.push_back([&] { return i; });
    x += arg * cpp[i]();
  }
  fan::print("pushback 1", c.elapsed());
#endif

#if defined(test_assign) && test_assign == 0
  std::vector<std::function<int()>> cpp;
  cpp.resize(count);
  c.start();
  uint32_t x = 0;
  for (uint32_t i = 0; i < count; i++) {
    cpp[i] = [&] { return i; };
    x += arg * cpp[i]();
  }
  fan::print("assign 0", c.elapsed());
#elif defined(test_assign) && test_assign == 1
  std::vector<fan::function_t<int()>> cpp;
  cpp.resize(count);
  c.start();
  uint32_t x = 0;

  for (uint32_t i = 0; i < count; i++) {
    cpp[i] = [&] { return i; };
    x += arg * cpp[i]();
  }
  fan::print("assign 1", c.elapsed());
#endif

  return x;
}