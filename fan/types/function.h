#include <functional>

namespace fan{
  template <typename T>
  using function_t = std::function<T>;
}