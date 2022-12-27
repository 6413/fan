template <typename, typename>
inline constexpr bool __is_type_same = false;
template <typename _Ty>
inline constexpr bool __is_type_same<_Ty, _Ty> = true;

void f(auto p) {
  constexpr bool val = __is_type_same<int, decltype(p)> ||
    __is_type_same<int, decltype(p)> ||
    __is_type_same<int, decltype(p)>;
  static_assert(val);
  if constexpr (val) {
    int x = p;
  }
}

int main() {
  f(0);
}