
#include <iostream>
#include <array>
template <uint8_t n_>
struct textureid_t {
  static constexpr std::array<const char*, 32> texture_names = {
    "_t00", "_t01", "_t02", "_t03",
    "_t04", "_t05", "_t06", "_t07",
    "_t08", "_t09", "_t10", "_t11",
    "_t12", "_t13", "_t14", "_t15"
    "_t16", "_t17", "_t18", "_t19", 
    "_t20", "_t21", "_t22", "_t23",
    "_t24", "_t25", "_t26", "_t27",
    "_t28", "_t29", "_t30", "_t31"
  };
  static constexpr uint8_t n = n_;
  static constexpr auto name = texture_names[n];

};
int main() {
  textureid_t<0> x;
}