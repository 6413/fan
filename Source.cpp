struct a_t {
  int y = 10;
  float z = 15;
 // void f() {}
};

int main() {
  if constexpr (fan_has_function(a_t, f)) {
    return 0;
  }
  return 1;
}