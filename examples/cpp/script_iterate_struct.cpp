#include fan_pch

struct a_t {
  int x = 5;
  int y = 10;
  std::string is_it_bad_var = "is_it_bad";
  int z = fan::random::value_i64(0, 1000);
};

dme b_t {
  a{int x = 5; },
  v{int y = 5; double z = 1.3; },
};

int main() {
  b_t b;
  b.a.x;

}
