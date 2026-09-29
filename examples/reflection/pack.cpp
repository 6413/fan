import fan;
import fan.reflection;
import std;

struct test_t {
  float a;
  double b;
  const char* c;
};

int main() {
  auto s = fan::refl::make_struct(1, 3.14f, true);
  fan::print(s, typeid(s).name());
  //unpack(std::tuple{1, 2});
  //unpack(std::array{1, 2, 3});
  //unpack(test_t{1.3, 2.4, "test"});
  fan::refl::unpack(std::tuple{1, 2, 3, 4});;
}
