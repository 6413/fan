
import fan;
import fan.reflection;

import std;

struct a_t {
  int a;
  double b;
  char c;
};

struct b_t {
  b_t() = default;
  int d;
  double e;
  char f;
  a_t a;
private:
  int aasdf;
};

template <typename T>
struct dme_t {
  consteval auto size() { return fan::refl::member_count<T>(); }
  constexpr auto& operator[](this auto&& self, std::size_t i) { return fan::refl::at_index<T>(self, i);}
  constexpr auto begin() { return &(*this)[0]; }
  constexpr auto end()   { return &(*this)[0] + size(); }
  constexpr std::string_view name(std::size_t i) const {
    std::string_view result;
    std::size_t cur = 0;
    template for (constexpr auto m : fan::refl::members<T>()) {
      if (cur++ == i) result = std::meta::identifier_of(m);
    }
    return result;
  }
  void print() { fan::print_reflect(static_cast<T&>(*this)); }
};

struct _t {
  int number;
};

struct st_t : dme_t<st_t> {
  _t a{.number=5};
  _t b{.number=5};
};

int main() {
  st_t st;
  st[0].number = 12;
  st[1].number = 15;
  static_assert(st.size() == 2);
  st.print();
  constexpr auto a_str = st.name(0);
  constexpr auto b_str = st.name(1);
  fan::print(a_str, b_str);

  int index = 0;
  for (auto& m : st) {
    fan::print(m.number);
    m.number = 20 + index;
    fan::print(m.number);
  }
}
