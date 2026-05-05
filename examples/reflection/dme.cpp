import fan;
import std;
import fan.reflection;
import fan.types.dme;

using a_t = fan::vec2;
struct b_t {
  fan::vec2 pos;
  f64_t weight;
};

struct item_t {
  int number;
};

struct st_t : fan::dme_t<st_t, item_t> {
  [[= a_t{1,   3.6}]] item_t a;
  [[= b_t{{2}, 3.2}]] item_t b;
};

constexpr st_t gst{.a{.number=32},.b{.number=64}};

static_assert(gst.size() == 2);
static_assert(sizeof(st_t) == sizeof(item_t) * 2);

// verify variable names
static_assert(gst.name(0) == "a");
static_assert(gst.name(1) == "b");

// verify variable indices like in enum starting from top: 0, 1 ....
static_assert(gst["a"] == 0);
static_assert(gst["b"] == 1);

// verify actual values of variables
static_assert(gst[0].number == 32);
static_assert(gst[1].number == 64);
static_assert(gst.a.number == 32);
static_assert(gst.b.number == 64);

// verify queried attribute name and type [= type]
constexpr auto attr_type_a  = gst.attr_type(0);
constexpr auto attr_type_b = gst.attr_type(1);
static_assert(fan::refl::is_same_name(attr_type_a, fan::refl::dealias(^^fan::vec2)));
static_assert(fan::refl::is_same_name(attr_type_b, ^^b_t));

static_assert(fan::refl::is_same_type(attr_type_a, ^^fan::vec2));
static_assert(fan::refl::is_same_type(attr_type_b, ^^b_t));

int main() {
  st_t st{
    .a{.number = 1},
    .b{.number = 2}
  };
  st.print();

  st[0].number = 12;
  st[1].number = 15;
  st.print();

  for (auto& member : st) (void)member.number;

  constexpr auto a0 = st.attr<0>();
  constexpr auto a1 = st.attr<1>();
  fan::print_reflect(a0);
  fan::print_reflect(a1);

  for (int i = 0; i < 100; ++i) {
    int idx = fan::random::value(0, 1);
    st.match(idx)({
      .a = [&]{ fan::assert(idx == 0); },
      .b = [&]{ fan::assert(idx == 1); }
    });
  }
}