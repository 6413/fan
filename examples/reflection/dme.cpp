#include <fan/utility.h>

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
  [[= a_t{1.f, 1.2f}]] item_t a;
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
static_assert(fan::refl::is_same_name(attr_type_a, fan::refl::dealias(^^a_t)));
static_assert(fan::refl::is_same_name(attr_type_b, ^^b_t));

static_assert(fan::refl::is_same_type(attr_type_a, ^^a_t));
static_assert(fan::refl::is_same_type(attr_type_b, ^^b_t));

struct a_ann_t { int x; f32_t y; };
struct b_ann_t { int x; };
using dme_built_t  = fan::dme_builder<item_t, void,
  ^^a_ann_t, fan::fixed_string{"a"},
  ^^b_ann_t, fan::fixed_string{"b"}
>::type;

// dme macro extension for reflection
#include <fan/reflection/dme.h>

struct inside_t {
  int y;
};

__dme(macro_dme_t, item_t,
  a,
  b, (int x; inside_t inside; void init() { fan::print_impl("b constructed"); }),
  c,
  d, (int x; int y;)
);

int main() {
  st_t st{
    .a{.number = 1},
    .b{.number = 2}
  };
  st.print();

  // modify numbers through operator[]
  st[0].number = 12;
  st[1].number = 15;
  st.print();

  for (auto& member : st) (void)member.number;

  // query custom attributes
  fan::print(st.attr<0>());
  fan::print(st.attr<1>());

  for (int i = 0; i < 100; ++i) {
    int idx = fan::random::value(0, 1);
    // missing any match will result in compile error
    st.match(idx)({
      .a = [&]{ fan::assert(idx == 0); },
      .b = [&]{ fan::assert(idx == 1); }
    });
  }

  for (auto i : st.runtime_table()) {
    fan::print(i);
  }

  fan::print(macro_dme_t{});
  macro_dme_t::ann<"b"> v; // calls ann constructor prints "b constructed"
}

// -------OUTPUT-------
/*
st_t {
  .a = item_t {
    .number = 1 [int]
  }
  [[{1.0000, 1.2000} [vec2_wrap_t]]]
  .b = item_t {
    .number = 2 [int]
  }
  [[b_t {
    .pos = {2.0000, 0.0000} [vec2_wrap_t]
    .weight = 3.2 [double]
  }]]
}
st_t {
  .a = item_t {
    .number = 12 [int]
  }
  [[{1.0000, 1.2000} [vec2_wrap_t]]]
  .b = item_t {
    .number = 15 [int]
  }
  [[b_t {
    .pos = {2.0000, 0.0000} [vec2_wrap_t]
    .weight = 3.2 [double]
  }]]
}
{1.0000, 1.2000} [vec2_wrap_t]
b_t {
  .pos = {2.0000, 0.0000} [vec2_wrap_t]
  .weight = 3.2 [double]
}
member_info {
  .name = a [const char*]
  .size = 4 [long unsigned int]
  .offset = 0 [long unsigned int]
  .type = 0x555555b01da0 [const std::type_info*]
}
member_info {
  .name = b [const char*]
  .size = 4 [long unsigned int]
  .offset = 4 [long unsigned int]
  .type = 0x555555b01da0 [const std::type_info*]
}
macro_dme_t {
  .a = item_t {
    .number = 0 [int]
  }
  .b = item_t {
    .number = 0 [int]
  }
  [[_ann_t {
    .x = 0 [int]
    .inside = inside_t {
      .y = 0 [int]
    }
  }]]
  .c = item_t {
    .number = 0 [int]
  }
  .d = item_t {
    .number = 0 [int]
  }
  [[_ann_t {
    .x = 0 [int]
    .y = 0 [int]
  }]]
}
b constructed
*/
