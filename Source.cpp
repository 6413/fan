#include <type_traits>

#include <iostream>
#include <type_traits>

#include <iostream>
#include <type_traits>


#define fan_create_id_declaration(rt, name, ...) rt name(__VA_ARGS__)
#define fan_create_id_definition(rt, name, ...) rt a_t::name(__VA_ARGS__)

#define fan_create_get_set(rt, name) \
  fan_create_id_declaration(rt, get_##name); \
  fan_create_id_declaration(void, set_##name, const rt&); \

#define fan_create_get_set_define(rt, name) \
  fan_create_id_definition(rt, get_##name){ return 0;} \
  fan_create_id_definition(void, set_##name, const rt&){} \

#define make_definitions(...) __VA_ARGS__

struct a_t {
  void f() {

  }
};


struct b_t {
  b_t() {
    get_a().f();
  }
  a_t& get_a();
};
a_t a;

a_t& b_t::get_a() {
  return a;
}

int main() {
  a_t a;
}