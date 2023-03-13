
#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include <type_traits>

#include <iostream>
#include <type_traits>

#include <iostream>
#include <type_traits>
#include <variant>


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
  a_t() {
    fan::print("a construct");
  }
  ~a_t() {
    fan::print("a destruct");
  }
};

struct b_t {
  b_t() {
    fan::print("b construct");
  }
  ~b_t() {
    fan::print("b destruct");
  }
};

struct e_t{
  #define BLL_set_declare_NodeReference 1
  #define BLL_set_declare_rest 0
  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix cid_list
  #define BLL_set_type_node uint32_t
  #define BLL_set_AreWeInsideStruct 1
  #include _FAN_PATH(BLL/BLL.h)

  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_declare_NodeReference 0
  #define BLL_set_declare_rest 1
  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix cid_list
  #define BLL_set_type_node uint32_t
  #define BLL_set_NodeData a_t a; b_t b; std::variant<a_t, b_t> var;
  #define BLL_set_Link 1
  #define BLL_set_AreWeInsideStruct 1
  #include _FAN_PATH(BLL/BLL.h)
};

struct a_t {

};

fan_has_variable_struct(x);

int main() {

  if constexpr (has_x<a_t>::value) {
    return 1;
  }

  return 0;
}