#define dme_reflect_type(type) ^^type

#define dme_var_full(name, body) dme_reflect_type(st_raw(body)), fan::fixed_string{#name}
#define dme_var_name(name)       ^^void, fan::fixed_string{#name}

#define dme_var_get(_1, _2, name, ...) name
#define dme_var(...) dme_var_get(__VA_ARGS__, dme_var_full, dme_var_name)(__VA_ARGS__)

#define dme_expand(...) __VA_ARGS__

#define dme_pair(name, body) \
  dme_reflect_type(st_raw(dme_expand body)), fan::fixed_string{#name}

#define dme_expand(...) __VA_ARGS__

#define dme_pair(name, body) \
  dme_reflect_type(st_raw(dme_expand body)), fan::fixed_string{#name}

#define dme_2(name, body) \
  dme_pair(name, body)

#define dme_4(name, body, ...) \
  dme_pair(name, body), dme_2(__VA_ARGS__)

#define dme_6(name, body, ...) \
  dme_pair(name, body), dme_4(__VA_ARGS__)

#define dme_8(name, body, ...) \
  dme_pair(name, body), dme_6(__VA_ARGS__)

#define __dme_get( \
  _1,_2,_3,_4,_5,_6,_7,_8, \
  NAME, ...) NAME

#define dme_select(...) \
  __dme_get(__VA_ARGS__, \
    dme_8, dme_8, \
    dme_6, dme_6, \
    dme_4, dme_4, \
    dme_2)

#define __dme(shared_type, ...) \
  typename fan::dme_builder< \
    shared_type, \
    void, \
    dme_select(__VA_ARGS__)(__VA_ARGS__) \
  >::type
