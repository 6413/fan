
#pragma once

#define _dme_expand(...) _dme_e1(_dme_e1(_dme_e1(_dme_e1(__VA_ARGS__))))
#define _dme_e1(...) _dme_e2(_dme_e2(_dme_e2(_dme_e2(__VA_ARGS__))))
#define _dme_e2(...) _dme_e3(_dme_e3(_dme_e3(_dme_e3(__VA_ARGS__))))
#define _dme_e3(...) _dme_e4(_dme_e4(_dme_e4(_dme_e4(__VA_ARGS__))))
#define _dme_e4(...) __VA_ARGS__

#define _dme_empty()
#define _dme_defer(id) id _dme_empty()
#define _dme_obstruct(...) __VA_ARGS__ _dme_defer(_dme_empty)()
#define _dme_parens ()

#define _dme_cat(a,b) _dme_cat_i(a,b)
#define _dme_cat_i(a,b) a##b

#define _dme_first(a, ...) a
#define _dme_rest(a, ...) __VA_ARGS__

#define _dme_is_paren(x) _dme_is_paren_check(_dme_is_paren_probe x)
#define _dme_is_paren_probe(...) ~, 1
#define _dme_is_paren_check(...) _dme_is_paren_check_n(__VA_ARGS__, 0)
#define _dme_is_paren_check_n(x, n, ...) n

#define _dme_strip_parens(...) _dme_strip_parens_i __VA_ARGS__
#define _dme_strip_parens_i(...) __VA_ARGS__

#define _dme_push_0(shared_t, sname) \
  _dme_specs.push_back(std::meta::data_member_spec( \
    ^^shared_t, {.name = std::string(sname)}));

#define _dme_push_1(shared_t, sname, body) \
  _dme_specs.push_back(std::meta::data_member_spec( \
    ^^shared_t, { \
      .name = std::string(sname), \
      .annotations = { \
        std::meta::reflect_constant([]{ \
          struct { _dme_strip_parens(body) } s{}; \
          return s; \
        }()) \
      } \
    }));

#define _dme_next() _dme_helper

#define _dme_helper(shared_t, name, ...) \
  _dme_dispatch( \
    _dme_is_paren(_dme_first(__VA_ARGS__)), \
    shared_t, \
    name, \
    __VA_ARGS__ \
  )

#define _dme_dispatch(has, ...) \
  _dme_cat(_dme_h, has)(__VA_ARGS__)

#define _dme_h0(shared_t, name, ...) \
  _dme_push_0(shared_t, #name) \
  __VA_OPT__( \
    _dme_obstruct(_dme_next) () (shared_t, __VA_ARGS__) \
  )

#define _dme_h1(shared_t, name, body, ...) \
  _dme_push_1(shared_t, #name, body) \
  __VA_OPT__( \
    _dme_obstruct(_dme_next) () (shared_t, __VA_ARGS__) \
  )

#define __dme(type_name, shared_type, ...) \
  struct type_name; \
  consteval { \
    std::vector<std::meta::info> _dme_specs; \
    _dme_expand( \
      _dme_helper(shared_type, __VA_ARGS__) \
    ) \
    std::meta::define_aggregate(^^type_name, _dme_specs); \
  }
