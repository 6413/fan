#include <iostream>
#include <fan/types/types.h>

template <typename t>
struct st0_t {
  using type_t = t;
};

#define _ProtocolC_t(p0) \
  ProtocolC_t<_return_type_of_t<decltype([] { \
    p0 v; \
    return v;\
  })>>

struct st_t {
  struct st0 : st0_t < _return_type_of_t<
    decltype([] { \
      int v; \
      return v; \
    })
    >> {
  };

  using st1 = st0_t < _return_type_of_t<
  decltype([] { \
    int v; \
    return v; \
    })
  >> ;
};

int main() {
  st_t st_t;


 // st_t::st0::type_t v;
}