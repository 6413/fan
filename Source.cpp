#include <iostream>
#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 1
#define _INCLUDE_TOKEN(p0, p1) <p0/p1>
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(types/vector.h)

struct b_t {
  struct a_t {

  };
};


int main() {
  constexpr auto a = fan::ofof(&ProtocolC_t::m_DSS);
  /*switch (0) {
    case Protocol_C2S_t::AN(&Protocol_C2S_t::KeepAlive):{
      break;
    }
  }*/
  //constexpr auto b = &st_t::b;
  //constexpr auto c = b - a;
  //constexpr auto x = st_t::AN(&st_t::a);
  return a;
}