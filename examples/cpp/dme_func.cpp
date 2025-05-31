#include <fan/types/dme.h>

#define DME_VERIFY_PRINT 1

#if DME_VERIFY_PRINT == 1
#include <stdio.h>
#include <typeinfo>
#endif

#define VERIFY_DME 1

#if VERIFY_DME == 1
#include <cassert>
#endif

#include <functional>


#pragma pack(push, 1)

struct data_we_want_t {
  int number;
  std::function<void()> f;
};

//struct dme_t : __dme_inherit(dme_t, data_we_want_t) {
//  __dme(Channel_ScreenShare_Share_ApplyToHostMouseButton, int x;) = { {.number = 5, .f = [] (auto* st) {
//    st->x = 5;
//  }} };
//  __dme(Channel_ScreenShare_Share_ApplyToHostMouseMotion, int y;) = { {.number = 5, .f = [](auto* st) {
//    st->y = 5;
//  }} } };
//  __dme(Channel_ScreenShare_Share_ApplyToHostMouseCoordinate, ) = { {.number = 5} };
//};
#pragma pack(pop)

struct a_t {

};

int main() {
  set_type(float);
  get_type();
  set_type(int);
  get_type();

  get_type() x;
 /* std::function<void(int)> f = [](auto x) {

  };

  f(a_t());*/

  /*dme_t dme;
  [] {

  };*/
}
