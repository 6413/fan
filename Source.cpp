#include <iostream>
#include <algorithm>
#include <vector>
#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 1
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include <fan/types/vector.h>
#include <fan/math/math.h>

//struct st_t{
//  int m_x;
//  void f(int x){
//    m_x = x;
//    if(m_x != x){
//      fan::throw_error("a");
//    }
//  }
//};
//
//std::atomic<st_t> st;
//
//void t0(){
//  while(1){
//    st.load().f(5);
//  }
//}
//void t1(){
//  while(1){
//    st.load().f(6);
//  }
//}
//#include <thread>
//int main(){
//  std::thread tt0(t0);
//  std::thread tt1(t1);
//  tt0.detach();
//  tt1.join();
//
//  return 0;
//}

struct a_t {
  void f() {

  }
};
struct b_t {
  void f() {

  }
};
struct c_t : a_t, b_t {
  void f() {
    a_t::f();
    b_t::f();
  }
};

int main() {
  c_t c;
  c.f();
}