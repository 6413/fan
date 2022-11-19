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

struct b_t {
  int y;
};

struct st_t {
  int x;
};

struct type_t {
  b_t& b;
  type_t(st_t* bc) : b(*(b_t*)bc) {
    
  }
  b_t* operator->() {
    return &b;
  }
};

void f(type_t b) {
  fan::print(&b->y);
}


struct a_t {
  cx_t t;
  //void f() {
  //  t.y = x;
  //}
  struct cx_t {

  };
};

int main() {
  st_t t;
  t.x = 5;
  fan::print(&t.x);
  f(&t);
}