#include <iostream>
#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 1
#define _INCLUDE_TOKEN(p0, p1) <p0/p1>
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(types/vector.h)

#include <functional>

//union x{
//  x() {  }
//  ~x() {}
//  struct _{
//    _()  {}
//    ~_() {  }
//
//    void open() {
//      f = std::function<int()>();
//    }
//    void close() {
//
//    }
//    std::function<int()> f;
//  }a;
//};

//#include <variant>
//
//union u_t{
//  float x;
//  int y;
//};
//
//struct a_t{
//  std::vector<int> f;
//};
//
//struct b_t {
//  std::vector<int> f;
//};
//
//void f(auto& a) {
//  if (rand() & 1) {
//    a = a_t();
//    auto& v = std::get<a_t>(a);
//    for (uint32_t i = 0; i < 250000000; i++) {
//      v.f.push_back(i);
//    }
//  }
//  else {
//    a = b_t();
//    auto& v = std::get<b_t>(a);
//    for (uint32_t i = 0; i < 250000000; i++) {
//      v.f.push_back(i);
//    }
//  }
//}

//struct a_t {
//  int x;
//  void f() {
//    fan::print(this);
//  }
//};
//
//struct b_t {
//  int y;
//  void f() {
//    fan::print(this);
//  }
//};
//
//struct c_t : a_t, b_t {
//  int z;
//  void f() {
//    a_t::f();
//    b_t::f();
//  }
//};


struct _t {
  //std::vector<int> x;
};

struct a_t {
  _t _____________;
  uint8_t x[fan::conditional_value_t<sizeof(_t) < 4, 4-sizeof(_t), 0>::value];
};

int main() {
  a_t t;
  sizeof(t);
  fan::print(32);
  //fan::print();
  //u_t a;
  //a.x = 5;
  //std::variant<a_t, b_t> a;
  //f(a);
  //a = a_t();
  //auto& v = std::get<a_t>(a);
  //v.f.push_back(5);
  //fan::print(v.f[0]);
  //a = b_t();
  //auto& vb = std::get<b_t>(a);
  //v.f.push_back(10);
  //fan::print(vb.f[0]);
 //// x a;
 // a.a.open();
 // a.a.f = []() {
 //   return 5;
 // };
 // return a.a.f();
}