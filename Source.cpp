
#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include <variant>
#include <tuple>

template <typename T = __empty_struct>
struct type_t {
  using t = T;
  void open() {
    fan::print("open", typeid(T).name());
  }
  void close() {
    fan::print("close", typeid(T).name());
  }
  ~type_t() {
    close();
  }
};

struct a_t {
  void f() {
    fan::print("a_t");
  }
  int pad[30];
};

struct b_t {
  void f() {
    //fan::print("b_t");
  }
  int pad[60];
};

struct c_t {
  int pad[15];
};

// a_t, b_t

struct main_t {
  struct a_t {

  };
  struct b_t {

  };
};

#define instance(...) \
fan::return_type_of_t<decltype([]{ \
    class { \
    public: \
      __VA_ARGS__ \
    }v; \
    return v; \
  })> 

int main() {
  std::vector<
    std::variant<
    type_t<a_t>*,
    type_t<b_t>*
    >
  > ptr;

  ////x;

  ////List<> list;


  //ptr.push_back(new type_t < a_t>);
  //ptr.push_back(new type_t < b_t>);
  //std::visit([](auto o) {
  //  o->open();
  //}, ptr[0]);
  //std::visit([](auto o) {
  //  delete o;
  //  }, ptr[0]);
}