#include <iostream>

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include <array>

//struct stages_t {
//  struct wrapper_t;
//  
//  inline static constexpr stages_t* get(void* ptr) {
//    return OFFSETLESS((wrapper_t*)ptr, stages_t, wrapper);
//  }
//
//  //#define 
//
//  struct wrapper_t : 
//    fan::return_type_of_t<
//      decltype([] {
//        struct stage0_t{
//          using st0 = stage0_t;
//
//            stage0_t() {
//              fan::print(this, get(this));
//            }
//            void f() {
//              fan::print("hi", x[0]);
//            }
//            uint8_t x[10];
//        }v;
//        return v;
//      })
//    >, 
//    fan::return_type_of_t<
//      decltype([] {
//        struct stage1_t{
//          using st1 = stage1_t;
//            stage1_t() {
//              fan::print(this, get(this));
//            }
//            uint8_t x[20];
//        }v;
//        return v;
//      })
//    >
//    
//    {
//
//  }wrapper;
//};

struct pile_t {
  int y;
};

pile_t pile;

struct a_t {
  int x;

  fan::function_t<void()> remove_physics = [this] {
    x = 5;
  };
};


enum {
  enum_x,
  enum_y,
  enum_z
};

template<typename T>
constexpr auto make_array(auto&&... list)
{
  std::array<T, sizeof...(list) / 2> arr{};
  T values[] = { list... };
  for (uint32_t i = 0; i < sizeof...(list); i += 2) {
      arr[values[i]] = values[i + 1];
  }
  return arr;
}

int main() {
  static constexpr uint32_t arr_size = 5;
  static constexpr uint32_t arr_value = 10;
  static constexpr auto c = make_array<int>(enum_z, 1, enum_x, 5, enum_y, 10);
  
  return c[0];
}