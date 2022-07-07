//#include <fan/types/types.h>
#include <cstdint>
#include <iostream>
#include <fan/types/types.h>
struct rectangle;
struct sprite;

template <typename type_t, typename instance_data_t>
using global_cb_t = void(*)(type_t* ptr, uint32_t src, uint32_t dst, instance_data_t*);

struct draw_pile_t;

struct rectangle {
  void move_cb();
};

struct draw_pile_t {

  template <typename T>
  static void global_move_cb(T x, uint32_t src, uint32_t dst);

  global_cb_t<void*, void*> m_global_cb;


  rectangle rectangle;

};

void rectangle::move_cb()  {
  draw_pile_t::global_move_cb(this, 0, 1);
}


template <typename T>
void draw_pile_t::global_move_cb(T x, uint32_t src, uint32_t dst) {
  fan::print(typeid(T).name());
}

